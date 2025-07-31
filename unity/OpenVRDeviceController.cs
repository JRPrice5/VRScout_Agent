using UnityEngine;
using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Collections.Generic;

/// <summary>
/// Unity client for OpenVR virtual devices driver
/// Connects to the IPC server and sends pose data for HMD and controllers
/// </summary>
public class OpenVRDeviceController : MonoBehaviour
{
    [Header("Device Transforms")]
    [Tooltip("Transform that represents the HMD position and rotation")]
    public Transform hmdTransform;
    
    [Tooltip("Transform that represents the left controller position and rotation")]
    public Transform leftControllerTransform;
    
    [Tooltip("Transform that represents the right controller position and rotation")]
    public Transform rightControllerTransform;

    [Header("Connection Settings")]
    [Tooltip("Name of the pipe to connect to")]
    public string pipeName = "VirtualDevicesControlPipe";
    
    [Tooltip("How often to send pose updates (in seconds)")]
    public float updateInterval = 0.01f; // 100Hz update rate
    
    [Tooltip("Maximum frame rate (0 for unlimited)")]
    public int maxFrameRate = 90; // Default to 90 FPS cap

    [Header("Debug")]
    public bool showDebugLogs = true;
    public bool sendTestButtonPresses = false;
    
    [Tooltip("Enable automatic pose updates in Update method. Set to false if another script will handle pose updates.")]
    public bool enableAutomaticUpdates = true;

    // Private variables
    private NamedPipeClientStream pipeClient;
    private StreamWriter pipeWriter;
    private Thread connectionThread;
    private bool isConnected = false;
    private bool shouldConnect = true;
    private float lastUpdateTime = 0f;
    
    // Variables to track current button states
    private bool leftTriggerClick = false;
    private bool leftATouch = false;
    private bool leftAClick = false;
    private bool leftGripClick = false;
    private bool leftSystemClick = false;
    private bool leftMenuClick = false;
    
    private bool rightTriggerClick = false;
    private bool rightATouch = false;
    private bool rightAClick = false;
    private bool rightGripClick = false;
    private bool rightSystemClick = false;
    private bool rightMenuClick = false;
    
    // Variables to track previous button states for change detection
    private bool prevLeftTriggerClick = false;
    private bool prevLeftATouch = false;
    private bool prevLeftAClick = false;
    private bool prevLeftGripClick = false;
    private bool prevLeftSystemClick = false;
    private bool prevLeftMenuClick = false;
    
    private bool prevRightTriggerClick = false;
    private bool prevRightATouch = false;
    private bool prevRightAClick = false;
    private bool prevRightGripClick = false;
    private bool prevRightSystemClick = false;
    private bool prevRightMenuClick = false;
    
    // Controller targeting
    private bool isLeftControllerTarget = true; // Default to left controller
    
    [Header("Input Keys")]
    public KeyCode triggerClickKey = KeyCode.Space;
    public KeyCode switchControllerKey = KeyCode.Tab; // Key to switch between left and right controller

    // Start is called before the first frame update
    void Start()
    {
        // Apply frame rate limit if specified
        if (maxFrameRate > 0)
        {
            Application.targetFrameRate = maxFrameRate;
            LogDebug($"Frame rate limited to {maxFrameRate} FPS");
        }
        
        // Start connection thread
        connectionThread = new Thread(ConnectionThreadFunction);
        connectionThread.IsBackground = true;
        connectionThread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        // Skip update if automatic updates are disabled
        if (!enableAutomaticUpdates) return;
        
        // Check if it's time to send an update
        if (isConnected && Time.time - lastUpdateTime >= updateInterval)
        {
            lastUpdateTime = Time.time;

            // Send HMD pose if available
            if (hmdTransform != null)
            {
                SendHMDPose(hmdTransform, true); // Transform Unity pose to OpenVR pose
            }

            // Send left controller pose if available
            if (leftControllerTransform != null)
            {
                SendControllerPose(leftControllerTransform, true, true); // Transform Unity pose to OpenVR pose
            }

            // Send right controller pose if available
            if (rightControllerTransform != null)
            {
                SendControllerPose(rightControllerTransform, false, true); // Transform Unity pose to OpenVR pose
            }

            // Send test button presses if enabled
            if (sendTestButtonPresses)
            {
                SendTestButtonInputs();
            }
        }
    }

    // Clean up when the script is destroyed
    void OnDestroy()
    {
        // Signal the connection thread to stop
        shouldConnect = false;
        
        // Disconnect from the server
        DisconnectFromServer();
        
        // Wait for the connection thread to finish
        if (connectionThread != null && connectionThread.IsAlive)
        {
            connectionThread.Join(1000); // Wait up to 1 second for the thread to finish
        }
    }

    // Thread function to handle connection and reconnection
    private void ConnectionThreadFunction()
    {
        while (shouldConnect)
        {
            if (!isConnected)
            {
                TryConnect();
            }

            // Sleep to avoid busy waiting
            Thread.Sleep(1000);
        }
    }

    // Try to connect to the named pipe
    private void TryConnect()
    {
        try
        {
            // Create the pipe client
            pipeClient = new NamedPipeClientStream(".", pipeName, PipeDirection.Out);
            
            LogDebug("Attempting to connect to pipe...");
            pipeClient.Connect(1000); // 1 second timeout
            
            // Create writer for sending messages
            pipeWriter = new StreamWriter(pipeClient);
            pipeWriter.AutoFlush = true;
            
            isConnected = true;
            LogDebug("Connected to pipe server successfully!");
        }
        catch (Exception e)
        {
            LogDebug($"Failed to connect to pipe: {e.Message}");
            DisconnectFromServer();
        }
    }

    // Disconnect from the server
    private void DisconnectFromServer()
    {
        isConnected = false;
        
        if (pipeWriter != null)
        {
            pipeWriter.Close();
            pipeWriter = null;
        }
        
        if (pipeClient != null)
        {
            pipeClient.Close();
            pipeClient = null;
        }
    }

    // Send HMD pose to the server
    private void SendHMDPose(Transform hmdTransform, bool transformToOpenVR = false)
    {
        if (!isConnected) return;

        try
        {
            // Get position and rotation
            Vector3 position = hmdTransform.position;
            Quaternion rotation = hmdTransform.rotation;
            
            // Ensure quaternion is valid (not NaN)
            if (!IsQuaternionValid(rotation))
            {
                rotation = Quaternion.identity;
                LogDebug("Invalid HMD quaternion detected, using identity instead");
            }
            
            // Get velocity and angular velocity (if available via Rigidbody)
            Vector3 velocity = Vector3.zero;
            Vector3 angularVelocity = Vector3.zero;
            
            Rigidbody rb = hmdTransform.GetComponent<Rigidbody>();
            if (rb != null)
            {
                velocity = rb.velocity;
                angularVelocity = rb.angularVelocity;
            }
            
            // Transform from Unity to OpenVR coordinate system if requested
            if (transformToOpenVR)
            {
                // Unity uses left-handed, OpenVR uses right-handed coordinate system
                position.z = -position.z;
                rotation.x = -rotation.x;
                rotation.y = -rotation.y;
                velocity.z = -velocity.z;
                angularVelocity.x = -angularVelocity.x;
                angularVelocity.y = -angularVelocity.y;
            }
            
            // Create JSON message
            string json = $"{{\"command\":\"hmd_pose\"," +
                          $"\"pos_x\":{position.x},\"pos_y\":{position.y},\"pos_z\":{position.z}," +
                          $"\"quat_w\":{rotation.w},\"quat_x\":{rotation.x},\"quat_y\":{rotation.y},\"quat_z\":{rotation.z}," +
                          $"\"vel_x\":{velocity.x},\"vel_y\":{velocity.y},\"vel_z\":{velocity.z}," +
                          $"\"ang_vel_x\":{angularVelocity.x},\"ang_vel_y\":{angularVelocity.y},\"ang_vel_z\":{angularVelocity.z}" +
                          $"}}";
            
            // Send message
            pipeWriter.WriteLine(json);
        }
        catch (Exception e)
        {
            LogDebug($"Error sending HMD pose: {e.Message}");
            DisconnectFromServer();
        }
    }

    // Send controller pose to the server
    private void SendControllerPose(Transform controllerTransform, bool isLeft, bool transformToOpenVR = false)
    {
        if (!isConnected) return;

        try
        {
            // Get position and rotation
            Vector3 position = controllerTransform.position;
            Quaternion rotation = controllerTransform.rotation;
            
            // Ensure quaternion is valid (not NaN)
            if (!IsQuaternionValid(rotation))
            {
                rotation = Quaternion.identity;
                LogDebug($"Invalid {(isLeft ? "left" : "right")} controller quaternion detected, using identity instead");
            }
            
            // Get velocity and angular velocity (if available via Rigidbody)
            Vector3 velocity = Vector3.zero;
            Vector3 angularVelocity = Vector3.zero;
            
            Rigidbody rb = controllerTransform.GetComponent<Rigidbody>();
            if (rb != null)
            {
                velocity = rb.velocity;
                angularVelocity = rb.angularVelocity;
            }
            
            // Transform from Unity to OpenVR coordinate system if requested
            if (transformToOpenVR)
            {
                // Unity uses left-handed, OpenVR uses right-handed coordinate system
                position.z = -position.z;
                rotation.x = -rotation.x;
                rotation.y = -rotation.y;
                velocity.z = -velocity.z;
                angularVelocity.x = -angularVelocity.x;
                angularVelocity.y = -angularVelocity.y;
            }
            
            // Create JSON message
            string controllerType = isLeft ? "left_controller_pose" : "right_controller_pose";
            string json = $"{{\"command\":\"{controllerType}\"," +
                          $"\"pos_x\":{position.x},\"pos_y\":{position.y},\"pos_z\":{position.z}," +
                          $"\"quat_w\":{rotation.w},\"quat_x\":{rotation.x},\"quat_y\":{rotation.y},\"quat_z\":{rotation.z}," +
                          $"\"vel_x\":{velocity.x},\"vel_y\":{velocity.y},\"vel_z\":{velocity.z}," +
                          $"\"ang_vel_x\":{angularVelocity.x},\"ang_vel_y\":{angularVelocity.y},\"ang_vel_z\":{angularVelocity.z}" +
                          $"}}";
            
            // Send message
            pipeWriter.WriteLine(json);
        }
        catch (Exception e)
        {
            LogDebug($"Error sending controller pose: {e.Message}");
            DisconnectFromServer();
        }
    }

    // Send controller button state to the server
    private void SendControllerButtonState(
        bool isLeft, 
        bool a_touch = false, bool a_click = false, 
        bool b_touch = false, bool b_click = false,
        bool trigger_touch = false, bool trigger_click = false, 
        bool grip_touch = false, bool grip_click = false,
        bool system_touch = false, bool system_click = false,
        bool joystick_touch = false, bool joystick_click = false,
        bool thumbrest_touch = false)
    {
        if (!isConnected)
        {
            return;
        }

        try
        {
            // Create JSON message
            string controllerType = isLeft ? "left_controller_input" : "right_controller_input";
            
            // Use parameter names that directly match what the IPC server expects
            // Include all button states in every message to ensure consistency
            string json = $"{{\"command\":\"{controllerType}\"," +
                        // A/X button
                        $"\"a_touch\":{a_touch.ToString().ToLower()}," +
                        $"\"a_click\":{a_click.ToString().ToLower()}," +
                        
                        // B/Y button
                        $"\"b_touch\":{b_touch.ToString().ToLower()}," +
                        $"\"b_click\":{b_click.ToString().ToLower()}," +
                        
                        // Trigger
                        $"\"trigger_touch\":{trigger_touch.ToString().ToLower()}," +
                        $"\"trigger_click\":{trigger_click.ToString().ToLower()}," +
                        
                        // Grip
                        $"\"grip_touch\":{grip_touch.ToString().ToLower()}," +
                        $"\"grip_click\":{grip_click.ToString().ToLower()}," +
                        
                        // System button
                        $"\"system_touch\":{system_touch.ToString().ToLower()}," +
                        $"\"system_click\":{system_click.ToString().ToLower()}," +
                        
                        // Joystick
                        $"\"joystick_touch\":{joystick_touch.ToString().ToLower()}," +
                        $"\"joystick_click\":{joystick_click.ToString().ToLower()}," +
                        
                        // Thumbrest
                        $"\"thumbrest_touch\":{thumbrest_touch.ToString().ToLower()}" +
                        $"}}";
            
            // Send message
            pipeWriter.WriteLine(json);
            pipeWriter.Flush(); // Ensure the message is sent immediately
            LogDebug($"Sent button state: {json}");
        }
        catch (Exception e)
        {
            LogDebug($"Error sending controller button state: {e.Message}");
        }
    }

    // Send controller analog values to the server
    private void SendControllerAnalogValues(bool isLeft, float triggerValue, float grip_value)
    {
        if (!isConnected) return;

        try
        {
            // Create JSON message
            string controllerType = isLeft ? "left_controller_analog" : "right_controller_analog";
            
            // The IPC server processes "trigger_value" and "grip_value"
            string json = $"{{\"command\":\"{controllerType}\"," +
                          $"\"trigger_value\":{triggerValue}," +
                          $"\"grip_value\":{grip_value}" +
                          $"}}";
            
            // Send message
            pipeWriter.WriteLine(json);
            pipeWriter.Flush(); // Ensure the message is sent immediately
            LogDebug($"Sent analog values: {json}");
        }
        catch (Exception e)
        {
            LogDebug($"Error sending controller analog values: {e.Message}");
            DisconnectFromServer();
        }
    }
    
    // Send test button inputs (for debugging)
    private void SendTestButtonInputs()
    {
        // Check for controller switch key
        if (Input.GetKeyDown(switchControllerKey))
        {
            isLeftControllerTarget = !isLeftControllerTarget;
            LogDebug($"Switched to {(isLeftControllerTarget ? "LEFT" : "RIGHT")} controller");
        }
        
        // Set button states on KeyDown and reset on KeyUp
        if (isLeftControllerTarget)
        {
            // Handle trigger click
            if (Input.GetKeyDown(triggerClickKey))
            {
                leftTriggerClick = true;
                LogDebug($"LEFT trigger click set to: {leftTriggerClick}");
            }
            else if (Input.GetKeyUp(triggerClickKey))
            {
                leftTriggerClick = false;
                LogDebug($"LEFT trigger click reset to: {leftTriggerClick}");
            }
            
            // Handle other button states
            // A Touch
            if (Input.GetKeyDown(KeyCode.T)) 
            { 
                leftATouch = true; 
                LogDebug($"LEFT A touch set to: {leftATouch}"); 
            }
            else if (Input.GetKeyUp(KeyCode.T)) 
            { 
                leftATouch = false; 
                LogDebug($"LEFT A touch reset to: {leftATouch}"); 
            }
            
            // A Click
            if (Input.GetKeyDown(KeyCode.C)) 
            { 
                leftAClick = true; 
                LogDebug($"LEFT A click set to: {leftAClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.C)) 
            { 
                leftAClick = false; 
                LogDebug($"LEFT A click reset to: {leftAClick}"); 
            }
            
            // Grip Click
            if (Input.GetKeyDown(KeyCode.G)) 
            { 
                leftGripClick = true; 
                LogDebug($"LEFT grip click set to: {leftGripClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.G)) 
            { 
                leftGripClick = false; 
                LogDebug($"LEFT grip click reset to: {leftGripClick}"); 
            }
            
            // System Click
            if (Input.GetKeyDown(KeyCode.S)) 
            { 
                leftSystemClick = true; 
                LogDebug($"LEFT system click set to: {leftSystemClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.S)) 
            { 
                leftSystemClick = false; 
                LogDebug($"LEFT system click reset to: {leftSystemClick}"); 
            }
            
            // Menu Click
            if (Input.GetKeyDown(KeyCode.M)) 
            { 
                leftMenuClick = true; 
                LogDebug($"LEFT menu click set to: {leftMenuClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.M)) 
            { 
                leftMenuClick = false; 
                LogDebug($"LEFT menu click reset to: {leftMenuClick}"); 
            }
            
            // Check if left controller button states have changed from what was previously sent
            bool leftChanged = (leftTriggerClick != prevLeftTriggerClick) ||
                              (leftATouch != prevLeftATouch) ||
                              (leftAClick != prevLeftAClick) ||
                              (leftGripClick != prevLeftGripClick) ||
                              (leftSystemClick != prevLeftSystemClick) ||
                              (leftMenuClick != prevLeftMenuClick);
            
            // Only send button states if they've changed
            if (leftChanged)
            {
                SendControllerButtonState(
                isLeft: true,
                a_touch: leftATouch, a_click: leftAClick,
                b_touch: false, b_click: false,
                trigger_touch: false, trigger_click: leftTriggerClick,
                grip_touch: false, grip_click: leftGripClick,
                system_touch: false, system_click: leftSystemClick,
                joystick_touch: false, joystick_click: false,
                thumbrest_touch: false);
                
                // Update previous states
                prevLeftTriggerClick = leftTriggerClick;
                prevLeftATouch = leftATouch;
                prevLeftAClick = leftAClick;
                prevLeftGripClick = leftGripClick;
                prevLeftSystemClick = leftSystemClick;
                prevLeftMenuClick = leftMenuClick;
            }
        }
        else // Right controller is targeted
        {
            // Handle trigger click
            if (Input.GetKeyDown(triggerClickKey))
            {
                rightTriggerClick = true;
                LogDebug($"RIGHT trigger click set to: {rightTriggerClick}");
            }
            else if (Input.GetKeyUp(triggerClickKey))
            {
                rightTriggerClick = false;
                LogDebug($"RIGHT trigger click reset to: {rightTriggerClick}");
            }
            
            // Handle other button states
            // A Touch
            if (Input.GetKeyDown(KeyCode.T)) 
            { 
                rightATouch = true; 
                LogDebug($"RIGHT A touch set to: {rightATouch}"); 
            }
            else if (Input.GetKeyUp(KeyCode.T)) 
            { 
                rightATouch = false; 
                LogDebug($"RIGHT A touch reset to: {rightATouch}"); 
            }
            
            // A Click
            if (Input.GetKeyDown(KeyCode.C)) 
            { 
                rightAClick = true; 
                LogDebug($"RIGHT A click set to: {rightAClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.C)) 
            { 
                rightAClick = false; 
                LogDebug($"RIGHT A click reset to: {rightAClick}"); 
            }
            
            // Grip Click
            if (Input.GetKeyDown(KeyCode.G)) 
            { 
                rightGripClick = true; 
                LogDebug($"RIGHT grip click set to: {rightGripClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.G)) 
            { 
                rightGripClick = false; 
                LogDebug($"RIGHT grip click reset to: {rightGripClick}"); 
            }
            
            // System Click
            if (Input.GetKeyDown(KeyCode.S)) 
            { 
                rightSystemClick = true; 
                LogDebug($"RIGHT system click set to: {rightSystemClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.S)) 
            { 
                rightSystemClick = false; 
                LogDebug($"RIGHT system click reset to: {rightSystemClick}"); 
            }
            
            // Menu Click
            if (Input.GetKeyDown(KeyCode.M)) 
            { 
                rightMenuClick = true; 
                LogDebug($"RIGHT menu click set to: {rightMenuClick}"); 
            }
            else if (Input.GetKeyUp(KeyCode.M)) 
            { 
                rightMenuClick = false; 
                LogDebug($"RIGHT menu click reset to: {rightMenuClick}"); 
            }
            
            // Check if right controller button states have changed
            bool rightChanged = (rightTriggerClick != prevRightTriggerClick) ||
                               (rightATouch != prevRightATouch) ||
                               (rightAClick != prevRightAClick) ||
                               (rightGripClick != prevRightGripClick) ||
                               (rightSystemClick != prevRightSystemClick) ||
                               (rightMenuClick != prevRightMenuClick);
            
            // Only send button states if they've changed
            if (rightChanged)
            {
                SendControllerButtonState(
                isLeft: false,
                a_touch: rightATouch, a_click: rightAClick,
                b_touch: false, b_click: false,
                trigger_touch: false, trigger_click: rightTriggerClick,
                grip_touch: false, grip_click: rightGripClick,
                system_touch: false, system_click: rightSystemClick,
                joystick_touch: false, joystick_click: false,
                thumbrest_touch: false);
                
                // Update previous states
                prevRightTriggerClick = rightTriggerClick;
                prevRightATouch = rightATouch;
                prevRightAClick = rightAClick;
                prevRightGripClick = rightGripClick;
                prevRightSystemClick = rightSystemClick;
                prevRightMenuClick = rightMenuClick;
            }
        }
        
        // Send analog values
        // SendControllerAnalogValues()
        // Check for ENTER key to simulate trigger value
        if (Input.GetKey(KeyCode.Return))
        {
            // Set trigger value to 1 when ENTER is pressed
            if (isLeftControllerTarget)
            {
                SendControllerAnalogValues(true, 1.0f, 0);
                LogDebug("LEFT trigger value set to 1.0");
            }
            else
            {
                SendControllerAnalogValues(false, 1.0f, 0);
                LogDebug("RIGHT trigger value set to 1.0");
            }
        }
        else if (Input.GetKeyUp(KeyCode.Return))
        {
            // Reset trigger value to 0 when ENTER is released
            if (isLeftControllerTarget)
            {
                SendControllerAnalogValues(true, 0.0f, 0);
                LogDebug("LEFT trigger value reset to 0.0");
            }
            else
            {
                SendControllerAnalogValues(false, 0.0f, 0);
                LogDebug("RIGHT trigger value reset to 0.0");
            }
        }
        
        // Check for grip key to simulate grip value
        if (Input.GetKey(KeyCode.G))
        {
            // Set grip value to 1 when G is pressed
            if (isLeftControllerTarget)
            {
                SendControllerAnalogValues(true, 0, 1.0f);
                LogDebug("LEFT grip value set to 1.0");
            }
            else
            {
                SendControllerAnalogValues(false, 0, 1.0f);
                LogDebug("RIGHT grip value set to 1.0");
            }
        }
        else if (Input.GetKeyUp(KeyCode.G))
        {
            // Reset grip value to 0 when G is released
            if (isLeftControllerTarget)
            {
                SendControllerAnalogValues(true, 0, 0.0f);
                LogDebug("LEFT grip value reset to 0.0");
            }
            else
            {
                SendControllerAnalogValues(false, 0, 0.0f);
                LogDebug("RIGHT grip value reset to 0.0");
            }
        }
    }

    // Log debug message if debug logging is enabled
    private void LogDebug(string message)
    {
        if (showDebugLogs)
        {
            Debug.Log($"[OpenVRDeviceController] {message}");
        }
    }
    
    // Check if a quaternion contains any NaN values
    private bool IsQuaternionValid(Quaternion q)
    {
        // Check if any component is NaN
        if (float.IsNaN(q.x) || float.IsNaN(q.y) || float.IsNaN(q.z) || float.IsNaN(q.w))
        {
            return false;
        }
        
        // Check if quaternion is normalized (has a magnitude close to 1)
        float magnitudeSq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        return !float.IsNaN(magnitudeSq) && Mathf.Abs(magnitudeSq - 1.0f) < 0.01f;
    }

    // Public methods for manually sending button states (can be called from other scripts or UI events)
    
    public void SetTriggerClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            trigger_click: isPressed);
    }
    
    public void SetTriggerTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            trigger_touch: isTouched);
    }
    
    public void SetGripClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            grip_click: isPressed);
    }
    
    public void SetGripTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            grip_touch: isTouched);
    }
    
    public void SetMenuClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        // Menu click maps to B button for right controller and A button for left controller in Oculus Touch
        if (isLeft)
            SendControllerButtonState(isLeft, a_click: isPressed);
        else
            SendControllerButtonState(isLeft, b_click: isPressed);
    }
    
    public void SetSystemClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            system_click: isPressed);
    }
    
    public void SetSystemTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            system_touch: isTouched);
    }
    
    public void SetJoystickTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            joystick_touch: isTouched);
    }
    
    public void SetJoystickClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            joystick_click: isPressed);
    }
    
    public void SetAButtonTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            a_touch: isTouched);
    }
    
    public void SetAButtonClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            a_click: isPressed);
    }
    
    public void SetBButtonTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            b_touch: isTouched);
    }
    
    public void SetBButtonClick(bool isLeft, bool isPressed)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            b_click: isPressed);
    }
    
    public void SetThumbrestTouch(bool isLeft, bool isTouched)
    {
        if (!isConnected) return;
        SendControllerButtonState(isLeft, 
            thumbrest_touch: isTouched);
    }
    
    public void SetTriggerValue(bool isLeft, float value)
    {
        if (!isConnected) return;
        SendControllerAnalogValues(isLeft, value, 0);
    }
    
    public void SetJoystickPosition(bool isLeft, float x, float y)
    {
        if (!isConnected) return;
        // Note: The IPC server currently only processes trigger_value
        // This method is kept for future compatibility if joystick values are supported
        SendControllerAnalogValues(isLeft, x, y);
    }
    
    /// <summary>
    /// Check if connected to the OpenVR driver
    /// </summary>
    /// <returns>True if connected to the driver</returns>
    public bool IsConnectedToDriver()
    {
        return isConnected;
    }
    
    /// <summary>
    /// Apply pose data from PoseReplayFrame to OpenVR devices
    /// </summary>
    /// <param name="frame">The pose frame containing position, rotation, and button states</param>
    public void ApplyPoseToOpenVR(XRT_OVR_Grabber.PoseReplayFrame frame)
    {
        if (!isConnected || frame == null)
        {
            return;
        }
        
        try
        {
            // Apply HMD pose using transform
            hmdTransform.position = frame.hmdPosition;
            hmdTransform.rotation = frame.hmdRotation;
            SendHMDPose(hmdTransform, true); // Transform Unity pose to OpenVR pose
            
            // Apply left controller pose using transform
            leftControllerTransform.position = frame.leftControllerPosition;
            leftControllerTransform.rotation = frame.leftControllerRotation;
            SendControllerPose(leftControllerTransform, true, true); // Transform Unity pose to OpenVR pose
            
            // Apply left controller button states
            // Only send analog values if button states are different from the current state to avoid flooding the pipe
            if (leftTriggerClick != frame.leftTriggerState)
            {
                SendControllerAnalogValues(true, frame.leftTriggerState ? 1.0f : 0.0f, 0);
                leftTriggerClick = frame.leftTriggerState;
            }
            
            if (leftGripClick != frame.leftGripState)
            {
                SendControllerAnalogValues(true, 0, frame.leftGripState ? 1.0f : 0.0f);
                leftGripClick = frame.leftGripState;
            }
            
            // Apply right controller pose using transform
            rightControllerTransform.position = frame.rightControllerPosition;
            rightControllerTransform.rotation = frame.rightControllerRotation;
            SendControllerPose(rightControllerTransform, false, true); // Transform Unity pose to OpenVR pose
            
            // Apply right controller button states
            if (rightTriggerClick != frame.rightTriggerState)
            {
                SendControllerAnalogValues(false, frame.rightTriggerState ? 1.0f : 0.0f, 0);
                rightTriggerClick = frame.rightTriggerState;
            }
            
            if (rightGripClick != frame.rightGripState)
            {
                SendControllerAnalogValues(false, 0, frame.rightGripState ? 1.0f : 0.0f);
                rightGripClick = frame.rightGripState;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[OpenVRDeviceController] Error applying pose to OpenVR: {e.Message}");
        }
    }
}
