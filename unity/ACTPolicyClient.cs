using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using System.IO;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.XR.Interaction.Toolkit;

// Key enum for virtual key simulation - matches the one used in VirtualKeySimulator


/// <summary>
/// Client for communicating with the ACT policy running in Python.
/// This script should be attached to a GameObject in your Unity VR scene.
/// </summary>
public class ACTPolicyClient : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] private string serverIP = "127.0.0.1";
    [SerializeField] private int serverPort = 5555;
    [SerializeField] private float connectionRetryInterval = 5.0f;

    [Header("Camera Settings")]
    [SerializeField] private Camera observationCamera;
    [Tooltip("Desired frame width in pixels.")]
    [SerializeField] private int imageWidth = 224;
    [Tooltip("Desired frame height in pixels.")]
    [SerializeField] private int imageHeight = 224;
    [Tooltip("Fixed aspect ratio for the capture (width/height). Set to 0 to use actual camera aspect ratio.")]
    [SerializeField] private float fixedAspectRatio = 1f;  // Default square aspect ratio (224x224)

    [Header("Update Frequency Settings")]
    [SerializeField] [Range(1, 120)] private int targetFrameRate = 30;
    [SerializeField] [Tooltip("How frequently to send state and apply actions (in seconds)")]
    private float updateInterval = 0.033f; // Default for ~30Hz
    private float updateTimer = 0f;
    private int updatesSinceLastAction = 0; // Counter for updates without action
    [SerializeField] [Tooltip("Number of updates to wait before retrying state send")]
    private int maxUpdatesBetweenSends = 3; // Retry sending state after this many updates

    [Header("VR Components")]
    [SerializeField] private Transform headTransform;
    [SerializeField] private Transform leftControllerTransform;
    [SerializeField] private Transform rightControllerTransform;
    [SerializeField] private Transform xrOriginTransform;

    [Header("Automation Settings")]
    [Tooltip("Set to true to have the policy control VR components")]
    [SerializeField] private bool enableMovementControl = true;
    
    [Tooltip("Set to true to use absolute positions, false to use deltas")]
    [SerializeField] private bool useAbsolutePositions = false;

    [SerializeField] private UnityEngine.XR.Interaction.Toolkit.Samples.StarterAssets.CustomDynamicMoveProvider dynamicMoveProvider;

    [SerializeField] private VirtualKeySimulator keySimulator;
    [SerializeField] private bool enableHeadControl = true;
    [SerializeField] private bool enableControllerPositionControl = true;
    [SerializeField] private bool enableControllerRotationControl = true;
    [SerializeField] private bool enableControllerButtonControl = true;
    [SerializeField] private bool enableXROriginControl = true;
    [SerializeField] private PoseSpace headPoseSpace = PoseSpace.World;
    [SerializeField] private PoseSpace controllerPoseSpace = PoseSpace.Local;
    [SerializeField] private PoseSpace xrOriginPoseSpace = PoseSpace.World;

    // Socket and connection variables
    private TcpClient client;
    private NetworkStream stream;
    private bool isConnected = false;
    private Thread receiveThread;
    private readonly object lockObject = new object();
    private bool shouldReconnect = true;
    
    // Camera original aspect ratio
    private float originalCameraAspect;

    // Action data received from Python
    private float[] latestAction;

    // Timestep synchronization
    private int clientTimestep = 0;
    private Dictionary<int, float[]> actionQueue = new Dictionary<int, float[]>();
    private int maxQueueSize = 10; // Limit queue size to prevent memory issues
    
    // Track controller input values from latest action
    private float leftIndexTrigger = 0f;
    private float rightIndexTrigger = 0f;
    private float leftHandTrigger = 0f;
    private float rightHandTrigger = 0f;
    private float leftThumbstickX = 0f;
    private float leftThumbstickY = 0f;
    private float rightThumbstickX = 0f;
    private float rightThumbstickY = 0f;
    
    // Track previous trigger values for slope calculation
    private float prevLeftHandTrigger = 0f;
    private float prevRightHandTrigger = 0f;
    
    // Track active state of triggers based on slope
    private bool leftHandTriggerActive = false;
    private bool rightHandTriggerActive = false;
    
    // Public properties to access thumbstick values
    public float LeftThumbstickX => leftThumbstickX;
    public float LeftThumbstickY => leftThumbstickY;
    public float RightThumbstickX => rightThumbstickX;
    public float RightThumbstickY => rightThumbstickY;

    // Buffer for receiving data
    private byte[] receiveBuffer = new byte[8192];
    private string dataBuffer = "";

    // Define the TrackedPoseDriver and Action Based Controllers
    private TrackedPoseDriver trackedPoseDriver;
    private ActionBasedController leftABC;
    private ActionBasedController rightABC;

    private void Start()
    {
        // Set update interval based on target frame rate
        updateInterval = 1f / targetFrameRate;
        
        // Disable tracking from device:
        var camera = Camera.main;
        if (camera != null)
        {
            trackedPoseDriver = camera.GetComponent<TrackedPoseDriver>();
            if (trackedPoseDriver != null)
            {
                trackedPoseDriver.enabled = false;
                Debug.Log("[Replayer] Disabled TrackedPoseDriver to allow replay");
            }
        }

        // Init Action Based controllers
        InitializeTriggerControls();

        // Configure DynamicMoveProvider to use our input if available
        if (dynamicMoveProvider != null)
        {
            dynamicMoveProvider.ActPolicyClient = this;
            dynamicMoveProvider.UseActPolicyInput = enableMovementControl;
            Debug.Log($"Set DynamicMoveProvider.UseActPolicyInput = {enableMovementControl}");
        }

        if (observationCamera == null)
        {
            Debug.LogWarning("Observation camera not assigned. Using main camera.");
            observationCamera = Camera.main;
        }
        
        // Store original camera aspect ratio
        originalCameraAspect = observationCamera.aspect;
        
        // If a fixed aspect ratio is specified (not 0), adjust the camera's aspect ratio
        if (fixedAspectRatio > 0)
        {
            observationCamera.aspect = fixedAspectRatio;
            Debug.Log($"[ACT Client] Applied fixed aspect ratio {fixedAspectRatio} to camera");
        }

        // Start connection in a separate thread to avoid blocking the main thread
        Task.Run(() => ConnectToServer());
    }

    private void OnDestroy()
    {
        shouldReconnect = false;
        DisconnectFromServer();

        // Re-enable live inputs afterward
        if (trackedPoseDriver != null)
        {
            trackedPoseDriver.enabled = true;
            Debug.Log("[Replayer] Replay ended - re-enabled TrackedPoseDriver");
        }
        
        // Restore original camera aspect ratio
        if (observationCamera != null && fixedAspectRatio > 0)
        {
            observationCamera.aspect = originalCameraAspect;
            Debug.Log($"[ACT Client] Restored original aspect ratio {originalCameraAspect}");
        }
    }
    
    private void Awake()
    {
        // Find VirtualKeySimulator if not assigned
        if (keySimulator == null)
        {
            keySimulator = FindObjectOfType<VirtualKeySimulator>();
            if (keySimulator == null)
            {
                Debug.LogWarning("No VirtualKeySimulator found in scene. Controller button simulation will be disabled.");
            }
        }
    }

    private void Update()
    {
        // Update timer
        updateTimer += Time.deltaTime;
        
        // Check if it's time to update based on the target frame rate
        if (updateTimer >= updateInterval)
        {
            // Reset timer
            updateTimer = 0f;
            
            bool actionApplied = false;
            lock (lockObject)
            {
                // Check if we have an action for the current timestep in the queue
                if (actionQueue.ContainsKey(clientTimestep))
                {
                    // Choose between delta or absolute position mode based on the toggle
                    if (useAbsolutePositions)
                    {
                        ApplyActionAbsPos(actionQueue[clientTimestep]);
                        Debug.Log($"Applied absolute action for timestep {clientTimestep}");
                    }
                    else
                    {
                        ApplyAction(actionQueue[clientTimestep]);
                        Debug.Log($"Applied delta action for timestep {clientTimestep}");
                    }
                    
                    actionQueue.Remove(clientTimestep); // Remove used action
                    // Increment client timestep
                    clientTimestep++;
                    // Flag that we applied an action this update
                    actionApplied = true;
                }
            }

            if (isConnected)
            {
                // Send state to server when:
                // 1. An action was just applied, OR
                // 2. We've gone maxUpdatesBetweenSends updates without applying an action
                if (actionApplied)
                {
                    // Reset counter when action is applied
                    updatesSinceLastAction = 0;
                    SendStateToServer();
                }
                else
                {
                    // Increment counter when no action is applied
                    updatesSinceLastAction++;
                    
                    // After maxUpdatesBetweenSends updates without an action, retry sending state
                    if (updatesSinceLastAction >= maxUpdatesBetweenSends)
                    {
                        SendStateToServer();
                        Debug.Log($"Retry sending state after {maxUpdatesBetweenSends} updates without action");
                        updatesSinceLastAction = 0; // Reset counter after retry
                    }
                }
            }
        }
    }

    private void ConnectToServer()
    {
        while (shouldReconnect)
        {
            try
            {
                DisconnectFromServer(); // Ensure we're disconnected first

                Debug.Log($"Connecting to ACT policy server at {serverIP}:{serverPort}...");
                client = new TcpClient();
                client.Connect(serverIP, serverPort);
                
                lock (lockObject)
                {
                    stream = client.GetStream();
                    isConnected = true;
                }
                
                Debug.Log("Connected to ACT policy server!");

                // Start receiving data
                receiveThread = new Thread(ReceiveData);
                receiveThread.IsBackground = true;
                receiveThread.Start();

                // Exit the reconnection loop
                break;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Failed to connect to server: {e.Message}");
                Debug.Log($"Retrying in {connectionRetryInterval} seconds...");
                Thread.Sleep((int)(connectionRetryInterval * 1000));
            }
        }
    }

    private void DisconnectFromServer()
    {
        lock (lockObject)
        {
            isConnected = false;
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(1000); // Wait up to 1 second for thread to exit
        }

        if (stream != null)
        {
            stream.Close();
            stream = null;
        }

        if (client != null)
        {
            client.Close();
            client = null;
        }
    }

    private void ReceiveData()
    {
        while (isConnected)
        {
            try
            {
                if (stream.CanRead && client.Available > 0)
                {
                    int bytesRead = stream.Read(receiveBuffer, 0, receiveBuffer.Length);
                    if (bytesRead > 0)
                    {
                        string receivedData = Encoding.UTF8.GetString(receiveBuffer, 0, bytesRead);
                        dataBuffer += receivedData;

                        // Process complete messages
                        ProcessBuffer();
                    }
                }
                else
                {
                    Thread.Sleep(10); // Small sleep to prevent CPU hogging
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Error receiving data: {e.Message}");
                lock (lockObject)
                {
                    isConnected = false;
                }
                
                // Attempt to reconnect
                if (shouldReconnect)
                {
                    Task.Run(() => ConnectToServer());
                }
                
                break;
            }
        }
    }

    private void ProcessBuffer()
    {
        // Check if we have a complete message
        int endIndex = dataBuffer.IndexOf("END_OF_MESSAGE");
        while (endIndex >= 0)
        {
            // Extract the message
            string message = dataBuffer.Substring(0, endIndex);
            
            // Remove the processed message from the buffer
            dataBuffer = dataBuffer.Substring(endIndex + "END_OF_MESSAGE".Length);
            
            // Process the message
            ProcessMessage(message);
            
            // Look for the next message
            endIndex = dataBuffer.IndexOf("END_OF_MESSAGE");
        }
    }

    private void ProcessMessage(string message)
    {
        try
        {
            // Parse the JSON message
            var actionData = JsonUtility.FromJson<ActionData>(message);
            
            lock (lockObject)
            {
                // Add action to queue based on timestep
                if (!actionQueue.ContainsKey(actionData.timestep))
                {
                    actionQueue[actionData.timestep] = actionData.action;
                    
                    // Clean up old entries if queue gets too large
                    if (actionQueue.Count > maxQueueSize)
                    {
                        var oldestTimestep = actionQueue.Keys.Min();
                        actionQueue.Remove(oldestTimestep);
                        Debug.LogWarning($"Removed old action for timestep {oldestTimestep} from queue");
                    }
                }
            }
            
            Debug.Log($"Received action for timestep {actionData.timestep}, current client timestep: {clientTimestep}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing message: {e.Message}");
        }
    }

    private void SendStateToServer()
    {
        try
        {   
            // Create state data
            var stateData = new StateData
            {
                qpos = GetCurrentQPos(),
                image = CaptureAndEncodeImage(),
                timestep = clientTimestep
            };

            // Convert to JSON
            string jsonData = JsonUtility.ToJson(stateData);
            
            // Send data to server
            byte[] data = Encoding.UTF8.GetBytes(jsonData + "END_OF_MESSAGE");
            
            lock (lockObject)
            {
                if (isConnected && stream != null)
                {
                    stream.Write(data, 0, data.Length);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending state data: {e.Message}");
            lock (lockObject)
            {
                isConnected = false;
            }
            
            // Attempt to reconnect
            if (shouldReconnect)
            {
                Task.Run(() => ConnectToServer());
            }
        }
    }

    private float[] GetCurrentQPos()
    {
        // Create array to hold all state variables
        // This should match the format used during training
        List<float> qpos = new List<float>();
        
        // Head position and rotation - respect pose space setting
        Vector3 headPos;
        Quaternion headRot;
        
        if (headPoseSpace == PoseSpace.World)
        {
            headPos = headTransform.position;
            headRot = headTransform.rotation;
        }
        else
        {
            headPos = headTransform.localPosition;
            headRot = headTransform.localRotation;
        }
        
        // Add head position
        qpos.Add(headPos.x);
        qpos.Add(headPos.y);
        qpos.Add(headPos.z);
        
        // Add head rotation (quaternion)
        qpos.Add(headRot.x);
        qpos.Add(headRot.y);
        qpos.Add(headRot.z);
        qpos.Add(headRot.w);
        
        // Left controller position and rotation - respect pose space setting
        Vector3 leftPos;
        Quaternion leftRot;
        
        if (controllerPoseSpace == PoseSpace.World)
        {
            leftPos = leftControllerTransform.position;
            leftRot = leftControllerTransform.rotation;
        }
        else
        {
            leftPos = leftControllerTransform.localPosition;
            leftRot = leftControllerTransform.localRotation;
        }
        
        // Add left controller position
        qpos.Add(leftPos.x);
        qpos.Add(leftPos.y);
        qpos.Add(leftPos.z);
        
        // Add left controller rotation (quaternion)
        qpos.Add(leftRot.x);
        qpos.Add(leftRot.y);
        qpos.Add(leftRot.z);
        qpos.Add(leftRot.w);
        
        // Right controller position and rotation - respect pose space setting
        Vector3 rightPos;
        Quaternion rightRot;
        
        if (controllerPoseSpace == PoseSpace.World)
        {
            rightPos = rightControllerTransform.position;
            rightRot = rightControllerTransform.rotation;
        }
        else
        {
            rightPos = rightControllerTransform.localPosition;
            rightRot = rightControllerTransform.localRotation;
        }
        
        // Add right controller position
        qpos.Add(rightPos.x);
        qpos.Add(rightPos.y);
        qpos.Add(rightPos.z);
        
        // Add right controller rotation (quaternion)
        qpos.Add(rightRot.x);
        qpos.Add(rightRot.y);
        qpos.Add(rightRot.z);
        qpos.Add(rightRot.w);
        
        // XR Origin position - respect pose space setting
        if (xrOriginTransform != null)
        {
            Vector3 xrOriginPos;
            
            if (xrOriginPoseSpace == PoseSpace.World)
            {
                xrOriginPos = xrOriginTransform.position;
            }
            else
            {
                xrOriginPos = xrOriginTransform.localPosition;
            }
            
            // Add XR Origin position
            qpos.Add(xrOriginPos.x);
            qpos.Add(xrOriginPos.y);
            qpos.Add(xrOriginPos.z);
        }
        else
        {
            // Add zeros if XR Origin transform is not available
            qpos.Add(0f);
            qpos.Add(0f);
            qpos.Add(0f);
        }
        
        // Use tracked controller input values from previously applied action
        // This ensures the state includes the current virtual controller state
        qpos.Add(leftIndexTrigger);
        qpos.Add(rightIndexTrigger);
        qpos.Add(leftHandTrigger);
        qpos.Add(rightHandTrigger);
        
        // Add thumbstick values from previous action
        qpos.Add(leftThumbstickX);
        qpos.Add(leftThumbstickY);
        qpos.Add(rightThumbstickX);
        qpos.Add(rightThumbstickY);
        
        return qpos.ToArray();
    }

    private string CaptureAndEncodeImage()
    {
        // Create a render texture
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        RenderTexture prevRT = RenderTexture.active;
        
        // Render the camera to the render texture
        observationCamera.targetTexture = rt;
        observationCamera.Render();
        RenderTexture.active = rt;
        
        // Read the render texture into a texture2D
        Texture2D texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply();
        
        // Clean up
        observationCamera.targetTexture = null;
        RenderTexture.active = prevRT;
        Destroy(rt);
        
        // Encode the texture as a PNG
        byte[] bytes = texture.EncodeToPNG();
        Destroy(texture);
        
        // Convert to base64 string
        return Convert.ToBase64String(bytes);
    }
    
    // Enum for pose space matching the one in Point3Replayer
    public enum PoseSpace { World, Local }    
    private void ApplyAction(float[] action)
    {
        if (action == null || action.Length == 0)
        {
            Debug.LogWarning("Received empty action array");
            return;
        }

        Debug.Log($"Applying action delta: {string.Join(", ", action)}");
        
        // Parse and apply the actions based on our training format
        // Action format from Python:
        // [0-2]: Head position delta, [3-6]: Head rotation delta (quaternion),
        // [7-9]: Left controller position delta, [10-13]: Left controller rotation delta,
        // [14-16]: Right controller position delta, [17-20]: Right controller rotation delta,
        // [21-23]: XR Origin position delta,
        // [24]: Left index trigger, [25]: Right index trigger, [26]: Left hand trigger, [27]: Right hand trigger,
        // [28-29]: Left thumbstick XY, [30-31]: Right thumbstick XY
        
        int index = 0;
        
        // Apply head position/rotation deltas
        if (enableHeadControl && headTransform != null)
        {
            // Extract position and rotation deltas from action
            Vector3 headPosDelta = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion headRotDelta = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            // Get current position and rotation
            Vector3 currentPos;
            Quaternion currentRot;
            
            if (headPoseSpace == PoseSpace.World)
            {
                currentPos = headTransform.position;
                currentRot = headTransform.rotation;
                
                // Apply deltas using SetPositionAndRotation (more efficient)
                headTransform.SetPositionAndRotation(currentPos + headPosDelta, headRotDelta * currentRot);
            }
            else
            {
                currentPos = headTransform.localPosition;
                currentRot = headTransform.localRotation;
                
                // For local transforms, we still need to set properties separately
                headTransform.localPosition = currentPos + headPosDelta;
                headTransform.localRotation = headRotDelta * currentRot;
            }
        }
        else
        {
            // Skip head position/rotation delta indices
            index += 7;
        }
        
        // Apply left controller position/rotation deltas
        if (enableControllerPositionControl && leftControllerTransform != null)
        {
            // Extract position and rotation deltas from action
            Vector3 leftPosDelta = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion leftRotDelta = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            // Get current position and rotation
            Vector3 currentPos;
            Quaternion currentRot;
            
            if (controllerPoseSpace == PoseSpace.World)
            {
                currentPos = leftControllerTransform.position;
                currentRot = leftControllerTransform.rotation;
                
                // Apply position delta
                if (enableControllerRotationControl)
                {
                    // Apply both position and rotation deltas
                    leftControllerTransform.SetPositionAndRotation(currentPos + leftPosDelta, leftRotDelta * currentRot);
                }
                else
                {
                    // Apply position delta only
                    leftControllerTransform.position = currentPos + leftPosDelta;
                }
            }
            else
            {
                currentPos = leftControllerTransform.localPosition;
                currentRot = leftControllerTransform.localRotation;
                
                // For local transforms, we still need to set properties separately
                leftControllerTransform.localPosition = currentPos + leftPosDelta;
                
                // Only apply rotation if rotation control is enabled
                if (enableControllerRotationControl)
                {
                    leftControllerTransform.localRotation = leftRotDelta * currentRot;
                }
            }
        }
        else
        {
            // Skip left controller delta indices
            index += 7;
        }
        
        // Apply right controller position/rotation deltas
        if (enableControllerPositionControl && rightControllerTransform != null)
        {
            // Extract position and rotation deltas from action
            Vector3 rightPosDelta = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion rightRotDelta = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            // Get current position and rotation
            Vector3 currentPos;
            Quaternion currentRot;
            
            if (controllerPoseSpace == PoseSpace.World)
            {
                currentPos = rightControllerTransform.position;
                currentRot = rightControllerTransform.rotation;
                
                // Apply position delta
                if (enableControllerRotationControl)
                {
                    // Apply both position and rotation deltas
                    rightControllerTransform.SetPositionAndRotation(currentPos + rightPosDelta, rightRotDelta * currentRot);
                }
                else
                {
                    // Apply position delta only
                    rightControllerTransform.position = currentPos + rightPosDelta;
                }
            }
            else
            {
                currentPos = rightControllerTransform.localPosition;
                currentRot = rightControllerTransform.localRotation;
                
                // For local transforms, we still need to set properties separately
                rightControllerTransform.localPosition = currentPos + rightPosDelta;
                
                // Only apply rotation if rotation control is enabled
                if (enableControllerRotationControl)
                {
                    rightControllerTransform.localRotation = rightRotDelta * currentRot;
                }
            }
        }
        else
        {
            // Skip right controller delta indices
            index += 7;
        }
        
        // Apply XR Origin position deltas
        if (enableXROriginControl && xrOriginTransform != null)
        {
            // Extract position deltas from action
            Vector3 xrOriginPosDelta = new Vector3(action[index++], action[index++], action[index++]);
            
            // Get current position
            Vector3 currentPos;
            
            if (xrOriginPoseSpace == PoseSpace.World)
            {
                currentPos = xrOriginTransform.position;
                
                // Apply position delta
                xrOriginTransform.position = currentPos + xrOriginPosDelta;
            }
            else
            {
                currentPos = xrOriginTransform.localPosition;
                
                // For local transforms
                xrOriginTransform.localPosition = currentPos + xrOriginPosDelta;
            }
        }
        else
        {
            // Skip XR Origin position delta indices
            index += 3;
        }
        
        // Apply controller button states (these are absolute values from next state, not deltas)
        if (enableControllerButtonControl && keySimulator != null && index + 3 < action.Length)
        {
            // Store controller input values for use in future state collection
            this.leftIndexTrigger = action[index++];
            this.rightIndexTrigger = action[index++];
            this.leftHandTrigger = action[index++];
            this.rightHandTrigger = action[index++];
            
            // Read thumbstick values if available in the action vector
            if (index + 3 < action.Length)
            {
                this.leftThumbstickX = action[index++];
                this.leftThumbstickY = action[index++];
                this.rightThumbstickX = action[index++];
                this.rightThumbstickY = action[index++];
                
                // Log thumbstick status if values exceed threshold
                if (Mathf.Abs(this.leftThumbstickX) > 0.1f || Mathf.Abs(this.leftThumbstickY) > 0.1f)
                {
                    Debug.Log($"Left thumbstick active: X={this.leftThumbstickX:F2}, Y={this.leftThumbstickY:F2}");
                }
                
                if (Mathf.Abs(this.rightThumbstickX) > 0.1f || Mathf.Abs(this.rightThumbstickY) > 0.1f)
                {
                    Debug.Log($"Right thumbstick active: X={this.rightThumbstickX:F2}, Y={this.rightThumbstickY:F2}");
                }
            }
            
            // Simulate key presses based on trigger values
            keySimulator.SetKeyState(Key.Y, this.leftHandTrigger > 0.02f);  // Left grab
            keySimulator.SetKeyState(Key.B, this.rightHandTrigger > 0.02f);  // Right grab

            if (this.leftHandTrigger > 0.02f || this.rightHandTrigger > 0.02f)
            {
                Debug.Log($"HandTrigger active: L={this.leftHandTrigger:F2}, R={this.rightHandTrigger:F2}");
            }
            // You can add additional key mappings for the index triggers if needed
            // For example:
            // keySimulator.SetKeyState(Key.T, this.leftIndexTrigger > 0.1f); 
            // keySimulator.SetKeyState(Key.G, this.rightIndexTrigger > 0.1f);
        }
    }
    
    /// <summary>
    /// Apply action with absolute positions and rotations instead of deltas.
    /// This function is designed to work with the modified dataset format where
    /// actions are the next absolute positions rather than deltas.
    /// </summary>
    /// <param name="action">Array of absolute position and rotation values</param>
    private void ApplyActionAbsPos(float[] action)
    {
        if (action == null || action.Length == 0)
        {
            Debug.LogWarning("Received empty action array");
            return;
        }

        Debug.Log($"Applying absolute action: {string.Join(", ", action)}");
        
        // Parse and apply the actions based on our training format
        // Action format from Python (absolute positions/rotations):
        // [0-2]: Head position, [3-6]: Head rotation (quaternion),
        // [7-9]: Left controller position, [10-13]: Left controller rotation,
        // [14-16]: Right controller position, [17-20]: Right controller rotation,
        // [21-23]: XR Origin position,
        // [24]: Left index trigger, [25]: Right index trigger, [26]: Left hand trigger, [27]: Right hand trigger,
        // [28-29]: Left thumbstick XY, [30-31]: Right thumbstick XY
        
        int index = 0;
        
        // Apply head absolute position/rotation
        if (enableHeadControl && headTransform != null)
        {
            // Extract absolute position and rotation from action
            Vector3 headPos = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion headRot = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            // Apply absolute position and rotation
            if (headPoseSpace == PoseSpace.World)
            {
                // Apply using SetPositionAndRotation (more efficient)
                headTransform.SetPositionAndRotation(headPos, headRot);
            }
            else
            {
                // For local transforms
                headTransform.localPosition = headPos;
                headTransform.localRotation = headRot;
            }
        }
        else
        {
            // Skip head position/rotation indices
            index += 7;
        }
        
        // Apply left controller absolute position/rotation
        if (enableControllerPositionControl && leftControllerTransform != null)
        {
            // Extract absolute position and rotation from action
            Vector3 leftPos = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion leftRot = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            if (controllerPoseSpace == PoseSpace.World)
            {
                if (enableControllerRotationControl)
                {
                    // Apply both position and rotation
                    leftControllerTransform.SetPositionAndRotation(leftPos, leftRot);
                }
                else
                {
                    // Apply position only
                    leftControllerTransform.position = leftPos;
                }
            }
            else
            {
                // For local transforms
                leftControllerTransform.localPosition = leftPos;
                
                // Only apply rotation if rotation control is enabled
                if (enableControllerRotationControl)
                {
                    leftControllerTransform.localRotation = leftRot;
                }
            }
        }
        else
        {
            // Skip left controller indices
            index += 7;
        }
        
        // Apply right controller absolute position/rotation
        if (enableControllerPositionControl && rightControllerTransform != null)
        {
            // Extract absolute position and rotation from action
            Vector3 rightPos = new Vector3(action[index++], action[index++], action[index++]);
            Quaternion rightRot = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
            
            if (controllerPoseSpace == PoseSpace.World)
            {
                if (enableControllerRotationControl)
                {
                    // Apply both position and rotation
                    rightControllerTransform.SetPositionAndRotation(rightPos, rightRot);
                }
                else
                {
                    // Apply position only
                    rightControllerTransform.position = rightPos;
                }
            }
            else
            {
                // For local transforms
                rightControllerTransform.localPosition = rightPos;
                
                // Only apply rotation if rotation control is enabled
                if (enableControllerRotationControl)
                {
                    rightControllerTransform.localRotation = rightRot;
                }
            }
        }
        else
        {
            // Skip right controller indices
            index += 7;
        }
        
        // Apply XR Origin absolute position
        if (enableXROriginControl && xrOriginTransform != null)
        {
            // Extract absolute position from action
            Vector3 xrOriginPos = new Vector3(action[index++], action[index++], action[index++]);
            
            if (xrOriginPoseSpace == PoseSpace.World)
            {
                // Apply absolute position
                xrOriginTransform.position = xrOriginPos;
            }
            else
            {
                // For local transforms
                xrOriginTransform.localPosition = xrOriginPos;
            }
        }
        else
        {
            // Skip XR Origin position indices
            index += 3;
        }
        
        // Apply controller button states (these are absolute values)
        if (enableControllerButtonControl && keySimulator != null && index + 3 < action.Length)
        {
            // Store controller input values for use in future state collection
            this.leftIndexTrigger = action[index++];
            this.rightIndexTrigger = action[index++];
            this.leftHandTrigger = action[index++];
            this.rightHandTrigger = action[index++];
            
            // Read thumbstick values if available in the action vector
            if (index + 3 < action.Length)
            {
                this.leftThumbstickX = action[index++];
                this.leftThumbstickY = action[index++];
                this.rightThumbstickX = action[index++];
                this.rightThumbstickY = action[index++];
                
                // Log thumbstick status if values exceed threshold
                if (Mathf.Abs(this.leftThumbstickX) > 0.1f || Mathf.Abs(this.leftThumbstickY) > 0.1f)
                {
                    Debug.Log($"Left thumbstick active: X={this.leftThumbstickX:F2}, Y={this.leftThumbstickY:F2}");
                }
                
                if (Mathf.Abs(this.rightThumbstickX) > 0.1f || Mathf.Abs(this.rightThumbstickY) > 0.1f)
                {
                    Debug.Log($"Right thumbstick active: X={this.rightThumbstickX:F2}, Y={this.rightThumbstickY:F2}");
                }
            }
            
            // Calculate slopes for trigger values
            float leftHandTriggerSlope = this.leftHandTrigger - this.prevLeftHandTrigger;
            float rightHandTriggerSlope = this.rightHandTrigger - this.prevRightHandTrigger;
            
            // Update trigger active states based on slopes
            // Activate when slope is positive (increasing), deactivate when slope is negative (decreasing)
            // Also ensure a minimum pressure threshold of 0.2 is met
            if (leftHandTriggerSlope > 0.01f && this.leftHandTrigger >= 0.2f) // Small threshold to avoid noise
            {
                leftHandTriggerActive = true;
            }
            else if (leftHandTriggerSlope < -0.01f || this.leftHandTrigger < 0.2f)
            {
                leftHandTriggerActive = false;
            }
            // else maintain previous state when slope is near zero and above minimum threshold
            
            if (rightHandTriggerSlope > 0.01f && this.rightHandTrigger >= 0.2f)
            {
                rightHandTriggerActive = true;
            }
            else if (rightHandTriggerSlope < -0.01f || this.rightHandTrigger < 0.2f)
            {
                rightHandTriggerActive = false;
            }
            // else maintain previous state when slope is near zero and above minimum threshold
            
            // Simulate key presses based on trigger active states
            keySimulator.SetKeyState(Key.Y, leftHandTriggerActive);  // Left grab
            keySimulator.SetKeyState(Key.B, rightHandTriggerActive);  // Right grab
            
            // Store current values as previous for next frame
            this.prevLeftHandTrigger = this.leftHandTrigger;
            this.prevRightHandTrigger = this.rightHandTrigger;

            if (this.leftHandTrigger > 0.1f || this.rightHandTrigger > 0.1f)
            {
                Debug.Log($"HandTrigger active: L={this.leftHandTrigger:F2}, R={this.rightHandTrigger:F2}");
            }
            // You can add additional key mappings for the index triggers if needed
            // For example:
            // keySimulator.SetKeyState(Key.T, this.leftIndexTrigger > 0.1f); 
            // keySimulator.SetKeyState(Key.G, this.rightIndexTrigger > 0.1f);
        }
    }

    // Helper classes for JSON serialization
    [Serializable]
    private class StateData
    {
        public float[] qpos;
        public string image;
        public int timestep; // Added timestep field for synchronization
    }

    [Serializable]
    private class ActionData
    {
        public float[] action;
        public int timestep;
    }

    // Helper function to activate controller objects
    void InitializeTriggerControls()
    {
        // LEFT
        leftABC = leftControllerTransform.GetComponent<ActionBasedController>();
        rightABC = rightControllerTransform.GetComponent<ActionBasedController>();

        leftABC.gameObject.SetActive(true);
        rightABC.gameObject.SetActive(true);
    }
}
