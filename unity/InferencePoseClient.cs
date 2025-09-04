using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using System.IO;
using Valve.VR;
using UnityEngine.Rendering;
using System.Diagnostics; // For Stopwatch
using Unity.Collections;

namespace CLOVR.IPCClient
{
    /// <summary>
    /// Client for communicating with a Python inference server.
    /// Sends real-time poses and left eye images to the server and applies received pose actions to virtual OpenVR devices.
    /// </summary>
    public class InferencePoseClient : MonoBehaviour
    {
        // Dynamic window tracking
        private int currentWindowSize = 0;
        private string currentDecayFactor = "0.0000";
        private float currentInferenceUpdateRate = 0.02f;
        
        // Dynamic window component toggle states
        private bool enableDynamicWindowSize = true;
        private bool enableDynamicDecayFactor = true;
        private bool enableDynamicUpdateRate = true;
        
        // GUI toggle states (for immediate feedback)
        private bool guiWindowSizeToggle = true;
        private bool guiDecayFactorToggle = true;
        private bool guiUpdateRateToggle = true;
        
        // Performance tracking
        private Dictionary<string, float> timingStats = new Dictionary<string, float>();
        private Dictionary<string, int> timingCounts = new Dictionary<string, int>();
        // Dictionary to track active timing operations
        private Dictionary<string, Stopwatch> activeTimers = new Dictionary<string, Stopwatch>();
        [Header("Performance Tracking")]
        [SerializeField] private bool enablePerformanceLogging = true;
        [SerializeField] private float performanceLogInterval = 5.0f;
        private float performanceLogTimer = 0f;
        
        /// <summary>
        /// Start timing a specific operation
        /// </summary>
        private void StartTiming(string operation)
        {
            if (!enablePerformanceLogging) return;
            
            // Create a new stopwatch for this specific operation
            Stopwatch operationStopwatch = new Stopwatch();
            operationStopwatch.Start();
            
            // Use thread-safe dictionary access
            lock (activeTimers)
            {
                // If there's already a timer for this operation (nested calls with same name),
                // we'll replace it, effectively measuring only the outermost call
                activeTimers[operation] = operationStopwatch;
            }
        }
        
        /// <summary>
        /// End timing for an operation and record the result
        /// </summary>
        private void EndTiming(string operation)
        {
            if (!enablePerformanceLogging) return;
            
            Stopwatch operationStopwatch = null;
            
            // Use thread-safe dictionary access
            lock (activeTimers)
            {
                // Check if we have a stopwatch for this operation
                if (activeTimers.TryGetValue(operation, out operationStopwatch))
                {
                    activeTimers.Remove(operation);
                }
                else
                {
                    // No stopwatch found for this operation
                    UnityEngine.Debug.LogWarning($"[PERF] EndTiming called for '{operation}' without matching StartTiming");
                    return;
                }
            }
            
            // Stop the stopwatch and record the time
            operationStopwatch.Stop();
            float elapsedMs = operationStopwatch.ElapsedMilliseconds;
            
            lock (timingStats)
            {
                if (!timingStats.ContainsKey(operation))
                {
                    timingStats[operation] = elapsedMs;
                    timingCounts[operation] = 1;
                }
                else
                {
                    // Running average
                    timingStats[operation] = (timingStats[operation] * timingCounts[operation] + elapsedMs) / (timingCounts[operation] + 1);
                    timingCounts[operation]++;
                }
            }
            
            // Log individual timing for immediate feedback
            UnityEngine.Debug.Log($"[PERF] {operation}: {elapsedMs:F2}ms");
        }
        
        /// <summary>
        /// Log all performance statistics
        /// </summary>
        private void LogPerformanceStats()
        {
            if (!enablePerformanceLogging || timingStats.Count == 0) return;
            
            StringBuilder sb = new StringBuilder("\n===== PERFORMANCE STATISTICS =====\n");
            foreach (var kvp in timingStats)
            {
                sb.AppendLine($"{kvp.Key}: {kvp.Value:F2}ms (avg of {timingCounts[kvp.Key]} samples)");
            }
            sb.AppendLine("================================");
            
            UnityEngine.Debug.Log(sb.ToString());
        }

        [Header("Network Settings")]
        [SerializeField] private string serverIP = "127.0.0.1";
        [SerializeField] private int serverPort = 5556;
        [SerializeField] private float connectionRetryInterval = 5.0f;
        [SerializeField, Range(1, 100), Tooltip("JPEG encoding quality (1-100). Higher values mean better quality but larger file size.")]
        private int jpgQuality = 75;

        [Header("Texture Processing")]
        public ComputeShader flipTextureShader;

        [Header("Eye Texture Settings")]
        [Tooltip("Desired frame width in pixels.")]
        [SerializeField] private int imageWidth = 224;
        [Tooltip("Desired frame height in pixels.")]
        [SerializeField] private int imageHeight = 224;
        [Tooltip("Enable texture compression to reduce network traffic")]
        [SerializeField] private bool enableTextureCompression = true;
        [Tooltip("Use AsyncGPUReadback for more efficient texture capture")]
        [SerializeField] private bool useAsyncGPUReadback = true;
        [Tooltip("Use asynchronous coroutines for non-blocking texture processing")]
        [SerializeField] private bool useAsyncProcessing = true;

        // OpenVR system reference
        private CVRSystem _vrSystem;

        [Header("Update Frequency Settings")]
        // Auto update rate control
        [SerializeField] private bool useAutoUpdateRate = true;
        [SerializeField] [Tooltip("How frequently to send state and apply actions (in seconds)")]
        private float updateInterval = 0.033f; // Default for ~30Hz
        private float updateTimer = 0f;
        private int updatesSinceLastAction = 0; // Counter for updates without action
        [SerializeField] [Tooltip("Number of updates to wait before retrying state send")]
        private int maxUpdatesBetweenSends = 3; // Retry sending state after this many updates

        [Header("OpenVR Device Controller")]
        [SerializeField] private OpenVRDeviceController deviceController;

        [Header("VR Components")]
        [SerializeField] private Transform headTransform;
        [SerializeField] private Transform leftControllerTransform;
        [SerializeField] private Transform rightControllerTransform;

        [Header("Button triggering Threshold")]
        [SerializeField] private float pressThreshold = 0.5f;
        [SerializeField] private float releaseThreshold = 0.4f;

        [Header("Communication Timeout Auto-Stop switch")]
        [SerializeField] private bool autoStop = true;

        // Socket and connection variables
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected = false;
        private Thread receiveThread;
        private readonly object lockObject = new object();
        private bool shouldReconnect = true;

        // Public properties to expose connection objects
        public NetworkStream NetworkStream => stream;
        public TcpClient TcpClient => client;
        public string ServerIP => serverIP;
        public int ServerPort => serverPort;

        // Flag if any Action data received from Python
        private bool actionReceived = false;

        // Timestep synchronization
        private int clientTimestep = 0;
        private Dictionary<int, float[]> actionQueue = new Dictionary<int, float[]>();
        private int maxQueueSize = 10; // Limit queue size to prevent memory issues
        // Track the latest timestep present in the action queue for O(1) retrieval
        private int latestTimestepInQueue = int.MinValue;

        // Status properties
        public bool IsConnected 
        { 
            get { return isConnected; } 
            set 
            { 
                shouldReconnect = value;
                isConnected = value;
                actionReceived = false;
            }
        }
        public int CurrentTimestep => clientTimestep;
        public int QueuedActions => actionQueue.Count;
        
        // Controller input values (stored from received actions)
        private float leftIndexTrigger = 0f;
        private float rightIndexTrigger = 0f;
        private float leftHandTrigger = 0f;
        private float rightHandTrigger = 0f;
        private float leftThumbstickX = 0f;
        private float leftThumbstickY = 0f;
        private float rightThumbstickX = 0f;
        private float rightThumbstickY = 0f;
        
        // Previous controller input states (for trigger and grip)
        private bool prevLeftTriggerState = false;
        private bool prevRightTriggerState = false;
        private bool prevLeftGripState = false;
        private bool prevRightGripState = false;
        
        // Latest HMD and controller positions and rotations (stored from received actions)
        private Vector3 latestHmdPosition = Vector3.zero;
        private Quaternion latestHmdRotation = Quaternion.identity;
        private Vector3 latestLeftControllerPosition = Vector3.zero;
        private Quaternion latestLeftControllerRotation = Quaternion.identity;
        private Vector3 latestRightControllerPosition = Vector3.zero;
        private Quaternion latestRightControllerRotation = Quaternion.identity;
        private bool useLatestPoseValues = false;
        
        // Enhanced texture pooling for better performance
        private Dictionary<string, RenderTexture> renderTexturePool = new Dictionary<string, RenderTexture>();
        private Dictionary<string, Texture2D> texture2DPool = new Dictionary<string, Texture2D>();
        private Dictionary<string, ComputeBuffer> computeBufferPool = new Dictionary<string, ComputeBuffer>();
        private int lastTextureWidth;
        private int lastTextureHeight;
        private int lastOutputWidth;
        private int lastOutputHeight;
        
        // Maximum number of textures to keep in each pool
        private const int MAX_POOL_SIZE = 3;

        private void Start()
        {
            // Initialize OpenVR
            if (!OpenVR.IsRuntimeInstalled())
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] SteamVR runtime not installed!");
                return;
            }

            // Initialize OpenVR with appropriate application type
            EVRInitError initError = EVRInitError.None;
            _vrSystem = OpenVR.Init(ref initError, EVRApplicationType.VRApplication_Other);
            
            if (initError != EVRInitError.None)
            {
                string errorMsg = OpenVR.GetStringForHmdError(initError);
                UnityEngine.Debug.LogError($"[InferencePoseClient] Failed to initialize OpenVR: {errorMsg} ({initError})");
                _vrSystem = null;
                return;
            }
            else if (_vrSystem == null)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] OpenVR system not available after initialization!");
                return;
            }
            else
            {
                UnityEngine.Debug.Log("[InferencePoseClient] OpenVR initialized successfully");
            }

            if (deviceController == null || deviceController.IsConnectedToDriver() == false)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] OpenVRDeviceController not assigned or not connected to driver!");
            }

            if (headTransform == null || leftControllerTransform == null || rightControllerTransform == null)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] One or more required transforms not assigned!");
            }

            // Start connection in a separate thread
            // Task.Run(() => ConnectToServer());
        }

        private void OnDestroy()
        {
            // Clean up resources
            shouldReconnect = false;
            DisconnectFromServer();
            
            // Shutdown OpenVR if we initialized it
            if (_vrSystem != null)
            {
                UnityEngine.Debug.Log("[InferencePoseClient] Shutting down OpenVR");
                OpenVR.Shutdown();
                _vrSystem = null;
            }
            
            // Clean up texture pool resources
            CleanupTexturePool();
        }

        /// <summary>
        /// Clean up all texture resources in the pool
        /// </summary>
        private void CleanupTexturePool()
        {
            // Clean up all RenderTextures in the pool
            if (renderTexturePool != null)
            {
                foreach (var rt in renderTexturePool.Values)
                {
                    if (rt != null)
                    {
                        rt.Release();
                        Destroy(rt);
                    }
                }
                renderTexturePool.Clear();
            }
            
            // Clean up all Texture2Ds in the pool
            if (texture2DPool != null)
            {
                foreach (var tex in texture2DPool.Values)
                {
                    if (tex != null)
                    {
                        Destroy(tex);
                    }
                }
                texture2DPool.Clear();
            }
            
            // Clean up all ComputeBuffers in the pool
            if (computeBufferPool != null)
            {
                foreach (var buffer in computeBufferPool.Values)
                {
                    if (buffer != null)
                    {
                        buffer.Release();
                    }
                }
                computeBufferPool.Clear();
            }
        }

        private void Update()
        {
            StartTiming("Update_Total");
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
                    // Apply the latest available action (highest timestep) using tracked max
                    if (actionQueue.Count > 0 && latestTimestepInQueue != int.MinValue)
                    {
                        int latestTimestep = latestTimestepInQueue;

                        ApplyAction(actionQueue[latestTimestep]);
                        UnityEngine.Debug.Log($"Applied latest action for timestep {latestTimestep}");

                        // Drop all queued (older) actions as they are now stale
                        actionQueue.Clear();
                        latestTimestepInQueue = int.MinValue;

                        // Advance client timestep to one past the applied action
                        clientTimestep = latestTimestep + 1;
                        // Flag that we applied an action this update
                        actionApplied = true;
                        actionReceived = true;
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
                        
                        // Use either synchronous or asynchronous processing based on settings
                        if (useAsyncProcessing)
                        {
                            StartCoroutine(SendStateToServerAsync());
                        }
                        else
                        {
                            SendStateToServer();
                        }
                    }
                    else
                    {
                        // Increment counter when no action is applied
                        updatesSinceLastAction++;
                        
                        // After maxUpdatesBetweenSends updates without an action, retry sending state
                        if (updatesSinceLastAction >= maxUpdatesBetweenSends)
                        {
                            // If autoStop is enabled, set isConnected to false to stop sending state
                            if (autoStop && actionReceived)
                            {
                                shouldReconnect = false;
                                isConnected = false;
                                actionReceived = false;
                            }

                            // Use either synchronous or asynchronous processing based on settings
                            if (useAsyncProcessing)
                            {
                                StartCoroutine(SendStateToServerAsync());
                            }
                            else
                            {
                                SendStateToServer();
                            }
                            UnityEngine.Debug.Log($"IsConnected: {isConnected}");
                            UnityEngine.Debug.Log($"Retry sending state after {maxUpdatesBetweenSends} updates without action");
                            updatesSinceLastAction = 0; // Reset counter after retry
                        }
                    }
                }
            }
            EndTiming("Update_Total");
        }

        private void ConnectToServer()
        {
            StartTiming("ConnectToServer_Total");
            while (shouldReconnect)
            {
                try
                {
                    DisconnectFromServer(); // Ensure we're disconnected first

                    UnityEngine.Debug.Log($"Connecting to inference server at {serverIP}:{serverPort}...");
                    client = new TcpClient();
                    client.Connect(serverIP, serverPort);
                    
                    lock (lockObject)
                    {
                        stream = client.GetStream();
                        isConnected = true;
                        actionReceived = false;
                    }
                    
                    UnityEngine.Debug.Log("Connected to inference server!");

                    // Start receiving data
                    receiveThread = new Thread(ReceiveData);
                    receiveThread.IsBackground = true;
                    receiveThread.Start();

                    // Exit the reconnection loop
                    break;
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Failed to connect to server: {e.Message}");
                    UnityEngine.Debug.Log($"Retrying in {connectionRetryInterval} seconds...");
                    Thread.Sleep((int)(connectionRetryInterval * 1000));
                }
            }
            EndTiming("ConnectToServer_Total");
        }

        private void DisconnectFromServer()
        {
            StartTiming("DisconnectFromServer_Total");
            lock (lockObject)
            {
                isConnected = false;
            }
            
            // Stop and clean up receive thread
            if (receiveThread != null && receiveThread.IsAlive)
            {
                try
                {
                    receiveThread.Join(1000); // Wait up to 1 second for thread to finish
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Error stopping receive thread: {e.Message}");
                }
                receiveThread = null;
            }
            
            // Close network stream
            if (stream != null)
            {
                try
                {
                    stream.Close();
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Error closing network stream: {e.Message}");
                }
                stream = null;
            }
            
            // Close client
            if (client != null)
            {
                try
                {
                    client.Close();
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Error closing client: {e.Message}");
                }
                client = null;
            }
            
            UnityEngine.Debug.Log("Disconnected from server");
            EndTiming("DisconnectFromServer_Total");
        }

        private void ReceiveData()
        {
            StartTiming("ReceiveData_Total");
            byte[] buffer = new byte[4096];
            StringBuilder messageBuilder = new StringBuilder();
            
            while (isConnected)
            {
                try
                {
                    lock (lockObject)
                    {
                        if (!isConnected || stream == null)
                        {
                            break;
                        }
                    }
                    
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 0)
                    {
                        // Connection closed
                        UnityEngine.Debug.Log("Server closed connection");
                        break;
                    }
                    
                    // Convert bytes to string
                    string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    messageBuilder.Append(data);
                    
                    // Process complete messages
                    ProcessBuffer(messageBuilder);
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogError($"Error receiving data: {e.Message}");
                    break;
                }
            }
            
            // Connection lost, attempt to reconnect
            lock (lockObject)
            {
                isConnected = false;
            }
            
            if (shouldReconnect)
            {
                UnityEngine.Debug.Log("Connection lost. Attempting to reconnect...");
                Task.Run(() => ConnectToServer());
            }
            EndTiming("ReceiveData_Total");
        }

        private void ProcessBuffer(StringBuilder buffer)
        {
            string bufferStr = buffer.ToString();
            int endIndex;
            
            // Process all complete messages in the buffer
            while ((endIndex = bufferStr.IndexOf("END_OF_MESSAGE")) != -1)
            {
                // Extract the complete message
                string message = bufferStr.Substring(0, endIndex);
                
                // Remove the processed message from the buffer
                buffer.Remove(0, endIndex + "END_OF_MESSAGE".Length);
                bufferStr = buffer.ToString();
                
                // Process the message
                ProcessMessage(message);
            }
        }

        private void ProcessMessage(string message)
        {
            StartTiming("ProcessMessage");
            try
            {
                // Parse the JSON message
                ActionData actionData = JsonUtility.FromJson<ActionData>(message);
                
                if (actionData != null && actionData.action != null)
                {
                    // Update window size and decay factor tracking
                    currentWindowSize = actionData.window_size;
                    currentDecayFactor = actionData.decay_factor ?? "0.0000";
                    
                    // Update toggle states from server
                    enableDynamicWindowSize = actionData.enable_dynamic_window_size;
                    enableDynamicDecayFactor = actionData.enable_dynamic_decay_factor;
                    enableDynamicUpdateRate = actionData.enable_dynamic_update_rate;
                    
                    // Sync GUI toggles with server state
                    guiWindowSizeToggle = enableDynamicWindowSize;
                    guiDecayFactorToggle = enableDynamicDecayFactor;
                    guiUpdateRateToggle = enableDynamicUpdateRate;
                    
                    // Update inference update rate and apply if auto-adjustment is enabled
                    if (float.TryParse(actionData.stepped_inference_update_rate ?? "0.02", out float newUpdateRate))
                    {
                        currentInferenceUpdateRate = newUpdateRate;
                        if (useAutoUpdateRate)
                        {
                            updateInterval = currentInferenceUpdateRate;
                        }
                    }
                    
                    lock (lockObject)
                    {
                        // Add the action to the queue with its timestep
                        actionQueue[actionData.timestep] = actionData.action;
                        UnityEngine.Debug.Log($"Added action for timestep {actionData.timestep}");
                        // Update tracked latest timestep
                        if (actionData.timestep > latestTimestepInQueue)
                        {
                            latestTimestepInQueue = actionData.timestep;
                        }
                        
                        // Limit queue size by removing oldest entries if needed
                        if (actionQueue.Count > maxQueueSize)
                        {
                            int oldestTimestep = int.MaxValue;
                            foreach (int timestep in actionQueue.Keys)
                            {
                                if (timestep < oldestTimestep)
                                {
                                    oldestTimestep = timestep;
                                }
                            }
                            actionQueue.Remove(oldestTimestep); // TODO: see if we can use more efficient removal
                        }
                    }
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"Error processing message: {e.Message}");
            }
            EndTiming("ProcessMessage");
        }

        private void SendStateToServer()
        {
            StartTiming("SendStateToServer_Total");
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
                UnityEngine.Debug.LogError($"Error sending state data: {e.Message}");
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
            EndTiming("SendStateToServer_Total");
        }

        private IEnumerator SendStateToServerAsync()
        {
            StartTiming("SendStateToServerAsync_Total");
            // Get the current qpos and timestep
            float[] qpos = null;
            int currentTimestep = 0;
            
            try
            {   
                StartTiming("GetCurrentQPos");
                qpos = GetCurrentQPos();
                EndTiming("GetCurrentQPos");
                
                lock (lockObject)
                {
                    currentTimestep = clientTimestep;
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"Error getting current state: {e.Message}");
                HandleConnectionError();
                yield break;
            }
            
            // Capture and encode the image asynchronously
            string base64Image = "";
            bool imageProcessingComplete = false;
            int textureWidth = imageWidth > 0 ? imageWidth : 224; // Default width if not set
            int textureHeight = imageHeight > 0 ? imageHeight : 224; // Default height if not set
            
            // Start the asynchronous image capture and processing
            StartTiming("CaptureAndEncodeImageAsync");
            yield return StartCoroutine(CaptureAndEncodeImageAsync((result) => {
                base64Image = result;
                imageProcessingComplete = true;
                EndTiming("CaptureAndEncodeImageAsync");
            }));
            
            // Wait for the image processing to complete
            while (!imageProcessingComplete)
            {
                yield return null;
            }
            
            try
            {
                // Create state data with texture dimensions and format
                StartTiming("CreateStateData_and_JSON");
                var stateData = new StateData
                {
                    qpos = qpos,
                    image = base64Image,
                    timestep = currentTimestep,
                    imageWidth = textureWidth,
                    imageHeight = textureHeight,
                    imageFormat = "jpg", // Now using JPG encoding instead of raw data
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
                EndTiming("CreateStateData_and_JSON");
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"Error sending state data asynchronously: {e.Message}");
                HandleConnectionError();
            }
            EndTiming("SendStateToServerAsync_Total");
        }

        /// <summary>
        /// Helper method to handle connection errors
        /// </summary>
        private void HandleConnectionError()
        {
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

        private float[] GetCurrentQPos()
        {
            StartTiming("GetCurrentQPos");
            // Create array to hold all state variables
            // This should match the format used during training
            List<float> qpos = new List<float>();
            
            if (useLatestPoseValues)
            {
                // Use the latest action values directly instead of getting from transform objects
                
                // Add head position
                qpos.Add(latestHmdPosition.x);
                qpos.Add(latestHmdPosition.y);
                qpos.Add(latestHmdPosition.z);
                
                // Add head rotation (quaternion)
                qpos.Add(latestHmdRotation.x);
                qpos.Add(latestHmdRotation.y);
                qpos.Add(latestHmdRotation.z);
                qpos.Add(latestHmdRotation.w);
                
                // Add left controller position
                qpos.Add(latestLeftControllerPosition.x);
                qpos.Add(latestLeftControllerPosition.y);
                qpos.Add(latestLeftControllerPosition.z);
                
                // Add left controller rotation (quaternion)
                qpos.Add(latestLeftControllerRotation.x);
                qpos.Add(latestLeftControllerRotation.y);
                qpos.Add(latestLeftControllerRotation.z);
                qpos.Add(latestLeftControllerRotation.w);
                
                // Add right controller position
                qpos.Add(latestRightControllerPosition.x);
                qpos.Add(latestRightControllerPosition.y);
                qpos.Add(latestRightControllerPosition.z);
                
                // Add right controller rotation (quaternion)
                qpos.Add(latestRightControllerRotation.x);
                qpos.Add(latestRightControllerRotation.y);
                qpos.Add(latestRightControllerRotation.z);
                qpos.Add(latestRightControllerRotation.w);
            }
            else
            {
                // Fall back to transform objects if no action has been received yet
                
                // Add head position
                qpos.Add(headTransform.position.x);
                qpos.Add(headTransform.position.y);
                qpos.Add(headTransform.position.z);
                
                // Add head rotation (quaternion)
                qpos.Add(headTransform.rotation.x);
                qpos.Add(headTransform.rotation.y);
                qpos.Add(headTransform.rotation.z);
                qpos.Add(headTransform.rotation.w);
                
                // Add left controller position
                qpos.Add(leftControllerTransform.position.x);
                qpos.Add(leftControllerTransform.position.y);
                qpos.Add(leftControllerTransform.position.z);
                
                // Add left controller rotation (quaternion)
                qpos.Add(leftControllerTransform.rotation.x);
                qpos.Add(leftControllerTransform.rotation.y);
                qpos.Add(leftControllerTransform.rotation.z);
                qpos.Add(leftControllerTransform.rotation.w);
                
                // Add right controller position
                qpos.Add(rightControllerTransform.position.x);
                qpos.Add(rightControllerTransform.position.y);
                qpos.Add(rightControllerTransform.position.z);
                
                // Add right controller rotation (quaternion)
                qpos.Add(rightControllerTransform.rotation.x);
                qpos.Add(rightControllerTransform.rotation.y);
                qpos.Add(rightControllerTransform.rotation.z);
                qpos.Add(rightControllerTransform.rotation.w);
            }
            
            // Add XR Origin position - defaulting to 0f as we don't have Origin data
            qpos.Add(0f);
            qpos.Add(0f);
            qpos.Add(0f);
            
            // Use stored controller input values from previously received actions
            
            // Add trigger values
            qpos.Add(leftIndexTrigger);
            qpos.Add(rightIndexTrigger);
            qpos.Add(leftHandTrigger);
            qpos.Add(rightHandTrigger);
            
            // Add thumbstick values
            qpos.Add(leftThumbstickX);
            qpos.Add(leftThumbstickY);
            qpos.Add(rightThumbstickX);
            qpos.Add(rightThumbstickY);
            
            EndTiming("GetCurrentQPos");
            return qpos.ToArray();
        }

        /// <summary>
        /// Captures and encodes the left eye image synchronously
        /// </summary>
        private string CaptureAndEncodeImage()
        {
            StartTiming("CaptureAndEncodeImage");
            if (_vrSystem == null)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] OpenVR system not available!");
                EndTiming("CaptureAndEncodeImage");
                return "";
            }

            try
            {
                // Get the left eye texture directly from OpenVR
                Texture2D eyeTexture = CollectTextureFromEye(EVREye.Eye_Left);
                
                if (eyeTexture == null)
                {
                    UnityEngine.Debug.LogError("[InferencePoseClient] Failed to collect left eye texture!");
                    EndTiming("CaptureAndEncodeImage");
                    return "";
                }
                
                // Process the texture (compress if needed) and encode it
                if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback)
                {
                    // Use AsyncGPUReadback for more efficient texture processing
                    // For synchronous operation in this method, we'll use a simple approach
                    // that mimics the async pattern but blocks until complete
                    byte[] encodedBytes = ProcessTextureWithGPUReadback(eyeTexture);
                    if (encodedBytes != null && encodedBytes.Length > 0)
                    {
                        Destroy(eyeTexture);
                        EndTiming("CaptureAndEncodeImage");
                        return Convert.ToBase64String(encodedBytes);
                    }
                }
                
                // Fallback to direct texture processing
                byte[] bytes = ProcessTextureDirectly(eyeTexture);
                Destroy(eyeTexture);
                
                if (bytes != null && bytes.Length > 0)
                {
                    EndTiming("CaptureAndEncodeImage");
                    return Convert.ToBase64String(bytes);
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error capturing eye texture: {e.Message}");
            }
            
            EndTiming("CaptureAndEncodeImage");
            return "";
        }

        /// <summary>
        /// Captures and encodes the left eye image asynchronously using coroutines
        /// </summary>
        private IEnumerator CaptureAndEncodeImageAsync(Action<string> onComplete)
        {
            StartTiming("CaptureAndEncodeImageAsync_Internal");
            if (_vrSystem == null)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] OpenVR system not available!");
                onComplete?.Invoke("");
                yield break;
            }

            Texture2D eyeTexture = null;
            
            try
            {
                // Get the left eye texture directly from OpenVR
                StartTiming("CollectTextureFromEye");
                eyeTexture = CollectTextureFromEye(EVREye.Eye_Left);
                EndTiming("CollectTextureFromEye");
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error collecting eye texture: {e.Message}");
                onComplete?.Invoke("");
                yield break;
            }
            
            if (eyeTexture == null)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] Failed to collect left eye texture!");
                onComplete?.Invoke("");
                yield break;
            }
            
            // Process the texture asynchronously if supported
            if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback)
            {
                // Start the asynchronous processing
                StartTiming("ProcessTextureWithGPUReadbackAsync");
                yield return StartCoroutine(ProcessTextureWithGPUReadbackAsync(eyeTexture, (bytes) => {
                    EndTiming("ProcessTextureWithGPUReadbackAsync");
                    // This callback will be executed when the GPU readback is complete
                    if (bytes != null && bytes.Length > 0)
                    {
                        string base64 = Convert.ToBase64String(bytes);
                        onComplete?.Invoke(base64);
                    }
                    else
                    {
                        onComplete?.Invoke("");
                    }
                }));
                EndTiming("CaptureAndEncodeImageAsync_Internal");
                // The callback will handle the completion, so we're done here
                yield break;
            }
            
            try
            {
                // Fallback to direct texture processing
                StartTiming("ProcessTextureDirectly");
                byte[] directBytes = ProcessTextureDirectly(eyeTexture);
                EndTiming("ProcessTextureDirectly");
                Destroy(eyeTexture);
                
                if (directBytes != null && directBytes.Length > 0)
                {
                    onComplete?.Invoke(Convert.ToBase64String(directBytes));
                }
                else
                {
                    onComplete?.Invoke("");
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error processing texture directly: {e.Message}");
                Destroy(eyeTexture);
                onComplete?.Invoke("");
            }
            EndTiming("CaptureAndEncodeImageAsync_Internal");
        }

        /// <summary>
        /// Collects the texture from the specified eye using OpenVR
        /// </summary>
        private Texture2D CollectTextureFromEye(EVREye eye)
        {
            StartTiming("CollectTextureFromEye_Internal");
            var tex = new Texture2D(2, 2);
            var nativeTex = System.IntPtr.Zero;
            
            StartTiming("GetMirrorTextureD3D11");
            var status = OpenVR.Compositor.GetMirrorTextureD3D11(eye, tex.GetNativeTexturePtr(), ref nativeTex);
            EndTiming("GetMirrorTextureD3D11");
            
            if (status == EVRCompositorError.None)
            {
                try
                {
                    uint width = 0, height = 0;
                    OpenVR.System.GetRecommendedRenderTargetSize(ref width, ref height);
                    
                    StartTiming("CreateExternalTexture");
                    tex = Texture2D.CreateExternalTexture((int)width, (int)height, TextureFormat.RGBA32, true, true, nativeTex);
                    EndTiming("CreateExternalTexture");
                }
                catch (System.Exception e)
                {
                    UnityEngine.Debug.LogError($"[InferencePoseClient] Error creating external texture: {e.Message}");
                    return null;
                }
            }
            else
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error collecting eye texture: {status}");
                return null;
            }
            
            EndTiming("CollectTextureFromEye_Internal");
            return tex;
        }
        
        /// <summary>
        /// Gets or creates a RenderTexture from the pool with the specified dimensions and format
        /// </summary>
        private RenderTexture GetRenderTextureFromPool(int width, int height, RenderTextureFormat format = RenderTextureFormat.ARGB32, bool enableRandomWrite = true)
        {
            StartTiming("GetRenderTextureFromPool");
            
            // Create a unique key for this texture configuration
            string key = $"{width}x{height}_{format}_{enableRandomWrite}";
            
            // Check if we have this texture in the pool
            if (renderTexturePool.TryGetValue(key, out RenderTexture rt) && rt != null)
            {
                // Found a pooled texture, return it
                EndTiming("GetRenderTextureFromPool");
                return rt;
            }
            
            // No suitable texture in pool, create a new one
            rt = new RenderTexture(width, height, 0, format, RenderTextureReadWrite.Linear);
            rt.enableRandomWrite = enableRandomWrite;
            rt.Create();
            
            // Add to pool
            renderTexturePool[key] = rt;
            
            EndTiming("GetRenderTextureFromPool");
            return rt;
        }

        /// <summary>
        /// Gets or creates a RenderTexture from the pool with the specified dimensions (legacy method for backward compatibility)
        /// </summary>
        private RenderTexture GetSourceRenderTexture(int width, int height)
        {
            lastTextureWidth = width;
            lastTextureHeight = height;
            return GetRenderTextureFromPool(width, height);
        }

        /// <summary>
        /// Gets or creates an output RenderTexture from the pool with the specified dimensions
        /// </summary>
        private RenderTexture GetOutputRenderTexture(int width, int height)
        {
            lastOutputWidth = width;
            lastOutputHeight = height;
            return GetRenderTextureFromPool(width, height);
        }

        /// <summary>
        /// Gets or creates a Texture2D from the pool with the specified dimensions and format
        /// </summary>
        private Texture2D GetTexture2DFromPool(int width, int height, TextureFormat format = TextureFormat.RGB24, bool mipChain = false)
        {
            StartTiming("GetTexture2DFromPool");
            
            // Create a unique key for this texture configuration
            string key = $"{width}x{height}_{format}_{mipChain}";
            
            // Check if we have this texture in the pool
            if (texture2DPool.TryGetValue(key, out Texture2D tex) && tex != null)
            {
                // Found a pooled texture, return it
                EndTiming("GetTexture2DFromPool");
                return tex;
            }
            
            // No suitable texture in pool, create a new one
            tex = new Texture2D(width, height, format, mipChain);
            
            // Add to pool
            texture2DPool[key] = tex;
            
            EndTiming("GetTexture2DFromPool");
            return tex;
        }

        /// <summary>
        /// Gets or creates a result Texture2D from the pool with the specified dimensions (legacy method for backward compatibility)
        /// </summary>
        private Texture2D GetResultTexture(int width, int height)
        {
            return GetTexture2DFromPool(width, height, TextureFormat.RGB24, false);
        }

        /// <summary>
        /// Gets or creates a ComputeBuffer from the pool with the specified size
        /// </summary>
        private ComputeBuffer GetComputeBufferFromPool(int size, int stride)
        {
            StartTiming("GetComputeBufferFromPool");
            
            // Create a unique key for this buffer configuration
            string key = $"buffer_{size}_{stride}";
            
            // Check if we have this buffer in the pool
            if (computeBufferPool.TryGetValue(key, out ComputeBuffer buffer) && buffer != null)
            {
                // Found a pooled buffer, return it
                EndTiming("GetComputeBufferFromPool");
                return buffer;
            }
            
            // No suitable buffer in pool, create a new one
            buffer = new ComputeBuffer(size, stride);
            
            // Add to pool
            computeBufferPool[key] = buffer;
            
            EndTiming("GetComputeBufferFromPool");
            return buffer;
        }

        /// <summary>
        /// Process texture using one-pass GPU processing with compute shader
        /// This is a fully asynchronous implementation using coroutines with no blocking wait
        /// Optimized to minimize CPU-GPU transfers and avoid texture creation/destruction
        /// </summary>
        private IEnumerator ProcessTextureWithOnePassGPUAsync(Texture2D texIn, Action<byte[]> onComplete)
        {
            StartTiming("ProcessTextureWithOnePassGPUAsync");
            try
            {
                // Calculate output dimensions based on compression settings
                int finalWidth = texIn.width;
                int finalHeight = texIn.height;
                bool needsResize = enableTextureCompression && imageWidth > 0 && imageHeight > 0;
                
                if (needsResize)
                {
                    finalWidth = imageWidth;
                    finalHeight = imageHeight;
                }
                
                // Get or create a render texture for output
                RenderTexture outputRT = GetRenderTextureFromPool(finalWidth, finalHeight);
                
                // Create a compute buffer to receive the encoded data
                // Each pixel is packed into a single uint (RGB)
                ComputeBuffer encodedBuffer = GetComputeBufferFromPool(finalWidth * finalHeight, sizeof(uint));
                
                // Find the appropriate kernel
                int kernelIndex = flipTextureShader.FindKernel("FlipResizeAndConvert");
                
                // Set compute shader parameters
                flipTextureShader.SetTexture(kernelIndex, "InputTexture", texIn);
                flipTextureShader.SetTexture(kernelIndex, "OutputTexture", outputRT);
                flipTextureShader.SetBuffer(kernelIndex, "EncodedBuffer", encodedBuffer);
                flipTextureShader.SetInt("InputWidth", texIn.width);
                flipTextureShader.SetInt("InputHeight", texIn.height);
                flipTextureShader.SetInt("OutputWidth", finalWidth);
                flipTextureShader.SetInt("OutputHeight", finalHeight);
                flipTextureShader.SetFloat("CompressionQuality", jpgQuality / 100f); // Convert 0-100 to 0-1 range
                
                // Calculate thread groups (8x8 threads per group)
                int threadGroupsX = Mathf.CeilToInt(finalWidth / 8.0f);
                int threadGroupsY = Mathf.CeilToInt(finalHeight / 8.0f);
                
                // Dispatch the compute shader
                flipTextureShader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
                
                // Request async readback of the encoded buffer
                // This avoids the expensive texture readback and JPG encoding
                AsyncGPUReadback.Request(encodedBuffer, (AsyncGPUReadbackRequest request) => 
                {
                    try
                    {
                        if (request.hasError)
                        {
                            UnityEngine.Debug.LogError("[InferencePoseClient] GPU readback error");
                            onComplete?.Invoke(null);
                            return;
                        }
                        
                        // Get the packed RGB data
                        NativeArray<uint> packedData = request.GetData<uint>();
                        
                        try
                        {
                            // Create a texture
                            Texture2D resultTexture = new Texture2D(finalWidth, finalHeight, TextureFormat.RGB24, false);
                            
                            // Unpack the RGB data
                            Color32[] colors = new Color32[packedData.Length];
                            for (int i = 0; i < packedData.Length; i++)
                            {
                                uint packed = packedData[i];
                                byte r = (byte)((packed >> 16) & 0xFF);
                                byte g = (byte)((packed >> 8) & 0xFF);
                                byte b = (byte)(packed & 0xFF);
                                colors[i] = new Color32(r, g, b, 255);
                            }
                            
                            // Set the pixels
                            resultTexture.SetPixels32(colors);
                            resultTexture.Apply(false);
                            
                            // Encode to JPG
                            byte[] jpgData = resultTexture.EncodeToJPG(jpgQuality);
                            
                            // Clean up
                            UnityEngine.Object.Destroy(resultTexture);
                            
                            // Return the result
                            onComplete?.Invoke(jpgData);
                        }
                        catch (Exception e)
                        {
                            UnityEngine.Debug.LogError($"[InferencePoseClient] Error processing texture: {e.Message}");
                            onComplete?.Invoke(null);
                        }
                    }
                    catch (Exception e)
                    {
                        UnityEngine.Debug.LogError($"[InferencePoseClient] Error processing buffer: {e.Message}");
                        onComplete?.Invoke(null);
                    }
                });
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error in GPU processing: {e.Message}");
                onComplete?.Invoke(null);
            }
            
            // No need to wait for completion - the callback will handle everything
            EndTiming("ProcessTextureWithOnePassGPUAsync");
            yield break;
        }

        /// <summary>
        /// Process texture using AsyncGPUReadback for efficient texture processing
        /// This is a fully asynchronous implementation using coroutines with no blocking wait
        /// Optimized to reuse textures and minimize CPU-GPU transfers
        /// </summary>
        private IEnumerator ProcessTextureWithGPUReadbackAsync(Texture2D texIn, Action<byte[]> onComplete)
        {
            // Use the new one-pass GPU method if compute shader is available
            if (flipTextureShader != null)
            {
                yield return StartCoroutine(ProcessTextureWithOnePassGPUAsync(texIn, onComplete));
                yield break;
            }
            
            // Fall back to the original method if compute shader is not available
            StartTiming("ProcessTextureWithGPUReadbackAsync_Internal");
            try
            {
                // Get or create a render texture from the pool
                RenderTexture source = GetRenderTextureFromPool(texIn.width, texIn.height);
                
                // Blit the input texture to the source render texture
                Graphics.Blit(texIn, source);
                
                // Calculate output dimensions based on compression settings
                int finalWidth = texIn.width;
                int finalHeight = texIn.height;
                bool needsResize = enableTextureCompression && imageWidth > 0 && imageHeight > 0;
                
                if (needsResize)
                {
                    finalWidth = imageWidth;
                    finalHeight = imageHeight;
                }
                
                // Request the readback directly from the source texture
                // Pass the dimensions to the callback
                AsyncGPUReadback.Request(source, 0, (AsyncGPUReadbackRequest request) => 
                {
                    try
                    {
                        OnCompleteReadbackOptimized(request, texIn.width, texIn.height, finalWidth, finalHeight, onComplete);
                    }
                    catch (Exception e)
                    {
                        UnityEngine.Debug.LogError($"[InferencePoseClient] Error in GPU readback callback: {e.Message}");
                        onComplete?.Invoke(null);
                    }
                });
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error in GPU readback: {e.Message}");
                onComplete?.Invoke(null);
            }
            
            // No need to wait for completion - the callback will handle everything
            EndTiming("ProcessTextureWithGPUReadbackAsync_Internal");
            yield break;
        }

        /// <summary>
        /// Optimized callback for when the GPU readback is complete
        /// Uses texture pooling and minimizes CPU-GPU transfers
        /// </summary>
        private void OnCompleteReadbackOptimized(AsyncGPUReadbackRequest request, int originalWidth, int originalHeight, int finalWidth, int finalHeight, Action<byte[]> onComplete)
        {
            StartTiming("OnCompleteReadbackOptimized");
            if (request.hasError)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] GPU readback error");
                onComplete?.Invoke(null);
                EndTiming("OnCompleteReadbackOptimized");
                return;
            }
            
            try
            {
                // Use GPU-based processing if compute shader is available (preferred path)
                if (flipTextureShader != null)
                {
                    // Get a pooled output render texture
                    RenderTexture outputRT = GetOutputRenderTexture(finalWidth, finalHeight);
                    
                    // Create a temporary texture to hold the original data directly from the request
                    // This is more efficient than creating a new Texture2D and copying data
                    // Using RGBA32 format consistently throughout the pipeline
                    Texture2D tempTexture = new Texture2D(request.width, request.height, TextureFormat.RGBA32, false, true);
                    
                    // Copy the data from request directly
                    tempTexture.LoadRawTextureData(request.GetData<byte>());
                    tempTexture.Apply(false);
                    
                    // Choose kernel based on whether we need to resize
                    bool needsResize = originalWidth != finalWidth || originalHeight != finalHeight;
                    int kernelIndex = needsResize ? 
                        flipTextureShader.FindKernel("FlipAndResize") : 
                        flipTextureShader.FindKernel("FlipVertical");
                    
                    // Set compute shader parameters
                    flipTextureShader.SetTexture(kernelIndex, "InputTexture", tempTexture);
                    flipTextureShader.SetTexture(kernelIndex, "OutputTexture", outputRT);
                    flipTextureShader.SetInt("InputWidth", originalWidth);
                    flipTextureShader.SetInt("InputHeight", originalHeight);
                    flipTextureShader.SetInt("OutputWidth", finalWidth);
                    flipTextureShader.SetInt("OutputHeight", finalHeight);
                    
                    // Calculate thread groups (8x8 threads per group)
                    int threadGroupsX = Mathf.CeilToInt(finalWidth / 8.0f);
                    int threadGroupsY = Mathf.CeilToInt(finalHeight / 8.0f);
                    
                    // Dispatch the compute shader
                    flipTextureShader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
                    
                    // Get or create a pooled result texture
                    // This uses RGB24 format (3 channels) for server compatibility
                    Texture2D resultTexture = GetResultTexture(finalWidth, finalHeight);
                    
                    // Read pixels from the render texture
                    RenderTexture.active = outputRT;
                    resultTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                    resultTexture.Apply(false);
                    
                    // Encode to JPG
                    byte[] jpgData = resultTexture.EncodeToJPG(jpgQuality);
                    
                    // Clean up temporary resources
                    RenderTexture.active = null;
                    Destroy(tempTexture);
                    
                    // Pass the JPG data
                    onComplete?.Invoke(jpgData);
                }
                else
                {
                    // Fallback to CPU-based processing when compute shader is not available
                    // Create a temporary texture to hold the original data
                    // Using RGBA32 for the initial texture to match the GPU readback format
                    Texture2D tempTexture = new Texture2D(request.width, request.height, TextureFormat.RGBA32, false);
                    
                    // Copy the data from request
                    tempTexture.LoadRawTextureData(request.GetData<byte>());
                    tempTexture.Apply(false);
                    
                    // Check if we need to resize
                    bool needsResize = originalWidth != finalWidth || originalHeight != finalHeight;
                    
                    if (needsResize)
                    {
                        // Get a pooled render texture for resizing
                        RenderTexture rt = GetOutputRenderTexture(finalWidth, finalHeight);
                        
                        // Flip the texture vertically
                        FlipTextureVertically(tempTexture);
                        
                        // Blit to resize
                        Graphics.Blit(tempTexture, rt);
                        
                        // Get or create a pooled result texture
                        // This uses RGB24 format (3 channels) for server compatibility
                        Texture2D resultTexture = GetResultTexture(finalWidth, finalHeight);
                        
                        // Read pixels from the render texture
                        RenderTexture.active = rt;
                        resultTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                        resultTexture.Apply(false);
                        
                        // Encode to JPG
                        byte[] jpgData = resultTexture.EncodeToJPG(jpgQuality);
                        
                        // Clean up temporary resources
                        RenderTexture.active = null;
                        Destroy(tempTexture);
                        
                        // Pass the JPG data
                        onComplete?.Invoke(jpgData);
                    }
                    else
                    {
                        // No compression - just flip
                        FlipTextureVertically(tempTexture);
                        
                        // Encode to JPG
                        byte[] jpgData = tempTexture.EncodeToJPG(jpgQuality);
                        
                        // Clean up
                        Destroy(tempTexture);
                        
                        // Pass the JPG data
                        onComplete?.Invoke(jpgData);
                    }
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error processing texture: {e.Message}");
                onComplete?.Invoke(null);
            }
            EndTiming("OnCompleteReadbackOptimized");
        }

        /// <summary>
        /// Legacy synchronous method for backward compatibility
        /// Updated to use the optimized implementation
        /// </summary>
        private byte[] ProcessTextureWithGPUReadback(Texture2D texIn)
        {
            StartTiming("ProcessTextureWithGPUReadback");
            if (!SystemInfo.supportsAsyncGPUReadback)
            {
                UnityEngine.Debug.LogWarning("[InferencePoseClient] AsyncGPUReadback not supported on this platform. Using direct processing.");
                EndTiming("ProcessTextureWithGPUReadback");
                return ProcessTextureDirectly(texIn);
            }
            
            // Create a TaskCompletionSource to wait for the async operation
            var completionSource = new TaskCompletionSource<byte[]>();
            
            // Calculate output dimensions based on compression settings
            int finalWidth = texIn.width;
            int finalHeight = texIn.height;
            bool needsResize = enableTextureCompression && imageWidth > 0 && imageHeight > 0;
            
            if (needsResize)
            {
                finalWidth = imageWidth;
                finalHeight = imageHeight;
            }
            
            // Get or create a render texture from the pool
            RenderTexture source = GetSourceRenderTexture(texIn.width, texIn.height);
            
            // Blit the input texture to the source render texture
            Graphics.Blit(texIn, source);
            
            // Request the readback using the optimized implementation
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(source, 0, (AsyncGPUReadbackRequest req) => {
                OnCompleteReadbackOptimized(req, texIn.width, texIn.height, finalWidth, finalHeight, (bytes) => {
                    completionSource.SetResult(bytes);
                });
            });
            
            // Wait for the result with a timeout
            if (completionSource.Task.Wait(1000))
            {
                EndTiming("ProcessTextureWithGPUReadback");
                return completionSource.Task.Result;
            }
            else
            {
                UnityEngine.Debug.LogWarning("[InferencePoseClient] GPU readback timed out or had an error");
                EndTiming("ProcessTextureWithGPUReadback");
                return null;
            }
        }

        /// <summary>
        /// Original callback for when the GPU readback is complete (kept for backward compatibility)
        /// </summary>
        private void OnCompleteReadback(AsyncGPUReadbackRequest request, int originalWidth, int originalHeight, Action<byte[]> onComplete)
        {
            StartTiming("OnCompleteReadback");
            if (request.hasError)
            {
                UnityEngine.Debug.LogError("[InferencePoseClient] GPU readback error");
                onComplete?.Invoke(null);
                EndTiming("OnCompleteReadback");
                return;
            }
            
            try
            {
                // Apply compression if enabled
                int finalWidth = originalWidth;
                int finalHeight = originalHeight;
                bool needsResize = enableTextureCompression && imageWidth > 0 && imageHeight > 0;
                
                if (needsResize)
                {
                    finalWidth = imageWidth;
                    finalHeight = imageHeight;
                }
                
                // Create a temporary texture to hold the original data
                // Using RGBA32 for the initial texture to match the GPU readback format
                Texture2D originalTexture = new Texture2D(request.width, request.height, TextureFormat.RGBA32, false);
                
                // Copy the data from request
                for (var i = 0; i < request.layerCount; i++)
                    originalTexture.SetPixels32(request.GetData<Color32>(i).ToArray(), i);
                
                originalTexture.Apply();
                
                // Use GPU-based processing if compute shader is available
                if (flipTextureShader != null)
                {
                    // Create output render texture
                    RenderTexture outputRT = RenderTexture.GetTemporary(finalWidth, finalHeight, 0, RenderTextureFormat.ARGB32);
                    outputRT.enableRandomWrite = true;
                    outputRT.Create();
                    
                    // Choose kernel based on whether we need to resize
                    int kernelIndex = needsResize ? 
                        flipTextureShader.FindKernel("FlipAndResize") : 
                        flipTextureShader.FindKernel("FlipVertical");
                    
                    // Set compute shader parameters
                    flipTextureShader.SetTexture(kernelIndex, "InputTexture", originalTexture);
                    flipTextureShader.SetTexture(kernelIndex, "OutputTexture", outputRT);
                    flipTextureShader.SetInt("InputWidth", originalWidth);
                    flipTextureShader.SetInt("InputHeight", originalHeight);
                    flipTextureShader.SetInt("OutputWidth", finalWidth);
                    flipTextureShader.SetInt("OutputHeight", finalHeight);
                    
                    // Calculate thread groups (8x8 threads per group)
                    int threadGroupsX = Mathf.CeilToInt(finalWidth / 8.0f);
                    int threadGroupsY = Mathf.CeilToInt(finalHeight / 8.0f);
                    
                    // Dispatch the compute shader
                    flipTextureShader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
                    
                    // Create the final texture at the desired size
                    Texture2D resultTexture = new Texture2D(finalWidth, finalHeight, TextureFormat.RGB24, false);
                    RenderTexture.active = outputRT;
                    resultTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                    resultTexture.Apply(false);
                    
                    // Encode to JPG instead of raw data
                    byte[] jpgData = resultTexture.EncodeToJPG(jpgQuality);
                    
                    // Clean up
                    RenderTexture.active = null;
                    RenderTexture.ReleaseTemporary(outputRT);
                    Destroy(originalTexture);
                    Destroy(resultTexture);
                    
                    // Pass the JPG data
                    onComplete?.Invoke(jpgData);
                }
                else
                {
                    // Fallback to CPU-based processing
                    if (needsResize)
                    {
                        // Create a temporary RenderTexture for resizing
                        RenderTexture rt = RenderTexture.GetTemporary(finalWidth, finalHeight, 0, RenderTextureFormat.ARGB32);
                        
                        // Flip the texture vertically
                        FlipTextureVertically(originalTexture);
                        
                        // Blit to resize
                        Graphics.Blit(originalTexture, rt);
                        
                        // Create the final texture at the desired size
                        Texture2D resizedTexture = new Texture2D(finalWidth, finalHeight, TextureFormat.RGB24, false);
                        RenderTexture.active = rt;
                        resizedTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                        resizedTexture.Apply(false);
                        
                        // Encode to JPG instead of raw data
                        byte[] jpgData = resizedTexture.EncodeToJPG(jpgQuality);
                        
                        // Clean up
                        RenderTexture.active = null;
                        RenderTexture.ReleaseTemporary(rt);
                        Destroy(originalTexture);
                        Destroy(resizedTexture);
                        
                        // Pass the JPG data
                        onComplete?.Invoke(jpgData);
                    }
                    else
                    {
                        // No compression - just flip
                        FlipTextureVertically(originalTexture);
                        
                        // Encode to JPG instead of raw data
                        byte[] jpgData = originalTexture.EncodeToJPG(jpgQuality);
                        
                        // Clean up
                        Destroy(originalTexture);
                        
                        // Pass the JPG data
                        onComplete?.Invoke(jpgData);
                    }
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error processing texture: {e.Message}");
            }
            EndTiming("OnCompleteReadback");
        }
        
        /// <summary>
        /// Process texture directly without AsyncGPUReadback
        /// Optimized to reuse textures when possible
        /// </summary>
        private byte[] ProcessTextureDirectly(Texture2D texIn)
        {
            StartTiming("ProcessTextureDirectly");
            try
            {
                // Get or create a render texture from the pool
                RenderTexture source = GetSourceRenderTexture(texIn.width, texIn.height);
                
                // Blit the input texture to the source render texture
                Graphics.Blit(texIn, source);
                
                // Apply compression if enabled
                int finalWidth = texIn.width;
                int finalHeight = texIn.height;
                bool needsResize = enableTextureCompression && imageWidth > 0 && imageHeight > 0;
                
                if (needsResize)
                {
                    finalWidth = imageWidth;
                    finalHeight = imageHeight;
                }
                    
                // Get or create an output render texture from the pool
                RenderTexture outputRT = GetOutputRenderTexture(finalWidth, finalHeight);
                    
                // Use compute shader for flipping and resizing if available
                if (flipTextureShader != null)
                {
                    // Choose kernel based on whether we need to resize
                    int kernelIndex = needsResize ? 
                        flipTextureShader.FindKernel("FlipAndResize") : 
                        flipTextureShader.FindKernel("FlipVertical");
                        
                    // Set compute shader parameters
                    flipTextureShader.SetTexture(kernelIndex, "InputTexture", source);
                    flipTextureShader.SetTexture(kernelIndex, "OutputTexture", outputRT);
                    flipTextureShader.SetInt("InputWidth", texIn.width);
                    flipTextureShader.SetInt("InputHeight", texIn.height);
                    flipTextureShader.SetInt("OutputWidth", finalWidth);
                    flipTextureShader.SetInt("OutputHeight", finalHeight);
                        
                    // Calculate thread groups (8x8 threads per group)
                    int threadGroupsX = Mathf.CeilToInt(finalWidth / 8.0f);
                    int threadGroupsY = Mathf.CeilToInt(finalHeight / 8.0f);
                        
                    // Dispatch the compute shader
                    flipTextureShader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
                }
                else
                {
                    // CPU fallback path if compute shader is not available
                    // Blit to resize
                    Graphics.Blit(source, outputRT);
                        
                    // Get a pooled texture for CPU flipping
                    Texture2D tempTexture = GetResultTexture(finalWidth, finalHeight);
                    RenderTexture.active = outputRT;
                    tempTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                    tempTexture.Apply(false);
                        
                    // Flip the texture vertically using CPU method
                    FlipTextureVertically(tempTexture);
                        
                    // Copy back to output render texture
                    Graphics.Blit(tempTexture, outputRT);
                }
                
                
                // Get or create a result texture from the pool
                Texture2D resultTexture = GetResultTexture(finalWidth, finalHeight);
                
                // Read pixels from the output render texture
                RenderTexture.active = needsResize ? outputRT : source;
                resultTexture.ReadPixels(new Rect(0, 0, finalWidth, finalHeight), 0, 0);
                resultTexture.Apply(false);
                
                // Encode to JPG instead of raw data
                byte[] jpgData = resultTexture.EncodeToJPG(jpgQuality);
                
                // Reset render texture active
                RenderTexture.active = null;
                
                EndTiming("ProcessTextureDirectly");
                return jpgData;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[InferencePoseClient] Error in direct texture processing: {e.Message}");
                EndTiming("ProcessTextureDirectly");
                return null;
            }
        }

        /// <summary>
        /// Flips a texture vertically (top to bottom), equivalent to Python's Image.transpose(Image.FLIP_TOP_BOTTOM)
        /// </summary>
        private void FlipTextureVertically(Texture2D texture)
        {
            StartTiming("FlipTextureVertically");
            // Fallback to CPU-based flipping (optimized version)
            Color32[] pixels = texture.GetPixels32();
            int width = texture.width;
            int height = texture.height;
            int halfHeight = height / 2;
            
            // Only iterate through half the height, swapping top and bottom pixels in-place
            for (int y = 0; y < halfHeight; y++)
            {
                int topRowStart = y * width;
                int bottomRowStart = (height - y - 1) * width;
                
                for (int x = 0; x < width; x++)
                {
                    Color32 temp = pixels[topRowStart + x];
                    pixels[topRowStart + x] = pixels[bottomRowStart + x];
                    pixels[bottomRowStart + x] = temp;
                }
            }
            
            texture.SetPixels32(pixels);
            texture.Apply(false); // false = don't make texture readable again (optimization)
            EndTiming("FlipTextureVertically");
        }

        private void StoreControllerInputValues(float leftIndex, float rightIndex, float leftHand, float rightHand, 
                                              float leftThumbX, float leftThumbY, float rightThumbX, float rightThumbY)
        {
            leftIndexTrigger = leftIndex;
            rightIndexTrigger = rightIndex;
            leftHandTrigger = leftHand;
            rightHandTrigger = rightHand;
            leftThumbstickX = leftThumbX;
            leftThumbstickY = leftThumbY;
            rightThumbstickX = rightThumbX;
            rightThumbstickY = rightThumbY;
        }
        
        private void ApplyAction(float[] action)
        {
            StartTiming("ApplyAction");
            if (action == null || action.Length == 0 || deviceController == null || deviceController.IsConnectedToDriver() == false)
            {
                UnityEngine.Debug.LogWarning("Cannot apply action: Invalid action data or missing device controller or not connected to driver");
                EndTiming("ApplyAction");
                return;
            }

            try
            {
                // Create a PoseReplayFrame from the action data
                var frame = new XRT_OVR_Grabber.PoseReplayFrame();
                
                int index = 0;
                
                // Action format from Python:
                // [0-2]: Head position delta, [3-6]: Head rotation delta (quaternion),
                // [7-9]: Left controller position delta, [10-13]: Left controller rotation delta,
                // [14-16]: Right controller position delta, [17-20]: Right controller rotation delta,
                // [21-23]: XR Origin position delta,
                // [24]: Left index trigger, [25]: Right index trigger, [26]: Left hand trigger, [27]: Right hand trigger,
                // [28-29]: Left thumbstick XY, [30-31]: Right thumbstick XY
                
                // Parse HMD position and rotation
                frame.hmdPosition = new Vector3(action[index++], action[index++], action[index++]);
                frame.hmdRotation = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
                
                // Parse left controller position and rotation
                frame.leftControllerPosition = new Vector3(action[index++], action[index++], action[index++]);
                frame.leftControllerRotation = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
                
                // Parse right controller position and rotation
                frame.rightControllerPosition = new Vector3(action[index++], action[index++], action[index++]);
                frame.rightControllerRotation = new Quaternion(action[index++], action[index++], action[index++], action[index++]);
                
                // Store the latest pose values for use in GetCurrentQPos
                latestHmdPosition = frame.hmdPosition;
                latestHmdRotation = frame.hmdRotation;
                latestLeftControllerPosition = frame.leftControllerPosition;
                latestLeftControllerRotation = frame.leftControllerRotation;
                latestRightControllerPosition = frame.rightControllerPosition;
                latestRightControllerRotation = frame.rightControllerRotation;
                useLatestPoseValues = true;
                
                // Skip XR Origin position delta (3 values)
                index += 3;
                
                // Store controller input values for use in GetCurrentQPos
                if (index + 7 < action.Length)
                {
                    // Left and right trigger values
                    float leftIndexTrigger = action[index++];
                    float rightIndexTrigger = action[index++];
                    float leftHandTrigger = action[index++];
                    float rightHandTrigger = action[index++];
                    
                    // Thumbstick values
                    float leftThumbstickX = action[index++];
                    float leftThumbstickY = action[index++];
                    float rightThumbstickX = action[index++];
                    float rightThumbstickY = action[index++];
                    
                    // Set button values
                    // frame.leftTriggerValue = leftIndexTrigger;
                    // frame.leftGripValue = leftHandTrigger;
                    // frame.rightTriggerValue = rightIndexTrigger;
                    // frame.rightGripValue = rightHandTrigger;
                    
                    // Set button states using edge detection
                    if (!prevLeftTriggerState && leftIndexTrigger > pressThreshold)
                    {
                        frame.leftTriggerState = true;
                    }
                    else if (prevLeftTriggerState && leftIndexTrigger < releaseThreshold)
                    {
                        frame.leftTriggerState = false;
                    }
                    else
                    {
                        frame.leftTriggerState = prevLeftTriggerState;
                    }
                    
                    if (!prevLeftGripState && leftHandTrigger > pressThreshold)
                    {
                        frame.leftGripState = true;
                    }
                    else if (prevLeftGripState && leftHandTrigger < releaseThreshold)
                    {
                        frame.leftGripState = false;
                    }
                    else
                    {
                        frame.leftGripState = prevLeftGripState;
                    }
                    
                    if (!prevRightTriggerState && rightIndexTrigger > pressThreshold)
                    {
                        frame.rightTriggerState = true;
                    }
                    else if (prevRightTriggerState && rightIndexTrigger < releaseThreshold)
                    {
                        frame.rightTriggerState = false;
                    }
                    else
                    {
                        frame.rightTriggerState = prevRightTriggerState;
                    }
                    
                    if (!prevRightGripState && rightHandTrigger > pressThreshold)
                    {
                        frame.rightGripState = true;
                    }
                    else if (prevRightGripState && rightHandTrigger < releaseThreshold)
                    {
                        frame.rightGripState = false;
                    }
                    else
                    {
                        frame.rightGripState = prevRightGripState;
                    }
                    
                    // Log button values
                    UnityEngine.Debug.Log($"Left Index Trigger: {leftIndexTrigger}, Left Hand Trigger: {leftHandTrigger}, Right Index Trigger: {rightIndexTrigger}, Right Hand Trigger: {rightHandTrigger}");

                    // Store current button states as previous for next comparison
                    prevLeftTriggerState = frame.leftTriggerState;
                    prevRightTriggerState = frame.rightTriggerState;
                    prevLeftGripState = frame.leftGripState;
                    prevRightGripState = frame.rightGripState;
                    
                    // Update button states
                    leftIndexTrigger = frame.leftTriggerState ? 1.0f : 0.0f;
                    leftHandTrigger = frame.leftGripState ? 1.0f : 0.0f;
                    rightIndexTrigger = frame.rightTriggerState ? 1.0f : 0.0f;
                    rightHandTrigger = frame.rightGripState ? 1.0f : 0.0f;

                    // log button states
                    // UnityEngine.Debug.Log($"Left Index Trigger: {leftIndexTrigger}, Left Hand Trigger: {leftHandTrigger}, Right Index Trigger: {rightIndexTrigger}, Right Hand Trigger: {rightHandTrigger}");
                    // Store these values for use in GetCurrentQPos
                    StoreControllerInputValues(leftIndexTrigger, rightIndexTrigger, leftHandTrigger, rightHandTrigger,
                                              leftThumbstickX, leftThumbstickY, rightThumbstickX, rightThumbstickY);
                }
                
                // Apply the pose to OpenVR devices
                deviceController.ApplyPoseToOpenVR(frame);
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"Error applying action: {e.Message}");
            }
            
            EndTiming("ApplyAction");
        }

        #region Public Control Methods

        /// <summary>
        /// Start the connection to the inference server
        /// </summary>
        public void StartConnection()
        {
            shouldReconnect = true;
            if (!isConnected)
            {
                Task.Run(() => ConnectToServer());
            }
        }

        /// <summary>
        /// Stop the connection to the inference server
        /// </summary>
        public void StopConnection()
        {
            shouldReconnect = false;
            DisconnectFromServer();
            
            // Clear stored controller input values
            leftIndexTrigger = 0f;
            rightIndexTrigger = 0f;
            leftHandTrigger = 0f;
            rightHandTrigger = 0f;
            leftThumbstickX = 0f;
            leftThumbstickY = 0f;
            rightThumbstickX = 0f;
            rightThumbstickY = 0f;
            
            // Clear stored pose values
            latestHmdPosition = Vector3.zero;
            latestHmdRotation = Quaternion.identity;
            latestLeftControllerPosition = Vector3.zero;
            latestLeftControllerRotation = Quaternion.identity;
            latestRightControllerPosition = Vector3.zero;
            latestRightControllerRotation = Quaternion.identity;
            useLatestPoseValues = false;
            
            // Clear action queue
            lock (lockObject)
            {
                actionQueue.Clear();
                latestTimestepInQueue = int.MinValue;
            }
        }

        /// <summary>
        /// Reset the client timestep counter
        /// </summary>
        public void ResetTimestep()
        {
            lock (lockObject)
            {
                clientTimestep = 0;
                actionQueue.Clear();
                latestTimestepInQueue = int.MinValue;
            }
        }

        /// <summary>
        /// Set the update interval (frequency)
        /// </summary>
        /// <param name="interval">Interval in seconds</param>
        public void SetUpdateInterval(float interval)
        {
            if (interval > 0)
            {
                updateInterval = interval;
            }
        }

        #endregion

        // Helper classes for JSON serialization
        [Serializable]
        private class StateData
        {
            public float[] qpos;
            public string image;
            public int timestep;
            public int imageWidth;
            public int imageHeight;
            public string imageFormat = "jpg"; // Now using JPG encoding instead of raw data
            
            // Toggle commands to send to Python server (using int for JsonUtility compatibility)
            public int toggle_window_size = -1;  // -1 = not set, 0 = false, 1 = true
            public int toggle_decay_factor = -1;
            public int toggle_update_rate = -1;
        }

        [Serializable]
        private class ActionData
        {
            public float[] action;
            public int timestep;
            public int window_size;
            public string decay_factor;
            public string stepped_inference_update_rate;
            public bool enable_dynamic_window_size;
            public bool enable_dynamic_decay_factor;
            public bool enable_dynamic_update_rate;
        }
        
        /// <summary>
        /// Display dynamic window information and toggle controls on screen
        /// </summary>
        private void OnGUI()
        {
            // Create a GUI style for the text
            GUIStyle style = new GUIStyle();
            style.fontSize = 16;
            style.normal.textColor = Color.white;
            style.fontStyle = FontStyle.Bold;
            
            // Create a GUI style for buttons
            GUIStyle buttonStyle = new GUIStyle(GUI.skin.button);
            buttonStyle.fontSize = 14;
            buttonStyle.fontStyle = FontStyle.Bold;
            
            // Display current values in top-left corner
            GUI.Label(new Rect(10, 10, 300, 25), $"Window Size: {currentWindowSize}", style);
            GUI.Label(new Rect(10, 35, 300, 25), $"Decay Factor: {currentDecayFactor}", style);
            GUI.Label(new Rect(10, 60, 300, 25), $"Update Rate: {currentInferenceUpdateRate:F4}s", style);
            GUI.Label(new Rect(10, 85, 300, 25), $"Auto Rate: {(useAutoUpdateRate ? "ON" : "OFF")}", style);
            
            // Dynamic Window Component Toggles
            GUI.Label(new Rect(10, 120, 300, 25), "Dynamic Window Controls:", style);
            
            // Window Size Toggle
            bool newWindowSizeToggle = GUI.Toggle(new Rect(10, 150, 200, 25), guiWindowSizeToggle, "Dynamic Window Size");
            if (newWindowSizeToggle != guiWindowSizeToggle)
            {
                guiWindowSizeToggle = newWindowSizeToggle;
                SendToggleCommand("window_size", guiWindowSizeToggle);
            }
            
            // Decay Factor Toggle
            bool newDecayFactorToggle = GUI.Toggle(new Rect(10, 175, 200, 25), guiDecayFactorToggle, "Dynamic Decay Factor");
            if (newDecayFactorToggle != guiDecayFactorToggle)
            {
                guiDecayFactorToggle = newDecayFactorToggle;
                SendToggleCommand("decay_factor", guiDecayFactorToggle);
            }
            
            // Update Rate Toggle
            bool newUpdateRateToggle = GUI.Toggle(new Rect(10, 200, 200, 25), guiUpdateRateToggle, "Dynamic Update Rate");
            if (newUpdateRateToggle != guiUpdateRateToggle)
            {
                guiUpdateRateToggle = newUpdateRateToggle;
                SendToggleCommand("update_rate", guiUpdateRateToggle);
            }
            
            // Status indicators
            GUI.Label(new Rect(220, 150, 100, 25), enableDynamicWindowSize ? "[ON]" : "[OFF]", 
                     enableDynamicWindowSize ? style : new GUIStyle(style) { normal = { textColor = Color.red } });
            GUI.Label(new Rect(220, 175, 100, 25), enableDynamicDecayFactor ? "[ON]" : "[OFF]", 
                     enableDynamicDecayFactor ? style : new GUIStyle(style) { normal = { textColor = Color.red } });
            GUI.Label(new Rect(220, 200, 100, 25), enableDynamicUpdateRate ? "[ON]" : "[OFF]", 
                     enableDynamicUpdateRate ? style : new GUIStyle(style) { normal = { textColor = Color.red } });
        }
        
        /// <summary>
        /// Send toggle command to Python server
        /// </summary>
        /// <param name="componentType">Type of component to toggle (window_size, decay_factor, update_rate)</param>
        /// <param name="enabled">Whether to enable or disable the component</param>
        private void SendToggleCommand(string componentType, bool enabled)
        {
            if (!isConnected)
            {
                UnityEngine.Debug.LogWarning($"Cannot send toggle command - not connected to server");
                return;
            }
            
            try
            {
                // Create state data with toggle command
                var stateData = new StateData
                {
                    qpos = GetCurrentQPos(),
                    image = "", // Empty image for toggle commands to reduce bandwidth
                    timestep = clientTimestep
                };
                
                // Set the appropriate toggle field (using int values for JsonUtility compatibility)
                switch (componentType)
                {
                    case "window_size":
                        stateData.toggle_window_size = enabled ? 1 : 0;
                        break;
                    case "decay_factor":
                        stateData.toggle_decay_factor = enabled ? 1 : 0;
                        break;
                    case "update_rate":
                        stateData.toggle_update_rate = enabled ? 1 : 0;
                        break;
                }
                
                // Convert to JSON and send
                string jsonData = JsonUtility.ToJson(stateData);
                byte[] data = Encoding.UTF8.GetBytes(jsonData + "END_OF_MESSAGE");
                
                lock (lockObject)
                {
                    if (isConnected && stream != null)
                    {
                        stream.Write(data, 0, data.Length);
                        UnityEngine.Debug.Log($"Sent toggle command: {componentType} = {enabled}");
                    }
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"Error sending toggle command: {e.Message}");
                HandleConnectionError();
            }
        }
    }
}
