using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Linq;

namespace CLOVR.IPCClient
{
    /// <summary>
    /// OCR-based reward tracking system for RL training.
    /// Receives score updates from OCR system via UDP and tracks rewards per timestep.
    /// </summary>
    public class OCRRewardTracker : MonoBehaviour
    {
        [Header("OCR Reward System Settings")]
        [SerializeField] private bool enableRewardTracking = true;
        [SerializeField] private string ocrServerIP = "127.0.0.1";
        [SerializeField] private int ocrUDPPort = 12345;
        [SerializeField] private string rlServerIP = "127.0.0.1";
        [SerializeField] private int rlServerPort = 5557;
        
        [Header("Reward Configuration")]
        [SerializeField] private float positiveStepReward = 1.0f;
        [SerializeField] private float negativeStepReward = -0.1f;
        [SerializeField] private float neutralStepReward = 0.0f;
        [SerializeField] private float episodeEndTimeout = 2.0f; // Seconds to wait before considering episode ended
        
        [Header("Debug Settings")]
        [SerializeField] private bool showDebugLogs = true;
        [SerializeField] private bool showRewardGUI = true;
        
        // Reference to the main inference client
        private InferencePoseClient inferenceClient;
        
        // UDP socket for receiving OCR scores
        private UdpClient udpClient;
        private Thread udpReceiveThread;
        private bool isReceivingUDP = false;
        
        // TCP client for sending rewards to RL server
        private TcpClient rlTcpClient;
        private NetworkStream rlStream;
        private bool isConnectedToRL = false;
        
        // Score and reward tracking
        private int currentScore = 0;
        private int lastValidScore = 0;
        private float lastScoreTime = 0f;
        private bool episodeActive = false;
        private int episodeStartTimestep = 0;
        
        // Reward tracking per timestep
        private Dictionary<int, float> timestepRewards = new Dictionary<int, float>();
        private List<int> processedTimesteps = new List<int>();
        private int lastProcessedTimestep = -1;
        
        // Episode management
        private bool waitingForEpisodeEnd = false;
        private Coroutine episodeEndCoroutine;
        private Coroutine rewardUpdateCoroutine;
        
        // Thread safety
        private readonly object scoreLock = new object();
        private readonly object rewardLock = new object();
        
        // GUI
        private Rect guiRect = new Rect(10, 10, 300, 200);
        
        void Start()
        {
            // Get reference to inference client
            inferenceClient = GetComponent<InferencePoseClient>();
            if (inferenceClient == null)
            {
                Debug.LogError("[OCRRewardTracker] InferencePoseClient component not found!");
                enabled = false;
                return;
            }
            
            if (enableRewardTracking)
            {
                StartRewardTracking();
            }
        }
        
        void OnDestroy()
        {
            StopRewardTracking();
        }
        
        public void StartRewardTracking()
        {
            if (!enableRewardTracking) return;
            
            Debug.Log("[OCRRewardTracker] Starting reward tracking system...");
            
            // Start UDP receiver for OCR scores
            StartUDPReceiver();
            
            // Start reward update coroutine
            if (rewardUpdateCoroutine == null)
            {
                rewardUpdateCoroutine = StartCoroutine(RewardUpdateCoroutine());
            }
        }
        
        public void StopRewardTracking()
        {
            Debug.Log("[OCRRewardTracker] Stopping reward tracking system...");
            
            // Stop UDP receiver
            StopUDPReceiver();
            
            // Stop coroutines
            if (episodeEndCoroutine != null)
            {
                StopCoroutine(episodeEndCoroutine);
                episodeEndCoroutine = null;
            }
            
            if (rewardUpdateCoroutine != null)
            {
                StopCoroutine(rewardUpdateCoroutine);
                rewardUpdateCoroutine = null;
            }
            
            // Disconnect from RL server
            DisconnectFromRLServer();
        }
        
        private void StartUDPReceiver()
        {
            try
            {
                udpClient = new UdpClient(ocrUDPPort);
                isReceivingUDP = true;
                
                udpReceiveThread = new Thread(UDPReceiveLoop)
                {
                    IsBackground = true
                };
                udpReceiveThread.Start();
                
                Debug.Log($"[OCRRewardTracker] UDP receiver started on port {ocrUDPPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[OCRRewardTracker] Failed to start UDP receiver: {e.Message}");
            }
        }
        
        private void StopUDPReceiver()
        {
            isReceivingUDP = false;
            
            if (udpClient != null)
            {
                udpClient.Close();
                udpClient = null;
            }
            
            if (udpReceiveThread != null && udpReceiveThread.IsAlive)
            {
                udpReceiveThread.Join(1000); // Wait up to 1 second
            }
        }
        
        private void UDPReceiveLoop()
        {
            IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
            
            while (isReceivingUDP)
            {
                try
                {
                    byte[] data = udpClient.Receive(ref remoteEndPoint);
                    string jsonData = Encoding.UTF8.GetString(data);
                    
                    // Parse score update
                    var scoreUpdate = JsonUtility.FromJson<ScoreUpdateMessage>(jsonData);
                    if (scoreUpdate != null && scoreUpdate.type == "score_update")
                    {
                        ProcessScoreUpdate(scoreUpdate.score);
                    }
                }
                catch (Exception e)
                {
                    if (isReceivingUDP) // Only log if we're still supposed to be receiving
                    {
                        Debug.LogError($"[OCRRewardTracker] UDP receive error: {e.Message}");
                    }
                }
            }
        }
        
        private void ProcessScoreUpdate(int newScore)
        {
            lock (scoreLock)
            {
                currentScore = newScore;
                
                if (newScore > 0)
                {
                    lastValidScore = newScore;
                    lastScoreTime = Time.time;
                    
                    // Start episode if not already active and inference client is connected
                    if (!episodeActive && inferenceClient != null && inferenceClient.IsConnected)
                    {
                        StartEpisode();
                    }
                    
                    // Cancel episode end waiting if score is received
                    if (waitingForEpisodeEnd)
                    {
                        waitingForEpisodeEnd = false;
                        if (episodeEndCoroutine != null)
                        {
                            StopCoroutine(episodeEndCoroutine);
                            episodeEndCoroutine = null;
                        }
                    }
                }
                else if (episodeActive)
                {
                    // Score is 0 or invalid, start episode end timer if not already waiting
                    if (!waitingForEpisodeEnd)
                    {
                        waitingForEpisodeEnd = true;
                        episodeEndCoroutine = StartCoroutine(EpisodeEndTimer());
                    }
                }
            }
            
            if (showDebugLogs)
            {
                Debug.Log($"[OCRRewardTracker] Score update: {newScore}");
            }
        }
        
        private void StartEpisode()
        {
            episodeActive = true;
            episodeStartTimestep = inferenceClient != null ? inferenceClient.CurrentTimestep : 0;
            lastProcessedTimestep = episodeStartTimestep - 1;
            
            // Clear previous rewards
            lock (rewardLock)
            {
                timestepRewards.Clear();
                processedTimesteps.Clear();
            }
            
            // Connect to RL server
            ConnectToRLServer();
            
            Debug.Log($"[OCRRewardTracker] Episode started at timestep {episodeStartTimestep}");
        }
        
        private IEnumerator EpisodeEndTimer()
        {
            yield return new WaitForSeconds(episodeEndTimeout);
            
            if (waitingForEpisodeEnd && episodeActive)
            {
                // Show confirmation dialog
                ShowEpisodeEndConfirmation();
            }
            
            waitingForEpisodeEnd = false;
            episodeEndCoroutine = null;
        }
        
        private void ShowEpisodeEndConfirmation()
        {
            // For now, automatically end episode after timeout
            // In a full implementation, you might want to show a proper dialog
            Debug.Log("[OCRRewardTracker] Episode timeout reached. Ending episode...");
            EndEpisode();
        }
        
        private void EndEpisode()
        {
            if (!episodeActive) return;
            
            episodeActive = false;
            waitingForEpisodeEnd = false;
            
            // Send final rewards to RL server
            SendRewardsToRLServer();
            
            Debug.Log($"[OCRRewardTracker] Episode ended. Total timesteps: {processedTimesteps.Count}");
        }
        
        private IEnumerator RewardUpdateCoroutine()
        {
            while (enabled)
            {
                if (episodeActive && inferenceClient != null)
                {
                    UpdateRewards();
                }
                
                yield return new WaitForSeconds(0.1f); // Update rewards at 10Hz
            }
        }
        
        private void UpdateRewards()
        {
            if (!episodeActive || inferenceClient == null) return;
            
            int currentTimestep = inferenceClient.CurrentTimestep;
            
            // Process new timesteps since last update
            for (int timestep = lastProcessedTimestep + 1; timestep <= currentTimestep; timestep++)
            {
                float reward = CalculateRewardForTimestep(timestep);
                
                lock (rewardLock)
                {
                    timestepRewards[timestep] = reward;
                    processedTimesteps.Add(timestep);
                }
                
                if (showDebugLogs && reward != 0)
                {
                    Debug.Log($"[OCRRewardTracker] Timestep {timestep}: Reward = {reward}");
                }
            }
            
            lastProcessedTimestep = currentTimestep;
        }
        
        private float CalculateRewardForTimestep(int timestep)
        {
            lock (scoreLock)
            {
                // If no valid score received yet, return neutral reward
                if (lastValidScore == 0)
                {
                    return neutralStepReward;
                }
                
                // Check if score increased compared to previous valid score
                // For simplicity, we'll use a basic approach here
                // In a more sophisticated implementation, you might track score changes more precisely
                
                if (currentScore > 0)
                {
                    // Score is positive, give positive reward
                    return positiveStepReward;
                }
                else
                {
                    // Score is 0 or invalid, give negative reward
                    return negativeStepReward;
                }
            }
        }
        
        private void ConnectToRLServer()
        {
            try
            {
                rlTcpClient = new TcpClient();
                rlTcpClient.Connect(rlServerIP, rlServerPort);
                rlStream = rlTcpClient.GetStream();
                isConnectedToRL = true;
                
                Debug.Log($"[OCRRewardTracker] Connected to RL server at {rlServerIP}:{rlServerPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[OCRRewardTracker] Failed to connect to RL server: {e.Message}");
                isConnectedToRL = false;
            }
        }
        
        private void DisconnectFromRLServer()
        {
            isConnectedToRL = false;
            
            if (rlStream != null)
            {
                rlStream.Close();
                rlStream = null;
            }
            
            if (rlTcpClient != null)
            {
                rlTcpClient.Close();
                rlTcpClient = null;
            }
        }
        
        private void SendRewardsToRLServer()
        {
            if (!isConnectedToRL || timestepRewards.Count == 0) return;
            
            try
            {
                // Prepare episode data
                var episodeData = new EpisodeDataMessage
                {
                    type = "episode_data",
                    data = new EpisodeData
                    {
                        timesteps = processedTimesteps.ToArray(),
                        rewards = processedTimesteps.Select(t => timestepRewards.ContainsKey(t) ? timestepRewards[t] : 0f).ToArray(),
                        done = true
                    }
                };
                
                string jsonData = JsonUtility.ToJson(episodeData);
                byte[] data = Encoding.UTF8.GetBytes(jsonData);
                
                rlStream.Write(data, 0, data.Length);
                rlStream.Flush();
                
                Debug.Log($"[OCRRewardTracker] Sent {processedTimesteps.Count} rewards to RL server");
            }
            catch (Exception e)
            {
                Debug.LogError($"[OCRRewardTracker] Failed to send rewards to RL server: {e.Message}");
            }
        }
        
        void OnGUI()
        {
            if (!showRewardGUI) return;
            
            GUI.Box(guiRect, "OCR Reward Tracker");
            
            GUILayout.BeginArea(new Rect(guiRect.x + 10, guiRect.y + 25, guiRect.width - 20, guiRect.height - 35));
            
            GUILayout.Label($"Episode Active: {episodeActive}");
            GUILayout.Label($"Current Score: {currentScore}");
            GUILayout.Label($"Last Valid Score: {lastValidScore}");
            GUILayout.Label($"Processed Timesteps: {processedTimesteps.Count}");
            GUILayout.Label($"RL Server Connected: {isConnectedToRL}");
            
            if (GUILayout.Button("Start Episode"))
            {
                if (!episodeActive && inferenceClient != null && inferenceClient.IsConnected)
                {
                    StartEpisode();
                }
            }
            
            if (GUILayout.Button("End Episode"))
            {
                if (episodeActive)
                {
                    EndEpisode();
                }
            }
            
            GUILayout.EndArea();
        }
        
        // Public methods for external control
        public void ForceStartEpisode()
        {
            if (!episodeActive && inferenceClient != null && inferenceClient.IsConnected)
            {
                StartEpisode();
            }
        }
        
        public void ForceEndEpisode()
        {
            if (episodeActive)
            {
                EndEpisode();
            }
        }
        
        public bool IsEpisodeActive => episodeActive;
        public int CurrentScore => currentScore;
        public int ProcessedTimestepsCount => processedTimesteps.Count;
        
        // Helper classes for JSON serialization
        [Serializable]
        private class ScoreUpdateMessage
        {
            public string type;
            public int score;
            public float timestamp;
        }
        
        [Serializable]
        private class EpisodeDataMessage
        {
            public string type;
            public EpisodeData data;
        }
        
        [Serializable]
        private class EpisodeData
        {
            public int[] timesteps;
            public float[] rewards;
            public bool done;
        }
    }
}
