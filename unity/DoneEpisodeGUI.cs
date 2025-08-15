using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

// Simple OnGUI tool to manually signal episode completion to the RL server.
// Sends a small JSON message {"type":"episode_control","done":true} over a TCP stream.
// You can either attach this to the same GameObject that manages the RL connection or
// set the host/port manually below to establish a lightweight connection.
//
// If you already manage a shared TcpClient/NetworkStream elsewhere (e.g., OCRRewardTracker),
// you can drag that component in the inspector and enable "Use Shared Stream" to reuse it.

public class DoneEpisodeGUI : MonoBehaviour
{
    [Header("Networking")]
    public string host = "127.0.0.1";
    public int port = 5557; // RL port (keep different from inference 5556)

    [Tooltip("Use an existing NetworkStream from another component (e.g., OCRRewardTracker)")]
    public bool useSharedStream = true;

    [Tooltip("Optional component holding a NetworkStream named 'rlStream'")]
    public MonoBehaviour sharedStreamProvider; // e.g., reference to OCRRewardTracker

    [Header("GUI")]
    public bool showGUI = true;
    public Rect guiRect = new Rect(20, 20, 260, 120);

    private TcpClient _client;
    private NetworkStream _stream;
    private bool _connected;

    void Start()
    {
        TryBindSharedStream();
        if (!_connected && !useSharedStream)
        {
            TryConnect();
        }
    }

    void OnDestroy()
    {
        if (!useSharedStream)
        {
            SafeClose();
        }
    }

    void TryBindSharedStream()
    {
        if (!useSharedStream || sharedStreamProvider == null) return;
        try
        {
            var providerType = sharedStreamProvider.GetType();
            var field = providerType.GetField("rlStream");
            if (field != null)
            {
                var val = field.GetValue(sharedStreamProvider) as NetworkStream;
                if (val != null)
                {
                    _stream = val;
                    _connected = _stream != null;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[DoneEpisodeGUI] Could not bind shared stream: {e.Message}");
        }
    }

    void TryConnect()
    {
        try
        {
            _client = new TcpClient();
            _client.Connect(host, port);
            _stream = _client.GetStream();
            _connected = true;
            Debug.Log($"[DoneEpisodeGUI] Connected to RL server {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[DoneEpisodeGUI] Failed to connect to RL server: {e.Message}");
            _connected = false;
        }
    }

    void SafeClose()
    {
        try { _stream?.Close(); } catch {}
        try { _client?.Close(); } catch {}
        _stream = null;
        _client = null;
        _connected = false;
    }

    void SendDoneMessage()
    {
        if (!_connected || _stream == null)
        {
            if (!useSharedStream)
            {
                TryConnect();
            }
            if (!_connected || _stream == null)
            {
                Debug.LogWarning("[DoneEpisodeGUI] Not connected to RL server.");
                return;
            }
        }

        try
        {
            var payload = new EpisodeControl { type = "episode_control", done = true };
            string json = JsonUtility.ToJson(payload) + "END_OF_MESSAGE";
            byte[] data = Encoding.UTF8.GetBytes(json);
            _stream.Write(data, 0, data.Length);
            _stream.Flush();
            Debug.Log("[DoneEpisodeGUI] Sent done signal to RL server.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DoneEpisodeGUI] Failed to send done signal: {e.Message}");
            _connected = false;
        }
    }

    void OnGUI()
    {
        if (!showGUI) return;
        GUILayout.BeginArea(guiRect, GUI.skin.box);
        GUILayout.Label("RL Episode Control");

        GUILayout.BeginHorizontal();
        GUILayout.Label("Connection:", GUILayout.Width(80));
        GUILayout.Label(_connected ? "Connected" : "Disconnected");
        GUILayout.EndHorizontal();

        if (GUILayout.Button("Mark Episode Done"))
        {
            SendDoneMessage();
        }

        GUILayout.Space(6);
        useSharedStream = GUILayout.Toggle(useSharedStream, "Use Shared Stream");
        if (!useSharedStream)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label("Host", GUILayout.Width(40));
            host = GUILayout.TextField(host, GUILayout.Width(120));
            GUILayout.Label(":", GUILayout.Width(10));
            string portStr = GUILayout.TextField(port.ToString(), GUILayout.Width(60));
            if (int.TryParse(portStr, out int p)) port = p;
            GUILayout.EndHorizontal();
        }

        GUILayout.EndArea();
    }

    [Serializable]
    private struct EpisodeControl
    {
        public string type;
        public bool done;
    }
}
