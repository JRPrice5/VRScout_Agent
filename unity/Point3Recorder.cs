using System;
using System.IO;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.InputSystem;

public class Point3Recorder : MonoBehaviour
{
    [Header("Avatar Transforms")]
    public Transform mainCamera;
    public Transform leftController;
    public Transform rightController;
    public Transform xrOrigin;

    [Header("Input Actions")]
    public InputActionReference toggleRecordAction;

    [Header("RGB Capture")]
    [Tooltip("Camera used for RGB snapshots.")]
    public Camera rgbCamera;
    [Tooltip("Desired frame width in pixels.")]
    public int captureWidth = 640;
    [Tooltip("Desired frame height in pixels.")]
    public int captureHeight = 480;
    [Tooltip("Fixed aspect ratio for the capture (width/height). Set to 0 to use actual camera aspect ratio.")]
    public float fixedAspectRatio = 4f/3f;

    [Header("Recording Settings")]
    [Tooltip("Frames per second to record poses and images.")]
    public float recordFrameRate = 30f;

    [Header("Output Paths (relative to Assets)")]
    [Tooltip("CSV file path, e.g. 'Logs/point_3_log.csv'")]
    public string csvFilePath = "Logs/point_3_log.csv";
    [Tooltip("RGB folder path, e.g. 'Logs/RGBFrames'")]
    public string rgbDirPath = "Logs/RGBFrames";

    private bool isRecording = false;
    private StreamWriter writer;
    private float recordStartTime;
    private float nextRecordTime;
    private int frameIndex = 0;

    private UnityEngine.XR.InputDevice leftDevice;
    private UnityEngine.XR.InputDevice rightDevice;

    void OnEnable()
    {
        toggleRecordAction.action.performed += OnTogglePerformed;
        toggleRecordAction.action.Enable();
    }

    void OnDisable()
    {
        toggleRecordAction.action.performed -= OnTogglePerformed;
        toggleRecordAction.action.Disable();
    }

    private void OnTogglePerformed(InputAction.CallbackContext ctx)
    {
        if (!isRecording) StartRecording();
        else StopRecording();
    }

    void Update()
    {
        if (!isRecording) return;

        float elapsed = Time.realtimeSinceStartup - recordStartTime;
        if (elapsed >= nextRecordTime)
        {
            RecordFrame(elapsed);
            nextRecordTime += 1f / recordFrameRate;
        }
    }

    private void StartRecording()
    {
        // Prepare directories
        string csvFull = ResolvePath(csvFilePath);
        string rgbFullDir = ResolvePath(rgbDirPath);
        Directory.CreateDirectory(Path.GetDirectoryName(csvFull));
        Directory.CreateDirectory(rgbFullDir);

        // Open CSV and write header
        writer = new StreamWriter(csvFull);
        writer.WriteLine("key,timestamp," +
            "head_pos_x,head_pos_y,head_pos_z,head_dir_x,head_dir_y,head_dir_z,head_dir_w," +
            "left_controller_pos_x,left_controller_pos_y,left_controller_pos_z,left_controller_dir_x,left_controller_dir_y,left_controller_dir_z,left_controller_dir_w," +
            "right_controller_pos_x,right_controller_pos_y,right_controller_pos_z,right_controller_dir_x,right_controller_dir_y,right_controller_dir_z,right_controller_dir_w," +
            "xr_origin_pos_x,xr_origin_pos_y,xr_origin_pos_z," +
            "IndexTrigger_0,IndexTrigger_1,HandTrigger_0,HandTrigger_1,Thumbstick_0_x,Thumbstick_0_y,Thumbstick_1_x,Thumbstick_1_y");

        isRecording = true;
        recordStartTime = Time.realtimeSinceStartup;
        nextRecordTime = 0f;
        frameIndex = 0;

        // Initialize XR devices
        leftDevice = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        rightDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);

        Debug.Log($"[Recorder] Started recording to CSV: {csvFull}");
    }

    private void StopRecording()
    {
        isRecording = false;
        writer?.Close();
        writer = null;
        Debug.Log($"[Recorder] Stopped recording. Total frames: {frameIndex}");
    }

    private void RecordFrame(float timestamp)
    {
        // Poses (using local positions for controllers and camera)
        Vector3 headPos = mainCamera.localPosition;
        Quaternion headRot = mainCamera.rotation;
        Vector3 leftPos = leftController.localPosition;
        Quaternion leftRot = leftController.rotation;
        Vector3 rightPos = rightController.localPosition;
        Quaternion rightRot = rightController.rotation;
        
        // XR Origin world position
        Vector3 xrOriginPos = xrOrigin.position;

        // XR inputs
        leftDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.trigger, out float idx0);
        rightDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.trigger, out float idx1);
        leftDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.grip, out float grip0);
        rightDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.grip, out float grip1);
        leftDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxis, out Vector2 stick0);
        rightDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxis, out Vector2 stick1);

        // Write CSV line
        writer.Write($"{frameIndex},{timestamp:F4}," +
            $"{headPos.x:F4},{headPos.y:F4},{headPos.z:F4},{headRot.x:F4},{headRot.y:F4},{headRot.z:F4},{headRot.w:F4}," +
            $"{leftPos.x:F4},{leftPos.y:F4},{leftPos.z:F4},{leftRot.x:F4},{leftRot.y:F4},{leftRot.z:F4},{leftRot.w:F4}," +
            $"{rightPos.x:F4},{rightPos.y:F4},{rightPos.z:F4},{rightRot.x:F4},{rightRot.y:F4},{rightRot.z:F4},{rightRot.w:F4}," +
            $"{xrOriginPos.x:F4},{xrOriginPos.y:F4},{xrOriginPos.z:F4}," +
            $"{idx0:F4},{idx1:F4},{grip0:F4},{grip1:F4},{stick0.x:F4},{stick0.y:F4},{stick1.x:F4},{stick1.y:F4}");
        writer.WriteLine();
        writer.Flush();

        // Capture RGB image
        CaptureRGBImage(frameIndex);

        frameIndex++;
    }

    private void CaptureRGBImage(int key)
    {
        int width = captureWidth;
        int height = captureHeight;
        
        // If a fixed aspect ratio is specified (not 0), adjust the camera's aspect ratio
        if (fixedAspectRatio > 0)
        {
            rgbCamera.aspect = fixedAspectRatio;
        }
        
        RenderTexture rt = RenderTexture.GetTemporary(width, height, 24);
        RenderTexture prev = rgbCamera.targetTexture;
        rgbCamera.targetTexture = rt;
        rgbCamera.Render();

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        rgbCamera.targetTexture = prev;
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        byte[] jpg = tex.EncodeToJPG();
        Destroy(tex);

        string filePath = Path.Combine(ResolvePath(rgbDirPath), $"s{key}.jpg");
        File.WriteAllBytes(filePath, jpg);
    }

    private string ResolvePath(string path)
    {
        return Path.IsPathRooted(path) ? path : Path.Combine(Application.dataPath, path);
    }
}
