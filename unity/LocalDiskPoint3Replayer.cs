using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;
using RootMotion.FinalIK;  // Final IK namespace
using Unity.XR.CoreUtils;
using UnityEngine.InputSystem.XR;
using Meta.XR.Simulator.Editor;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
using UnityEngine.XR.Interaction.Toolkit;   // for ActionBasedController


public class LocalDiskPoint3Replayer : MonoBehaviour
{
    [Header("Avatar Transforms")]
    public Transform mainCamera;
    public Transform leftController;
    public Transform rightController;

    [Header("Data Paths")]
    public string csvFilePath = "Assets/point_3_log.csv";
    public string rgbDirPath = "Assets/RGBFrames";
    public string depthDirPath = "Assets/DepthFrames";

    [Header("UI Displays")]
    public RawImage rgbDisplay;
    public RawImage depthDisplay;

    [Header("Playback Settings")]
    public int startKey = 0;
    public int endKey = int.MaxValue;
    public int keyStep = 5;
    public float speed = 1f;

    [Header("Final IK Calibration")]
    [SerializeField] private VRIK ik;                        // Drag your avatar’s VRIK component here
    public enum CalibrationMode { HeightOnly, ArmsOnly, Both, None }

    [Header("Calibration Mode")]
    [SerializeField] private CalibrationMode mode = CalibrationMode.HeightOnly;

    [Header("Height Calibration")]
    [SerializeField] private float avatarDefaultHeight = 1.8f;
    [SerializeField] private float minAllowedHeight = 1.35f;
    [SerializeField] private float maxAllowedHeight = 2.2f;

    [Header("Arm-Span Calibration")]
    [SerializeField] private float avatarDefaultArmSpan = 1.6f; // meters
    [SerializeField] private float minArmSpan = 0.5f;
    [SerializeField] private float maxArmSpan = 2.2f;
    public enum PoseSpace { World, Local }

    [Header("Playback Settings")]
    [SerializeField] private PoseSpace headPoseSpace = PoseSpace.World;
    [SerializeField] private PoseSpace controllerPoseSpace = PoseSpace.Local;

    [Tooltip("Drag your VirtualKeySimulator here")]
    public VirtualKeySimulator keySimulator;

    private TrackedPoseDriver trackedPoseDriver;
    private ActionBasedController leftABC;
    private ActionBasedController rightABC;
    // private AxisControl trigger0, grip0, trigger1, grip1;


    class Record
    {
        public int key;
        public float time;
        public Vector3 headPos; public Quaternion headRot;
        public Vector3 leftPos; public Quaternion leftRot;
        public Vector3 rightPos; public Quaternion rightRot;
        public float indexTrigger0, indexTrigger1;
        public float handTrigger0, handTrigger1;
    }

    List<Record> records;
    int currentIndex = 0;
    float replayStartTime;

    Dictionary<int, string> rgbFiles;
    Dictionary<int, string> depthFiles;

    void Start()
    {
        // 0) Disable tracking from device:
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


        // 1) Resolve all three paths
        string csvPath = Resolve(csvFilePath);
        string rgbPath = Resolve(rgbDirPath);
        string depthPath = Resolve(depthDirPath);

        // 2) Debug: print them and whether they exist
        Debug.Log($"[Replayer] CSV path:   {csvPath}  exists? {File.Exists(csvPath)}");
        Debug.Log($"[Replayer] RGB folder: {rgbPath}  exists? {Directory.Exists(rgbPath)}");
        Debug.Log($"[Replayer] Depth folder:{depthPath} exists? {Directory.Exists(depthPath)}");

        // 3) Guard early so we don’t continue with missing files
        if (!File.Exists(csvPath))
        {
            Debug.LogError($"[Replayer] Aborting: CSV not found at {csvPath}");
            enabled = false; return;
        }
        if (!Directory.Exists(rgbPath))
        {
            Debug.LogError($"[Replayer] Aborting: RGB directory not found at {rgbPath}");
            enabled = false; return;
        }
        if (!Directory.Exists(depthPath))
        {
            Debug.LogError($"[Replayer] Aborting: Depth directory not found at {depthPath}");
            enabled = false; return;
        }

        // 4) Load & report how many records
        var all = LoadRecords(csvPath);
        Debug.Log($"[Replayer] LoadRecords() returned {all.Count} total entries");

        records = all
            .Where(r => r.key >= startKey && r.key <= endKey && ((r.key - startKey) % keyStep) == 0)
            .OrderBy(r => r.time)
            .ToList();
        Debug.Log($"[Replayer] Records after filter: {records.Count} (firstKey={(records.Count > 0 ? records[0].key : -1)}, lastKey={(records.Count > 0 ? records[^1].key : -1)})");

        if (records.Count == 0)
        {
            Debug.LogError("[Replayer] No records remain after filtering; aborting.");
            enabled = false; return;
        }

        // 5) Normalize timestamps
        float t0 = records[0].time;
        for (int i = 0; i < records.Count; i++)
        {
            records[i].time -= t0;
        }
        Debug.Log($"[Replayer] Normalized times. First record now at t=0, last at t={records[^1].time:F2}s");

        // 6) calibrate by averaging the first few recorded head heights --
        // compute avgHeadY & avgUserSpan
        int sampleCount = 30;
        float stabilityThreshold = 0.02f;  // 2 cm tolerance
        int stableIndex = -1;
        int maxStart = Mathf.Max(0, records.Count - sampleCount);

        // find a stable head-height window
        for (int start = 0; start <= maxStart; start++)
        {
            float minY = float.MaxValue, maxY = float.MinValue;
            for (int j = 0; j < sampleCount; j++)
            {
                float y = records[start + j].headPos.y;
                minY = Mathf.Min(minY, y);
                maxY = Mathf.Max(maxY, y);
            }
            if (maxY - minY <= stabilityThreshold)
            {
                stableIndex = start;
                break;
            }
        }

        // choose window start & actual frame count
        int windowStart = (stableIndex >= 0 ? stableIndex : 0);
        int count = Mathf.Min(sampleCount, records.Count);

        // average head-Y and arm-span over that window
        float sumHeadY = 0f;
        float sumSpan = 0f;
        for (int i = windowStart; i < windowStart + count; i++)
        {
            var r = records[i];
            sumHeadY += r.headPos.y;
            sumSpan += Vector3.Distance(r.leftPos, r.rightPos);
        }

        float avgHeadY = sumHeadY / count;
        float avgUserSpan = sumSpan / count;

        // choose calibration
        switch (mode)
        {
            case CalibrationMode.HeightOnly:
                CalibrateHeight(avgHeadY);
                break;
            case CalibrationMode.ArmsOnly:
                CalibrateArms(avgUserSpan);
                break;
            case CalibrationMode.Both:
                CalibrateHeight(avgHeadY);
                CalibrateArms(avgUserSpan);
                break;
            case CalibrationMode.None:
                break;
        }

        // 7) Index image files
        rgbFiles = IndexDirectory(rgbPath, @"s(\d+)\.jpg");
        depthFiles = IndexDirectory(depthPath, @"depth_(\d+)\.png");

        // Set rawImage transparency
        Color c = rgbDisplay.color;
        c.a = 1f;
        rgbDisplay.color = c;
        Color c2 = depthDisplay.color;
        c2.a = 1f;
        depthDisplay.color = c2;


        // 8) Final checks
        Debug.Log($"[Replayer] RGB files indexed:   {rgbFiles.Count}");
        Debug.Log($"[Replayer] Depth files indexed: {depthFiles.Count}");

        // 9) Start clock
        replayStartTime = Time.time;
    }

    // Measures the HMD height and scales the avatar root via VRIK.
    private void CalibrateByHeight()
    {
        // Read raw HMD height; requires floor-level tracking origin :contentReference[oaicite:8]{index=8}
        float rawHeight = mainCamera.localPosition.y;
        // Clamp to human height range :contentReference[oaicite:9]{index=9}
        float userHeight = Mathf.Clamp(rawHeight, minAllowedHeight, maxAllowedHeight);
        // Compute scale vs. avatarDefaultHeight :contentReference[oaicite:10]{index=10}
        float scaleFactor = userHeight / avatarDefaultHeight;
        // Apply to VRIK root to uniformly scale entire avatar :contentReference[oaicite:11]{index=11}
        ik.references.root.localScale = Vector3.one * scaleFactor;
        Debug.Log($"[Replayer] Height calib: user={userHeight:F2}m, model={avatarDefaultHeight:F2}m → scale={scaleFactor:F2}");
    }

    // Finds the first span of `sampleCount` frames where head Y varies by ≤ stabilityThreshold,
    // then averages that window (or falls back to the very first frames).
    private void CalibrateByRecordedHeight(
        List<Record> records,
        int sampleCount = 30,
        float stabilityThreshold = 0.02f  // e.g. 2 cm tolerance
    )
    {
        int stableIndex = -1;
        int maxStart = records.Count - sampleCount;

        // 1) Search for a stable window
        for (int start = 0; start <= maxStart; start++)
        {
            float minY = float.MaxValue, maxY = float.MinValue;
            for (int j = 0; j < sampleCount; j++)
            {
                float y = records[start + j].headPos.y;
                minY = Mathf.Min(minY, y);
                maxY = Mathf.Max(maxY, y);
            }
            if (maxY - minY <= stabilityThreshold)
            {
                stableIndex = start;
                break;
            }
        }

        // 2) Accumulate the chosen window (or fallback to first frames)
        int count = Mathf.Min(sampleCount, records.Count);
        float sum = 0f;
        if (stableIndex >= 0)
        {
            Debug.Log($"[Replayer] Using stable window starting at index {stableIndex}");
            for (int i = stableIndex; i < stableIndex + count; i++)
                sum += records[i].headPos.y;
        }
        else
        {
            Debug.LogWarning($"[Replayer] No stable window found; using first {count} frames");
            for (int i = 0; i < count; i++)
                sum += records[i].headPos.y;
        }

        // 3) Compute average, clamp, scale exactly as before
        float avgHeadY = sum / count;
        float userHeight = Mathf.Clamp(avgHeadY, minAllowedHeight, maxAllowedHeight);
        float scaleFactor = userHeight / avatarDefaultHeight;
        ik.references.root.localScale = Vector3.one * scaleFactor;

        Debug.Log($"[Replayer] Stable-height calib: avgHeadY={avgHeadY:F2}m → user={userHeight:F2}m → scale={scaleFactor:F2}");
    }

    private void CalibrateHeight(float avgHeadY)
    {
        float clamped = Mathf.Clamp(avgHeadY, minAllowedHeight, maxAllowedHeight);
        float scaleY = clamped / avatarDefaultHeight;
        // preserve existing X and Z:
        Vector3 s = ik.references.root.localScale;
        ik.references.root.localScale = new Vector3(s.x, s.y * scaleY, s.z);
        Debug.Log($"Height only → scaleY={scaleY:F2}");
    }

    private void CalibrateArms(float avgSpan)
    {
        float clamped = Mathf.Clamp(avgSpan, minArmSpan, maxArmSpan);
        float scaleXZ = clamped / avatarDefaultArmSpan;
        // preserve existing Y:
        Vector3 s = ik.references.root.localScale;
        ik.references.root.localScale = new Vector3(s.x * scaleXZ, s.y, s.z * scaleXZ);
        Debug.Log($"Arms only → scaleXZ={scaleXZ:F2} →x={s.x * scaleXZ:F2} y={s.y:F2} z={s.z * scaleXZ:F2} ");
    }


    void Update()
    {
        if (records == null)
        {
            Debug.LogWarning("[Replayer] records==null; skipping Update.");
            return;
        }

        if (currentIndex >= records.Count)
        {
            // Re-enable live inputs afterward
            if (trackedPoseDriver != null)
            {
                trackedPoseDriver.enabled = true;
                Debug.Log("[Replayer] Replay ended — re-enabled TrackedPoseDriver");
            }
            
            leftABC.enableInputActions = true;
            rightABC.enableInputActions = true;
            Debug.Log("[Replay] ActionBasedController.inputActions re-enabled");

            return;
        }
        float elapsed = (Time.time - replayStartTime) * speed;
        //Debug.Log($"[Replayer] t={elapsed:F2}s, nextRecordTime={records[currentIndex].time:F2}s");

        while (currentIndex < records.Count && records[currentIndex].time <= elapsed)
        {
            var r = records[currentIndex];
            Debug.Log($"[Replayer] Applying frame #{currentIndex} key={r.key} @ t={r.time:F2}s");
            ApplyPose(r);
            SimulateTriggers(r);
            DisplayImages(r.key);
            currentIndex++;
        }
    }

    string Resolve(string path)
    {
        if (Path.IsPathRooted(path)) return path;
        return Path.Combine(Application.dataPath, path);
    }

    List<Record> LoadRecords(string csvPath)
    {
        var list = new List<Record>();
        using var reader = new StreamReader(csvPath);

        // 1) Read header
        var headerLine = reader.ReadLine();
        if (headerLine == null)
        {
            Debug.LogError("[Replayer] CSV is empty!");
            return list;
        }
        var headers = SplitCsvLine(headerLine);
        var idx = headers
            .Select((h, i) => (h, i))
            .ToDictionary(p => p.h, p => p.i);

        // ensure required columns exist...
        // (omitted for brevity; same as before)

        double firstTs = double.NaN;
        int lineNum = 1;
        Record lastGood = null;

        while (!reader.EndOfStream)
        {
            lineNum++;
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;

            var tok = SplitCsvLine(line);
            if (tok.Length < headers.Length)
            {
                Debug.LogWarning($"Line {lineNum}: only {tok.Length} cols (need {headers.Length}), skipping");
                continue;
            }

            // parse key
            if (!int.TryParse(tok[idx["key"]], NumberStyles.Integer, CultureInfo.InvariantCulture, out int key))
            {
                Debug.LogError($"Line {lineNum}: bad key='{tok[idx["key"]]}' — skipping");
                continue;
            }

            // parse timestamp
            var tsStr = tok[idx["timestamp"]];
            if (!double.TryParse(tsStr, NumberStyles.Float, CultureInfo.InvariantCulture, out double tsDbl))
            {
                Debug.LogError($"Line {lineNum}: bad timestamp='{tsStr}' — skipping");
                continue;
            }
            if (double.IsNaN(firstTs)) firstTs = tsDbl;
            float timeSec = (float)((tsDbl - firstTs) / 1000.0);

            // helpers that fallback to lastGood
            float SafeF(string col, float fallback)
            {
                var s = tok[idx[col]];
                if (float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    return v;
                // either empty or invalid: log once and return fallback
                Debug.LogWarning($"Line {lineNum}: invalid {col}='{s}', using fallback {fallback}");
                return fallback;
            }

            Vector3 headPos, leftPos, rightPos;
            Quaternion headRot, leftRot, rightRot;

            if (lastGood != null)
            {
                headPos = lastGood.headPos;
                leftPos = lastGood.leftPos;
                rightPos = lastGood.rightPos;
                headRot = lastGood.headRot;
                leftRot = lastGood.leftRot;
                rightRot = lastGood.rightRot;
            }
            else
            {
                headPos = Vector3.zero;
                leftPos = Vector3.zero;
                rightPos = Vector3.zero;
                headRot = Quaternion.identity;
                leftRot = Quaternion.identity;
                rightRot = Quaternion.identity;
            }

            // now overwrite each component if it parses successfully:
            headPos.x = SafeF("head_pos_x", headPos.x);
            headPos.y = SafeF("head_pos_y", headPos.y);
            headPos.z = SafeF("head_pos_z", headPos.z);

            headRot.x = SafeF("head_dir_x", headRot.x);
            headRot.y = SafeF("head_dir_y", headRot.y);
            headRot.z = SafeF("head_dir_z", headRot.z);
            headRot.w = SafeF("head_dir_w", headRot.w);

            leftPos.x = SafeF("left_controller_pos_x", leftPos.x);
            leftPos.y = SafeF("left_controller_pos_y", leftPos.y);
            leftPos.z = SafeF("left_controller_pos_z", leftPos.z);

            leftRot.x = SafeF("left_controller_dir_x", leftRot.x);
            leftRot.y = SafeF("left_controller_dir_y", leftRot.y);
            leftRot.z = SafeF("left_controller_dir_z", leftRot.z);
            leftRot.w = SafeF("left_controller_dir_w", leftRot.w);

            rightPos.x = SafeF("right_controller_pos_x", rightPos.x);
            rightPos.y = SafeF("right_controller_pos_y", rightPos.y);
            rightPos.z = SafeF("right_controller_pos_z", rightPos.z);

            rightRot.x = SafeF("right_controller_dir_x", rightRot.x);
            rightRot.y = SafeF("right_controller_dir_y", rightRot.y);
            rightRot.z = SafeF("right_controller_dir_z", rightRot.z);
            rightRot.w = SafeF("right_controller_dir_w", rightRot.w);

            // build and store the record
            var rec = new Record
            {
                key = key,
                time = timeSec,
                headPos = headPos,
                headRot = headRot,
                leftPos = leftPos,
                leftRot = leftRot,
                rightPos = rightPos,
                rightRot = rightRot,
                indexTrigger0 = SafeF("IndexTrigger_0", 0f),
                indexTrigger1 = SafeF("IndexTrigger_1", 0f),
                handTrigger0 = SafeF("HandTrigger_0", 0f),
                handTrigger1 = SafeF("HandTrigger_1", 0f),
            };
            list.Add(rec);
            lastGood = rec;
        }

        Debug.Log($"[Replayer] Parsed {list.Count} usable records with fallback.");
        return list;
    }


    // Splits one CSV line into fields, respecting quotes (and doubled quotes inside).
    static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        bool inQuotes = false;
        var sb = new System.Text.StringBuilder();

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                // If this is a double-quote inside quoted field, consume the escape:
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    sb.Append('"');
                    i++; // skip next quote
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if (c == ',' && !inQuotes)
            {
                fields.Add(sb.ToString());
                sb.Clear();
            }
            else
            {
                sb.Append(c);
            }
        }
        // final field
        fields.Add(sb.ToString());
        return fields.ToArray();
    }


    Dictionary<int, string> IndexDirectory(string dir, string pattern)
    {
        var dict = new Dictionary<int, string>();
        var rx = new Regex(pattern, RegexOptions.IgnoreCase);
        foreach (var file in Directory.EnumerateFiles(dir))
        {
            var name = Path.GetFileName(file);
            var m = rx.Match(name);
            if (m.Success && int.TryParse(m.Groups[1].Value, out var k))
            {
                dict[k] = file;
                Debug.Log($"[Replayer] Indexed file for key={k}: {name}");
            }
        }
        return dict;
    }
    void ApplyPose(Record r)
    {
        // — HEAD —
        if (mainCamera != null)
        {
            if (headPoseSpace == PoseSpace.World)
                mainCamera.SetPositionAndRotation(r.headPos, r.headRot);
            else
            {
                mainCamera.localPosition = r.headPos;
                mainCamera.localRotation = r.headRot;
            }
        }

        // — LEFT CONTROLLER —
        if (leftController != null)
        {
            if (controllerPoseSpace == PoseSpace.World)
                leftController.SetPositionAndRotation(r.leftPos, r.leftRot);
            else
            {
                leftController.localPosition = r.leftPos;
                leftController.localRotation = r.leftRot;
            }
        }

        // — RIGHT CONTROLLER —
        if (rightController != null)
        {
            if (controllerPoseSpace == PoseSpace.World)
                rightController.SetPositionAndRotation(r.rightPos, r.rightRot);
            else
            {
                rightController.localPosition = r.rightPos;
                rightController.localRotation = r.rightRot;
            }
        }
    }

    void InitializeTriggerControls()
    {
        // LEFT
        leftABC = leftController.GetComponent<ActionBasedController>();
        rightABC = rightController.GetComponent<ActionBasedController>();

        leftABC.gameObject.SetActive(true);
        rightABC.gameObject.SetActive(true);
    }

    void SimulateTriggers(Record r)
    {
        // Left grab
        if (r.handTrigger0 > 0.1f)  // trigger value in recording
        {
            keySimulator.SetKeyState(Key.Y, true);
        }
        else
        {
            // Simulate key up
            keySimulator.SetKeyState(Key.Y, false);
        }
        // Right grab
        if (r.handTrigger1 > 0.1f)  // trigger value in recording
        {
            keySimulator.SetKeyState(Key.B, true);
        }
        else
        {
            // Simulate key up
            keySimulator.SetKeyState(Key.B, false);
        }
    }

    void DisplayImages(int key)
    {
        if (rgbDisplay == null) Debug.LogWarning("[Replayer] rgbDisplay RawImage is null!");
        if (depthDisplay == null) Debug.LogWarning("[Replayer] depthDisplay RawImage is null!");

        if (rgbFiles.TryGetValue(key, out var rgbPath))
        {
            var data = File.ReadAllBytes(rgbPath);
            var tex = new Texture2D(2, 2);
            tex.LoadImage(data);
            rgbDisplay.texture = tex;
        }
        else Debug.LogWarning($"[Replayer] No RGB frame for key={key}");

        if (depthFiles.TryGetValue(key, out var dp))
        {
            var data = File.ReadAllBytes(dp);
            var tex = new Texture2D(2, 2);
            tex.LoadImage(data);
            depthDisplay.texture = tex;
        }
        else Debug.LogWarning($"[Replayer] No Depth frame for key={key}");
    }
}
