using UnityEngine;
using System.Collections;
using RootMotion.FinalIK;

/// <summary>
/// Automatically calibrates a Humanoid avatar's height and arm span
/// based on the user's VR rig (headset and controllers).
/// Attach this to a GameObject in your scene and assign references in the Inspector.
/// </summary>
public class VRAvatarCalibrator : MonoBehaviour
{
    public enum CalibrationMode { HeightOnly, ArmsOnly, Both }

    [Header("References")]
    [Tooltip("VRIK component on your Humanoid avatar")]
    [SerializeField] private VRIK ik;
    [Tooltip("Transform of the XR camera (user's head)")]
    [SerializeField] private Transform xrCamera;
    [Tooltip("Transform of the left-hand controller")]
    [SerializeField] private Transform leftController;
    [Tooltip("Transform of the right-hand controller")]
    [SerializeField] private Transform rightController;

    [Header("Calibration Settings")]
    [Tooltip("Which calibration(s) to perform on start")]
    [SerializeField] private CalibrationMode mode = CalibrationMode.Both;
    [Tooltip("Default height of your avatar model in meters")]
    [SerializeField] private float avatarDefaultHeight = 1.75f;
    [Tooltip("Minimum allowed user height to clamp against")]
    [SerializeField] private float minAllowedHeight = 1.2f;
    [Tooltip("Maximum allowed user height to clamp against")]
    [SerializeField] private float maxAllowedHeight = 2.1f;
    [Tooltip("Default arm span of your avatar model in meters")]
    [SerializeField] private float avatarDefaultArmSpan = 1.4f;
    [Tooltip("Minimum allowed user ArmSpan to clamp against")]
    [SerializeField] private float minAllowedArmSpan = 1.2f;
    [Tooltip("Maximum allowed user ArmSpan to clamp against")]
    [SerializeField] private float maxAllowedArmSpan = 2.1f;
    [Tooltip("Delay (seconds) before sampling user pose for calibration")]
    [SerializeField] private float calibrationDelay = 2f;

    private void Start()
    {
        // Begin calibration after a short delay
        StartCoroutine(CalibrationRoutine());
    }

    private IEnumerator CalibrationRoutine()
    {
        yield return new WaitForSeconds(calibrationDelay);

        // Sample user head height and arm span
        float userHeadY = xrCamera.position.y;
        float userArmSpan = Vector3.Distance(leftController.position, rightController.position);

        // Perform selected calibration(s)
        switch (mode)
        {
            case CalibrationMode.HeightOnly:
                CalibrateHeight(userHeadY);
                break;
            case CalibrationMode.ArmsOnly:
                CalibrateArms(userArmSpan);
                break;
            case CalibrationMode.Both:
                CalibrateHeight(userHeadY);
                CalibrateArms(userArmSpan);
                break;
        }
    }

    /// <summary>
    /// Scales the avatar's root transform on the Y axis to match the user's height.
    /// </summary>
    /// <param name="headY">User's head Y position in world space</param>
    private void CalibrateHeight(float headY)
    {
        float clampedY = Mathf.Clamp(headY, minAllowedHeight, maxAllowedHeight);
        float scaleY = clampedY / avatarDefaultHeight;
        Vector3 rootScale = ik.references.root.localScale;
        ik.references.root.localScale = new Vector3(rootScale.x, scaleY, rootScale.z);
        Debug.Log($"[Calibration] Height adjusted: userHeadY={headY:F2}, scaleY={scaleY:F2}");
    }

    /// <summary>
    /// Adjusts VRIK arm length multipliers to match the user's arm span.
    /// </summary>
    /// <param name="span">Distance between left and right controllers</param>
    private void CalibrateArms(float span)
    {
        float clampedX = Mathf.Clamp(span, minAllowedArmSpan, maxAllowedArmSpan);
        float scale = clampedX / avatarDefaultArmSpan;
        ik.solver.leftArm.armLengthMlp = scale;
        ik.solver.rightArm.armLengthMlp = scale;
        Debug.Log($"[Calibration] Arms adjusted: userSpan={span:F2}, scale={scale:F2}");
    }
}
