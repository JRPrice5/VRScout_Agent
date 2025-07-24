# OCR-Based Reward System Setup Guide

This guide explains how to set up and use the OCR-based reward system for ACT RL training.

## Overview

The OCR reward system consists of two main components:
1. **Python OCR Script** (`ocr_reward_system.py`) - Monitors screen area and sends scores via UDP
2. **Unity Reward Tracker** (`OCRRewardTracker.cs`) - Receives scores and tracks rewards per timestep

## Prerequisites

### Software Requirements
- **Tesseract OCR**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Default installation path: `D:\Tesseract-OCR\tesseract.exe`
  - Make sure to add Tesseract to your system PATH or note the installation path

### Python Dependencies
Install required Python packages:
```bash
pip install opencv-python mss pytesseract numpy pillow
```

## Setup Instructions

### 1. Install Tesseract OCR
1. Download Tesseract from the official repository
2. Install to `D:\Tesseract-OCR\` (or note your custom path)
3. Verify installation by running `tesseract --version` in command prompt

### 2. Configure Unity Client
1. Add the `OCRRewardTracker.cs` script to your Unity project
2. Attach the `OCRRewardTracker` component to the same GameObject as `InferencePoseClient`
3. Configure the OCR Reward Tracker in the Unity Inspector:

#### Inspector Settings:
```
OCR Reward System Settings:
├── Enable Reward Tracking: ✓ (checked)
├── OCR Server IP: 127.0.0.1
├── OCR UDP Port: 12345
├── RL Server IP: 127.0.0.1
└── RL Server Port: 5557

Reward Configuration:
├── Positive Step Reward: 1.0
├── Negative Step Reward: -0.1
├── Neutral Step Reward: 0.0
└── Episode End Timeout: 2.0

Debug Settings:
├── Show Debug Logs: ✓ (checked)
└── Show Reward GUI: ✓ (checked)
```

### 3. Start the OCR System

#### Option A: Using the Batch Script (Recommended)
1. Double-click `run_ocr_reward_system.bat`
2. The OCR system will start with a GUI interface

#### Option B: Command Line
```bash
cd d:\PythonProject\ALOHA\act\rl
python ocr_reward_system.py
```

#### Option C: Custom Configuration
```bash
python ocr_reward_system.py \
    --tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe" \
    --unity-host 127.0.0.1 \
    --unity-port 12345 \
    --ocr-frequency 10
```

## Usage Workflow

### 1. Setup Phase
1. **Start RL Training Server**:
   ```bash
   python rl/train_rl_example.py --pretrained_ckpt_path path/to/model.ckpt --ckpt_dir path/to/ckpt
   ```

2. **Start OCR System**:
   - Run `run_ocr_reward_system.bat`
   - Click "Select Screen Region" in the GUI
   - Click and drag to select the area where scores are displayed
   - Click "Start OCR" to begin monitoring

3. **Start Unity Client**:
   - Launch your Unity VR application
   - Ensure `OCRRewardTracker` component is enabled
   - Connect to the RL training server (port 5557)

### 2. Training Phase
1. **Episode Start**: 
   - Episode automatically starts when:
     - Unity client connects to RL server (`isConnected = true`)
     - OCR detects a non-zero score

2. **Reward Tracking**:
   - OCR monitors screen at 10Hz
   - Rewards are calculated per timestep:
     - **Positive reward** (+1.0): When score > 0
     - **Negative reward** (-0.1): When score = 0
     - **Neutral reward** (0.0): When OCR fails

3. **Episode End**:
   - Episode ends when:
     - OCR returns invalid/empty scores for >2 seconds
     - User manually ends episode via GUI
   - Rewards are automatically sent to RL training server

### 3. Monitoring
- **OCR GUI**: Shows current score and system status
- **Unity GUI**: Shows episode status and reward statistics
- **Console Logs**: Detailed debugging information

## Configuration Options

### OCR System Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tesseract-path` | `D:\Tesseract-OCR\tesseract.exe` | Path to Tesseract executable |
| `--unity-host` | `127.0.0.1` | Unity client IP address |
| `--unity-port` | `12345` | UDP port for communication |
| `--ocr-frequency` | `10` | OCR processing frequency (Hz) |
| `--region-*` | None | Manual screen region coordinates |
| `--no-gui` | False | Run without GUI interface |

### Unity Inspector Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `positiveStepReward` | 1.0 | Reward when score > 0 |
| `negativeStepReward` | -0.1 | Reward when score = 0 |
| `neutralStepReward` | 0.0 | Reward when OCR fails |
| `episodeEndTimeout` | 2.0 | Seconds before episode end |

## Troubleshooting

### Common Issues

#### 1. OCR Not Detecting Scores
**Symptoms**: OCR returns 0 or empty results
**Solutions**:
- Ensure screen region covers the score area completely
- Check if score text is clearly visible and not overlapped
- Adjust screen resolution or text size
- Verify Tesseract installation

#### 2. Unity Not Receiving Scores
**Symptoms**: Unity shows "Score: 0" continuously
**Solutions**:
- Check UDP port configuration (default: 12345)
- Verify firewall settings allow UDP traffic
- Ensure OCR system is running and connected
- Check Unity console for error messages

#### 3. RL Server Connection Failed
**Symptoms**: "RL Server Connected: False" in Unity GUI
**Solutions**:
- Verify RL training server is running on port 5557
- Check server IP address configuration
- Ensure no firewall blocking TCP connections
- Restart both Unity client and RL server

#### 4. Episode Not Starting
**Symptoms**: "Episode Active: False" despite valid scores
**Solutions**:
- Ensure Unity client is connected to RL server first
- Check if OCR is detecting non-zero scores
- Manually start episode using Unity GUI button
- Verify `enableRewardTracking` is checked

### Debug Information

#### OCR System Logs
```
OCR: '123' -> Score: 123
Sending score to Unity: 123
```

#### Unity Console Logs
```
[OCRRewardTracker] Score update: 123
[OCRRewardTracker] Episode started at timestep 45
[OCRRewardTracker] Timestep 46: Reward = 1.0
```

## Performance Considerations

### System Requirements
- **CPU**: OCR processing is CPU-intensive
- **Memory**: Minimal memory usage (~50MB)
- **Network**: Low bandwidth UDP communication

### Optimization Tips
1. **Reduce OCR frequency** if performance is an issue
2. **Select smaller screen regions** for faster processing
3. **Close unnecessary applications** during training
4. **Use SSD storage** for better I/O performance

## Advanced Configuration

### Custom Screen Regions
You can specify screen regions via command line:
```bash
python ocr_reward_system.py \
    --region-top 100 \
    --region-left 200 \
    --region-width 300 \
    --region-height 50
```

### Multiple Score Areas
For monitoring multiple score areas, run multiple OCR instances:
```bash
# Terminal 1 - Main score
python ocr_reward_system.py --unity-port 12345

# Terminal 2 - Bonus score  
python ocr_reward_system.py --unity-port 12346
```

### Integration with Existing Workflows
The OCR system can be integrated into existing training scripts:
```python
import subprocess
import time

# Start OCR system
ocr_process = subprocess.Popen([
    'python', 'ocr_reward_system.py', 
    '--no-gui', 
    '--region-top', '100',
    '--region-left', '200', 
    '--region-width', '300',
    '--region-height', '50'
])

# Run training
# ... your training code ...

# Stop OCR system
ocr_process.terminate()
```

## API Reference

### UDP Message Format (OCR → Unity)
```json
{
    "type": "score_update",
    "score": 123,
    "timestamp": 1642678901.234
}
```

### TCP Message Format (Unity → RL Server)
```json
{
    "type": "episode_data",
    "data": {
        "timesteps": [45, 46, 47, 48],
        "rewards": [1.0, 1.0, -0.1, 1.0],
        "done": true
    }
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review console logs for error messages
3. Verify all configuration settings
4. Test components individually (OCR, Unity, RL server)

## Version History

- **v1.0**: Initial implementation with basic OCR and reward tracking
- Features: Screen region selection, UDP communication, episode management
