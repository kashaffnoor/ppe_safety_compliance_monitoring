# Safety Compliance Monitoring - PPE Detection System

A real-time Personal Protective Equipment (PPE) detection system for factory floor monitoring using YOLOv8 and computer vision.

## Features

- Real-time detection of safety gear (helmet detection)
- Support for both webcam and video file input
- Visual warnings when safety gear is missing
- Compliance tracking and statistics
- Clean, professional web interface

## Quick Start

### Option 1: Use Pre-trained Model (Our System - Without GPU)

The system will automatically use a pre-trained YOLOv8 model. This provides general object detection capability and works immediately on any system.

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run src/app.py
```

Or use the batch file:
```bash
run_app.bat
```

The app will automatically use a pre-trained YOLOv8 model. Click "Load Model" in the sidebar, then select webcam or upload a video.

### Option 2: Train Custom Model (Requires GPU)

For better accuracy with PPE-specific detection, you can train on the Hard Hat Workers dataset. **Note: Training requires a GPU with CUDA support. Your current CPU does not support the required training libraries.**

**If you have access to a GPU-enabled machine:**

1. **Activate virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Train the model:**
   ```bash
   python src/train.py
   ```

3. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

**Dataset Information:**
- 7,035 images (4,216 train, 1,053 valid, 1,763 test)
- 27,039 annotations
- Classes: head, helmet, person
- Helmet detection for PPE compliance

## Usage

1. Open the web interface (default: http://localhost:8501)
2. Click "Load Model" in the sidebar
3. Choose input source:
   - **Webcam**: Select "Webcam" and click "Start Detection"
   - **Video File**: Upload a video file and click "Start Detection"
4. The system will:
   - Detect workers and safety gear in real-time
   - Show green boxes for detected safety gear
   - Show red warning banners when gear is missing
   - Track compliance statistics in the sidebar

## Configuration

Default settings in `src/config.py`:

**Model Configuration:**
- **Model Type**: yolov8n (nano - fastest)
- **Confidence Threshold**: 0.5 (50% confidence required)
- **IOU Threshold**: 0.45 (for non-maximum suppression)

**Required PPE Classes:**
- Helmet (class 1) - Required for PPE compliance

**Video Processing:**
- **Target FPS**: 30
- **Display Resolution**: 1280x720

You can adjust these values in `src/config.py` based on your needs.

## Project Structure

```
Safety Compliance Monitoring/
├── data/                  # Dataset directory
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── src/                   # Source code
│   ├── app.py            # Streamlit GUI
│   ├── detection.py      # Detection pipeline
│   ├── train.py          # Training script
│   ├── config.py         # Configuration
│   └── utils.py          # Utility functions
├── models/               # Trained models
├── logs/                 # Log files
├── runs/                 # Training runs
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── QUICK_START.md       # Quick start guide
```

## Model Performance

**Pre-trained YOLOv8n:**
- General object detection (80 classes)
- Good for demos and testing
- Fast inference (~30 FPS)

**Custom Trained PPE Model (Hard Hat Workers Dataset):**
- Specialized for helmet detection
- Classes: head, helmet, person
- Expected mAP@50: ~0.85-0.90
- Real-time inference: ~25-30 FPS on modern hardware

## Troubleshooting

**Webcam not working:**
- Check browser camera permissions
- Ensure no other application is using the camera
- Try a different browser (Chrome recommended)

**Low FPS:**
- Reduce display resolution in `config.py`
- Close other applications
- Use a smaller model (yolov8n is already the smallest)

**Model not loading:**
- Click "Load Model" button in the sidebar first
- Check console for error messages
- Ensure dependencies are installed correctly

**Training fails:**
- Verify dataset structure matches the required format
- Check that `data/data.yaml` exists and has correct paths
- Ensure images and labels exist in the directories
- See QUICK_START.md for dataset download options

## Requirements

- Python 3.8 or higher
- Windows, Linux, or macOS
- Webcam (for live detection)
- GPU recommended for training (optional, CPU works)

## License

This project uses the following open-source libraries:
- Ultralytics YOLOv8 (AGPL-3.0)
- OpenCV (Apache 2.0)
- Streamlit (Apache 2.0)

## Support

For detailed setup instructions and dataset options, see [QUICK_START.md](QUICK_START.md).
