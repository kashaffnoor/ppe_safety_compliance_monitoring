# Quick Start Guide

## Option 1: Use Pre-trained Model (Recommended)

The system will automatically use a pre-trained YOLOv8 model. This provides general object detection capability and works immediately on any system.

### Steps:
1. **Activate virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

   Or use the batch file:
   ```bash
   run_app.bat
   ```

4. **In the web interface:**
   - Click "Load Model" (will use pre-trained YOLOv8n)
   - Select "Webcam" or upload a video
   - Click "Start Detection"

**Note:** The pre-trained model does general object detection (80 classes) and works well for demos and testing.

## Option 2: Train Custom Model (Requires GPU)

For better accuracy with PPE-specific detection, you can train on the Hard Hat Workers dataset. **Note: Training requires a GPU with CUDA support.**

**If you have access to a GPU-enabled machine:**

1. **Activate virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Train the model:**
   ```bash
   python src/train.py
   ```

   This will train on the prepared dataset (7,035 images, 27,039 annotations).
   Training takes approximately 1-3 hours on GPU.

3. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

4. **In the web interface:**
   - Click "Load Model" (will use your trained model)
   - Select "Webcam" or upload a video
   - Click "Start Detection"

## Current Dataset Status

The Hard Hat Workers dataset is already prepared and configured:

**Dataset Information:**
- **Total Images:** 7,035 (4,216 train, 1,053 valid, 1,763 test)
- **Total Annotations:** 27,039
- **Classes:** head (0), helmet (1), person (2)
- **PPE Compliance:** Helmet detection required

**Current data.yaml:**
```yaml
path: ../data
train: train/images
val: valid/images
test: test/images

names:
  0: head
  1: helmet
  2: person

required_classes: [1]  # helmet
```

## Using a Different Dataset

If you want to use a different PPE dataset, organize it as:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Update `data.yaml` with your class names and update `src/config.py` to match your required PPE classes.

## Common Issues

**Issue:** Webcam not working
- **Solution:** Check browser permissions for camera access

**Issue:** Low FPS
- **Solution:** Reduce image size in config.py or close other applications

**Issue:** Model not detecting PPE well
- **Solution:** Train on a PPE-specific dataset using Option 2 above

**Issue:** "Model not loaded" error
- **Solution:** Click "Load Model" button in the sidebar first
