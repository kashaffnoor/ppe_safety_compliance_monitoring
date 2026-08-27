"""
Streamlit GUI for PPE Detection System
Professional industrial-grade interface for safety compliance monitoring
Optimized for speed and performance
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from detection import PPEDetector
    from config import MODEL_CONFIG, REQUIRED_CLASSES, VIDEO_CONFIG, UI_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SafetyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional industrial interface - Optimized
st.markdown("""
<style>
    /* Fast, minimal CSS */
    .stApp {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    
    /* Status cards - Minimal styling */
    .status-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .status-card-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .status-card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Alert boxes */
    .alert-box {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .alert-details {
        font-size: 0.8rem;
        color: #fca5a5;
    }
    
    .success-box {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        border: 2px solid #059669;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .success-details {
        font-size: 0.8rem;
        color: #a7f3d0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .info-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .info-details {
        font-size: 0.8rem;
        color: #bfdbfe;
    }
    
    /* Progress bars */
    .progress-container {
        background: #1e293b;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
        margin-top: 0.25rem;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
    }
    
    .progress-green {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
    }
    
    .progress-red {
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
    }
    
    .progress-blue {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
    }
    
    /* Horizontal stats cards */
    .horizontal-stats {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.25rem;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
    }
    
    /* Video container */
    .video-container {
        border: 2px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'violations': 0,
            'compliant_detections': 0
        }
    if 'current_warning' not in st.session_state:
        st.session_state.current_warning = None
    if 'video_source' not in st.session_state:
        st.session_state.video_source = None
    if 'model_loaded_time' not in st.session_state:
        st.session_state.model_loaded_time = None
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

def load_model():
    """Load the trained PPE detection model"""
    try:
        with st.spinner("Loading model..."):
            model_paths = [
                'runs/train/ppe_detection/weights/best.pt',
                'runs/detect/ppe_detection/weights/best.pt',
                'yolov8n.pt'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    st.session_state.detector = PPEDetector(model_path, {
                        'MODEL_CONFIG': MODEL_CONFIG,
                        'REQUIRED_CLASSES': REQUIRED_CLASSES
                    })
                    st.session_state.model_loaded_time = time.time()
                    st.success(f"✅ Model loaded: {model_path}")
                    return True
            
            st.session_state.detector = PPEDetector('yolov8n.pt', {
                'MODEL_CONFIG': MODEL_CONFIG,
                'REQUIRED_CLASSES': REQUIRED_CLASSES
            })
            st.session_state.model_loaded_time = time.time()
            st.info("ℹ️ Using pre-trained YOLOv8n model")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return False

def process_webcam_frame():
    """Process frame from webcam - Optimized for speed"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG['display_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG['display_height'])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    
    try:
        while st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, frame_stats = st.session_state.detector.process_frame(frame)
            
            # Update statistics (less frequently for performance)
            st.session_state.frame_count += 1
            if st.session_state.frame_count % 5 == 0:
                st.session_state.stats = st.session_state.detector.get_statistics()
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            with frame_placeholder.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Update warning (less frequently)
            if st.session_state.frame_count % 5 == 0:
                if frame_stats['violations'] > 0:
                    st.session_state.current_warning = {
                        'type': 'violation',
                        'message': "⚠️ SAFETY VIOLATION DETECTED",
                        'details': f"Missing: {', '.join(frame_stats['missing_gear'])}",
                    }
                else:
                    st.session_state.current_warning = None
            
            # Display warning (only when changed)
            with warning_placeholder.container():
                if st.session_state.current_warning:
                    st.markdown(f"""
                    <div class="alert-box">
                        <div class="alert-title">{st.session_state.current_warning['message']}</div>
                        <div class="alert-details">{st.session_state.current_warning['details']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if not st.session_state.detector.required_class_ids:
                        st.markdown(f"""
                        <div class="info-box">
                            <div class="info-title">🎯 Detection Active</div>
                            <div class="info-details">General object detection mode</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <div class="success-title">✅ Compliant</div>
                            <div class="success-details">All safety gear detected</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Minimal delay for UI responsiveness
            time.sleep(0.01)
            
    finally:
        cap.release()

def process_video_file(uploaded_file):
    """Process uploaded video file - plays back at the video's natural speed"""
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file")
        os.remove(temp_path)
        return
    
    # Read the video's real frame rate so playback can be paced to match it
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30  # fallback if the file doesn't report a valid FPS
    
    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    
    start_time = time.time()
    
    try:
        while st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # If detection has fallen behind real time, jump ahead to the
            # frame that *should* be showing right now instead of processing
            # every frame in between - this is what keeps playback at
            # natural speed even when inference is slower than the frame rate.
            target_frame_idx = (time.time() - start_time) * video_fps
            current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if target_frame_idx - current_frame_idx > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Process frame
            annotated_frame, frame_stats = st.session_state.detector.process_frame(frame)
            
            # Update statistics (less frequently)
            st.session_state.frame_count += 1
            if st.session_state.frame_count % 5 == 0:
                st.session_state.stats = st.session_state.detector.get_statistics()
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            with frame_placeholder.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Update warning (less frequently)
            if st.session_state.frame_count % 5 == 0:
                if frame_stats['violations'] > 0:
                    st.session_state.current_warning = {
                        'type': 'violation',
                        'message': "⚠️ SAFETY VIOLATION DETECTED",
                        'details': f"Missing: {', '.join(frame_stats['missing_gear'])}",
                    }
                else:
                    st.session_state.current_warning = None
            
            # Display warning
            with warning_placeholder.container():
                if st.session_state.current_warning:
                    st.markdown(f"""
                    <div class="alert-box">
                        <div class="alert-title">{st.session_state.current_warning['message']}</div>
                        <div class="alert-details">{st.session_state.current_warning['details']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if not st.session_state.detector.required_class_ids:
                        st.markdown(f"""
                        <div class="info-box">
                            <div class="info-title">🎯 Detection Active</div>
                            <div class="info-details">General object detection mode</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <div class="success-title">✅ Compliant</div>
                            <div class="success-details">All safety gear detected</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Wait until this frame's natural display time has actually
            # arrived, so fast detection doesn't play the video too quickly
            next_frame_due = start_time + (cap.get(cv2.CAP_PROP_POS_FRAMES) / video_fps)
            sleep_time = next_frame_due - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    finally:
        cap.release()
        os.remove(temp_path)

def display_statistics():
    """Display detection statistics - Horizontal layout at bottom"""
    stats = st.session_state.stats
    
    # Calculate metrics
    if stats['total_frames'] > 0:
        compliance_rate = ((stats['total_frames'] - stats['violations']) / stats['total_frames']) * 100
        detection_rate = (stats['total_detections'] / stats['total_frames']) * 100
    else:
        compliance_rate = 100
        detection_rate = 0
    
    # Horizontal stats row
    st.markdown('<div class="horizontal-stats">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-icon">📊</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{stats["total_frames"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Frames</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-icon">🎯</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{stats["total_detections"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Detections</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card" style="border-color: #dc2626;">', unsafe_allow_html=True)
        st.markdown('<div class="stat-icon">⚠️</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value" style="color: #ef4444;">{stats["violations"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Violations</div>', unsafe_allow_html=True)
        if stats['total_frames'] > 0:
            violation_rate = (stats['violations'] / stats['total_frames']) * 100
            st.markdown(f'<div class="progress-container"><div class="progress-bar progress-red" style="width: {violation_rate}%"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-card" style="border-color: #059669;">', unsafe_allow_html=True)
        st.markdown('<div class="stat-icon">✅</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value" style="color: #10b981;">{compliance_rate:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Compliance</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="progress-container"><div class="progress-bar progress-green" style="width: {compliance_rate}%"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="stat-card" style="border-color: #3b82f6;">', unsafe_allow_html=True)
        st.markdown('<div class="stat-icon">📈</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value" style="color: #3b82f6;">{detection_rate:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Detect Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="progress-container"><div class="progress-bar progress-blue" style="width: {min(detection_rate, 100)}%"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_system_status():
    """Display system status - Compact"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.detector:
            model_status = "✅ Model Loaded"
            status_color = "#10b981"
        else:
            model_status = "⚠️ Model Not Loaded"
            status_color = "#f59e0b"
        
        st.markdown(f'<div class="status-card">', unsafe_allow_html=True)
        st.markdown('<div class="status-card-title">AI Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-card-value" style="color: {status_color};">{model_status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.processing:
            detection_status = "🔴 Active"
            status_color = "#ef4444"
        else:
            detection_status = "⚪ Idle"
            status_color = "#64748b"
        
        st.markdown(f'<div class="status-card">', unsafe_allow_html=True)
        st.markdown('<div class="status-card-title">Detection</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-card-value" style="color: {status_color};">{detection_status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🛡️ SafetyGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Real-time PPE Detection & Safety Compliance Monitoring</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # Model loading
        st.markdown("#### 🤖 Model")
        if st.button("📥 Load Model", type="primary", use_container_width=True):
            load_model()
        
        st.markdown("---")
        
        # Source selection
        st.markdown("#### 📹 Input Source")
        source_type = st.radio("Select Input", ["📷 Webcam", "🎬 Video File"], label_visibility="collapsed")
        
        if source_type == "🎬 Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'], label_visibility="collapsed")
            st.session_state.video_source = uploaded_file
        else:
            st.session_state.video_source = "webcam"
        
        st.markdown("---")
        
        # Controls
        st.markdown("#### 🎮 Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", type="primary", disabled=st.session_state.detector is None, use_container_width=True):
                st.session_state.processing = True
                st.session_state.frame_count = 0
        
        with col2:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                st.session_state.processing = False
        
        if st.button("🔄 Reset Stats", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.reset_statistics()
                st.session_state.stats = st.session_state.detector.get_statistics()
                st.session_state.model_loaded_time = time.time()
                st.session_state.frame_count = 0
                st.success("✅ Reset")
    
    # Main content - Video feed
    st.markdown('<div class="section-header">📹 Live Detection Feed</div>', unsafe_allow_html=True)
    
    if st.session_state.processing:
        if st.session_state.video_source == "webcam":
            process_webcam_frame()
        elif st.session_state.video_source:
            process_video_file(st.session_state.video_source)
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<div class="info-title">👋 Ready to Start</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-details">Load model and click "Start" to begin</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics at the bottom
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Statistics</div>', unsafe_allow_html=True)
    display_statistics()
    
    # System status
    st.markdown("---")
    st.markdown('<div class="section-header">🔧 System Status</div>', unsafe_allow_html=True)
    display_system_status()

if __name__ == "__main__":
    main()
