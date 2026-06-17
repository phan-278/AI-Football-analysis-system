# 📖 1. Description
**AI Football Analysis System** is an intelligent video processing and tracking system designed for football matches. It tracks players, referees, and the ball, maps their positions onto a 2D tactical pitch using homography, estimates speed and distance covered, calculates ball possession, and generates automated tactical reports with statistics.

# 🏷️ 2. Badges/Tags
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# 🚀 3. Intro
Welcome to the AI Football Analysis System project! This project aims to provide comprehensive video analysis for football matches, allowing coaches and analysts to track player movements, monitor team formations, calculate speed and distance, and generate tactical reports and heatmaps from regular broadcast footage.

# 📂 4. Project Structure
```text
AI-Football-analysis-system/
├── 📁 camera_movement_estimator/ # Adjusts object positions based on camera motion
├── 📁 input_videos/           # Input match videos for processing
├── 📁 models/                 # Pre-trained AI models (YOLO, ReID)
├── 📁 nlp_report/             # Generates tactical PDF reports using LLMs
├── 📁 output_videos/          # Output videos with tracking and tactical map overlays
├── 📁 outputs/                # Exported stats JSON and tactical PDF reports
├── 📁 reid/                   # OSNet-based Person Re-Identification to handle lost tracks
├── 📁 speed_estimator/        # Calculates player speed and distance covered
├── 📁 stats_aggregator/       # Aggregates player/team stats, generates heatmaps
├── 📁 stubs/                  # Cached data (tracking, homography) to speed up dev
├── 📁 tatical_map/            # Overlays the 2D tactical map PiP on the video
├── 📁 team_assign/            # Clusters player colors, assigns teams & ball possession
├── 📁 trackers/               # Object tracking using YOLO and ByteTrack
├── 📁 utils/                  # Helper math and video functions
├── 📁 view_tranformer/        # Estimates homography and transforms pixel coordinates to real-world meters
├── 📄 main.py                 # Main execution script
├── 📄 main_report.py          # Main script to run only the NLP report generation
└── 📄 README.md               # Project documentation
```

# 💻 5. Technologies
- **Programming Language:** 🐍 Python
- **Computer Vision:** 👁️ OpenCV
- **Object Detection & Tracking:** 🎯 YOLO, ByteTrack
- **Re-Identification (ReID):** 👤 OSNet
- **Math & Matrix Operations:** 🧮 NumPy
- **Reporting:** 📝 ReportLab, LLM APIs (Groq)

# ✨ 6. Features
- **Object Tracking & ReID:** Robustly tracks players, referees, and the ball across frames, using ReID to handle occlusions and ID switches.
- **Homography Transformation:** Converts perspective video coordinates into a top-down static 2D pitch map (Eagle View).
- **Camera Movement Estimation:** Adjusts for camera panning and zooming to ensure accurate real-world positioning.
- **Speed & Distance Estimation:** Calculates the running speed and total distance covered by each player in real-world units (km/h, meters).
- **Ball Possession & Team Assignment:** Automatically clusters player colors to define teams and determines ball possession.
- **Tactical Map & Heatmaps:** Generates a Picture-in-Picture (PiP) 2D tactical map and spatial heatmaps.
- **Automated NLP Reporting:** Exports match statistics to JSON and generates an automated PDF tactical report using LLMs.

# ⌨️ 7. Keyboard Shortcuts
- *This project primarily runs as an offline processing pipeline via `main.py` without real-time keyboard interaction.*

# ⚙️ 8. The Process
Building this system involved tackling several computer vision and data processing challenges. A significant hurdle was handling dynamic camera movements (panning and zooming) in broadcast footage, which required implementing a robust camera movement estimator to adjust pixel positions accurately. Furthermore, mapping 2D pixel coordinates to real-world meters involved applying homography transformations based on pitch keypoints. Player tracking was enhanced by combining ByteTrack with OSNet Re-Identification to solve ID switching during occlusions or fast movements.

# 🧠 9. What I Learned
Through this project, I gained hands-on experience with:
- 📊 **Model Fine-Tuning & Tracking:** Fine-tuning object detection models and integrating ByteTrack for robust multi-object tracking.
- 🔗 **Re-Identification (ReID):** Adding and utilizing ReID mechanisms to maintain consistent player identities across frames.
- 📐 **Camera Movement Estimation:** Extracting corner features and using optical flow to measure and compensate for camera movement.
- 🗺️ **Dynamic Homography & Mapping:** Transforming pixel coordinates to real-world coordinates to display on a 2D tactical map, and using dynamic homography adjustments to measure player speed accurately.
- 🎨 **Machine Learning for Team Assignment:** Utilizing K-Means clustering to extract color features from player jerseys and assign teams.
- 🤖 **Automated Tactical Reporting:** Calling Large Language Model (LLM) APIs to generate automated tactical match reports based on the aggregated data.

# 🚀 10. How to Improve
In the future, I plan to expand the system by adding:
- **Action Recognition:** Detecting specific player actions like passing, shooting, or tackling.
- **Offside Line Detection:** Automatically drawing the offside line based on the last defender's position.
- **Real-time Processing:** Optimizing the pipeline to process live camera feeds from stadiums.
- **Advanced ReID Mechanisms:** Integrating SOLIDER models and replacing the current ReID mechanism with StrongSORT for more robust multi-object tracking.
- **Pitch Keypoint Fine-Tuning:** Fine-tuning the pitch keypoint detection model to better fit real-world keypoints across various football stadiums.

# 🛠️ 11. How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/phan-278/AI-Football-analysis-system
   cd AI-Football-analysis-system
   ```
2. **Download Models & Assets:**
   Ensure you place the YOLO models (`best_yolo11s.pt`, `best1_pitch.pt`) and ReID weights into the `models/` folder, and input videos into the `input_videos/` folder.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the pipeline:**
   Execute the main script to process the video and generate outputs.
   ```bash
   python main.py
   ```

# 🎥 12. Video Demo
📹 The output videos with tactical overlays will be saved in the `output_videos/` directory.