# 3D Pose Visualization

This project now includes 3D pose mapping capabilities that visualize landmarks in 3D space with calculated distances.

## New Files

### `pose_3d_visualizer.py`
Core module for 3D pose visualization containing:
- `extract_3d_points()`: Converts MediaPipe landmarks to 3D coordinates using depth information
- `calculate_key_distances()`: Computes distances between key body landmarks
- `visualize_3d_pose()`: Creates 3D matplotlib plot of the pose skeleton
- `create_3d_pose_map()`: Main function that combines extraction, distance calculation, and visualization

### `demo_3d_visualization.py`
Demo script that shows how to use the 3D visualizer with live camera feed.

## Features

- **3D Landmark Extraction**: Converts 2D MediaPipe landmarks to 3D coordinates using depth data from Intel RealSense
- **Depth Reliability**: Uses 3x3 pixel window analysis to ensure depth measurements are reliable
- **Skeleton Visualization**: Draws connected skeleton in 3D space
- **Distance Calculations**: Computes key body measurements:
  - Shoulder width
  - Left/right arm lengths
  - Left/right leg lengths
  - Torso height
- **Fallback Support**: Works with 2D data (sets Z=0) when depth is unavailable

## Usage

### Basic Usage
```python
from pose_3d_visualizer import create_3d_pose_map

# With depth information (RealSense)
points_3d, distances = create_3d_pose_map(landmarks, depth_info, img_width, img_height)

# Without depth (2D only)
points_3d, distances = create_3d_pose_map(landmarks)
```

### Running the Demo
```bash
python demo_3d_visualization.py
```

Controls:
- Press 'v' to generate 3D visualization of current pose
- Press 'q' to quit

### Integration with Main App
You can integrate 3D visualization into your main pose detection app:

```python
from pose_3d_visualizer import create_3d_pose_map

# In your pose processing loop
if results.pose_landmarks:
    depth_info = cap.get_depth_info() if hasattr(cap, 'get_depth_info') else None
    points_3d, distances = create_3d_pose_map(
        results.pose_landmarks.landmark,
        depth_info,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        show_plot=False  # Set to True to show plot
    )

    # Use points_3d and distances for further analysis
    print(f"Shoulder width: {distances.get('shoulder_width', 'N/A')}")
```

## Requirements

- matplotlib (for 3D plotting)
- mediapipe
- numpy
- opencv-python
- pyrealsense2 (optional, for depth-enhanced visualization)

## Output

The 3D visualization shows:
- Red dots for landmark positions
- Blue lines connecting skeleton joints
- Text labels for each landmark
- Distance measurements displayed in a text box
- Coordinate system in meters (X=right, Y=up, Z=forward from camera)

When depth data is available, the visualization represents actual 3D positions. Without depth, landmarks are shown at Z=0 with realistic X,Y positions based on image coordinates.