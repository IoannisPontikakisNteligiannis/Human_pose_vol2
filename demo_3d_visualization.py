#!/usr/bin/env python3
"""
Demo script for 3D pose visualization using the pose_3d_visualizer module.
This script demonstrates how to use the 3D pose mapping with distances.
"""

import cv2
import mediapipe as mp
from pose_3d_visualizer import create_3d_pose_map

# Try to import RealSense adapter
try:
    from realsense_adapter import RealSenseCapture
    USE_REALSENSE = True
    print("Using Intel RealSense camera for depth-enhanced 3D visualization")
except ImportError:
    USE_REALSENSE = False
    print("RealSense not available, using webcam with 2D visualization (Z=0)")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    # Initialize camera
    if USE_REALSENSE:
        FRAME_WIDTH, FRAME_HEIGHT = 640, 480
        cap = RealSenseCapture(FRAME_WIDTH, FRAME_HEIGHT)
    else:
        cap = cv2.VideoCapture(0)
        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'v' to visualize current pose in 3D")
    print("Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose
        results = pose.process(rgb_frame)

        # Display instructions
        cv2.putText(frame, "Press 'v' for 3D visualization", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.pose_landmarks:
            # Draw pose landmarks on frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Pose detected - Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Check for visualization key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                print("Creating 3D pose visualization...")

                # Get depth info if available
                depth_info = None
                if USE_REALSENSE and hasattr(cap, 'get_depth_info'):
                    try:
                        depth_info = cap.get_depth_info()
                    except:
                        depth_info = None

                # Create 3D visualization
                points_3d, distances = create_3d_pose_map(
                    results.pose_landmarks.landmark,
                    depth_info,
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                    show_plot=True
                )

                # Print distances to console
                print("\nCalculated distances:")
                for key, value in distances.items():
                    if value is not None:
                        print(f"  {key}: {value:.3f} meters")
                    else:
                        print(f"  {key}: Not available")

                print("\n3D points (first 5 landmarks):")
                for i in range(min(5, len(points_3d))):
                    point = points_3d.get(i)
                    if point:
                        print(f"  Landmark {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] meters")
                    else:
                        print(f"  Landmark {i}: Not available")

        else:
            cv2.putText(frame, "No pose detected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show frame
        cv2.imshow("Pose Detection - Press 'v' for 3D view", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()