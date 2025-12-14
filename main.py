import cv2
import numpy as np
import mediapipe as mp
import time

from pose_recorder import PoseRecorder
from display_utils import DisplayManager,CombinedVisualizer
from fps_counter import FPSCounter
from bicep_curl_detector import BicepCurlDetector
from shoulder_abduction_detector import ShoulderAbductionDetector
from reaction_time_detector import ReactionTimeDetector
from hand_tracker import HandTracker
from pose_3d_visualizer import create_3d_pose_map
from hand_gesture_detector import HandGestureDetector

 # Try to use C++ optimized version, fallback to Python if unavailable
# try:
#     from angle_calculator_cpp import calculate_all_angles
#     print("✓ Using C++ optimized angle calculator")
# except ImportError:
from angle_calculator import calculate_all_angles
print("✓ Using Python angle calculator")


# Initialize MediaPipe
mp_pose = mp.solutions.pose


def main():
    """Main function with POSE + HAND tracking integration"""
    # Configuration
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PROCESS_EVERY_N_FRAMES = 1  
    PROCESS_HANDS_EVERY_N_FRAMES = 1  # Process hands less frequently for performance

    # Set up video capture
    # Set up video capture (prefer Intel RealSense if available)
    USE_REALSENSE = False
    try:
        from realsense_adapter import RealSenseCapture
        cap = RealSenseCapture(FRAME_WIDTH, FRAME_HEIGHT)
        USE_REALSENSE = True
        print("✓ Using Intel RealSense camera")
    except Exception:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open video source")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Initialize components
    pose_recorder = PoseRecorder()
    display_manager = DisplayManager(FRAME_WIDTH, FRAME_HEIGHT)
    fps_counter = FPSCounter()

    # Depth overlay toggle (works when using RealSenseCapture)
    depth_overlay_enabled = False

    def depth_colormap(depth_mm, max_depth_mm=4000):
        if depth_mm is None:
            return None
        depth_clipped = np.clip(depth_mm, 0, max_depth_mm)
        depth_8u = (depth_clipped.astype(np.float32) / max_depth_mm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        return depth_color

    # Initialize hand tracking components
    hand_tracker = HandTracker(max_num_hands=2)
    gesture_detector = HandGestureDetector()
    combined_viz = CombinedVisualizer()
    hand_tracking_enabled = True  # Start with hand tracking ON

    # Initialize Reaction Time Detector
    reaction_detector = ReactionTimeDetector()
    reaction_detector._last_printed_result = None

    # Initialize exercise detectors
    right_bicep_detector = BicepCurlDetector(arm='right')
    left_bicep_detector = BicepCurlDetector(arm='left')
    right_abduction_detector = ShoulderAbductionDetector(arm='right')
    left_abduction_detector = ShoulderAbductionDetector(arm='left')

    # Exercise selection variables
    current_exercise = 'bicep'
    current_detector = right_bicep_detector
    exercise_detection_enabled = True

    # Zoom and move variables for exercise tracker
    exercise_zoom = 1.0
    exercise_offset = (0, 0)
    
    # Touch gesture drag state
    touch_dragging = False
    touch_start_hand_pos = None
    touch_start_offset = None

    # Frame skipping variables
    frame_counter = 0
    hand_frame_counter = 0
    saved_filename = None

    # Cache for skipped frames (POSE)
    cached_pose_results = None
    cached_pose_landmarks = None
    cached_angles_dict = None
    cached_exercise_results = None

    # Cache for skipped frames (HANDS)
    cached_hand_results = None
    cached_gesture_result = {'gesture': 'No Hand', 'confidence': 0.0}

    # Gesture control cooldown (prevent rapid triggering)
    last_gesture_action_time = 0
    gesture_cooldown = 2.0  # seconds

    # Setup MediaPipe instance
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0,
    ) as pose:

        print("\n=== CONTROLS ===")
        print("H - Toggle Hand Tracking ON/OFF")
        print("G - Enable/Disable Gesture Controls")
        print("R - Start Recording | S - Stop Recording")
        print("E - Toggle Exercise Detection")
        print("A - Switch Arm | W - Switch Exercise")
        print("X - Reset Counter | T - Show Stats")
        print("P - Reset Exercise Tracker Position")
        print("Q - Quit")
        print("================\n")

        gesture_controls_enabled = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame to correct handedness (since camera is mirrored)
            # frame = cv2.flip(frame, 1)

            # Frame skipping logic
            frame_counter += 1
            hand_frame_counter += 1
            process_pose_frame = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)
            process_hand_frame = (hand_frame_counter % PROCESS_HANDS_EVERY_N_FRAMES == 0) and hand_tracking_enabled

            # Start timing
            frame_start_time = time.time()

            # === POSE PROCESSING ===
            if process_pose_frame:
                
                # Recolor image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Make pose detection
                pose_results = pose.process(image_rgb)

                # Initialize angles dictionary
                angles_dict = {
                    "left_elbow": None, "right_elbow": None,
                    "left_shoulder": None, "right_shoulder": None,
                    "left_hip": None, "right_hip": None,
                    "left_knee": None, "right_knee": None
                }

                exercise_results = None

                if pose_results and pose_results.pose_landmarks:
                    pose_landmarks = pose_results.pose_landmarks.landmark
                    # Provide depth info to angle calculator when available
                    depth_info = None
                    try:
                        if hasattr(cap, 'get_depth_info'):
                            depth_info = cap.get_depth_info()
                    except Exception:
                        depth_info = None
                    # Try depth-aware calculation, fall back if the angle calculator doesn't accept depth
                    try:
                        angles_dict = calculate_all_angles(pose_landmarks, depth_info)
                    except TypeError:
                        angles_dict = calculate_all_angles(pose_landmarks)

                    if exercise_detection_enabled:
                        exercise_results = current_detector.update(angles_dict)

                    if pose_recorder.recording:
                        pose_recorder.add_frame(pose_results.pose_landmarks, angles_dict)
                else:
                    pose_landmarks = None
                    if exercise_detection_enabled:
                        exercise_results = current_detector.update(angles_dict)

                # Check for visualization key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('3') and pose_results and pose_results.pose_landmarks:
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
                        pose_results.pose_landmarks.landmark,
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

                # Cache pose results
                cached_pose_results = pose_results
                cached_pose_landmarks = pose_landmarks
                cached_angles_dict = angles_dict.copy()
                cached_exercise_results = exercise_results

                current_fps = fps_counter.update('processing')
            else:
                # Use cached pose data
                pose_results = cached_pose_results
                pose_landmarks = cached_pose_landmarks
                angles_dict = cached_angles_dict if cached_angles_dict else {
                    "left_elbow": None, "right_elbow": None,
                    "left_shoulder": None, "right_shoulder": None,
                    "left_hip": None, "right_hip": None,
                    "left_knee": None, "right_knee": None
                }
                exercise_results = cached_exercise_results
                current_fps = fps_counter.update('skipped')

            # === HAND PROCESSING (if enabled) ===
            if process_hand_frame:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Process hands
                hand_results = hand_tracker.process(image_rgb)

                # Detect gesture from right hand (primary control hand)
                gesture_result = {'gesture': 'No Hand', 'confidence': 0.0}
                if hand_tracker.has_right_hand():
                    gesture_result = gesture_detector.detect_gesture(
                        hand_tracker.get_all_landmarks('right')
                    )

                # Handle gesture controls for exercise tracker
                if gesture_result['gesture'] == 'Pinch':
                    exercise_zoom = max(0.5, exercise_zoom * 0.95)
                elif gesture_result['gesture'] == 'Spread':
                    exercise_zoom = min(2.0, exercise_zoom * 1.05)
                elif gesture_result['gesture'] == 'Touch':
                    if hand_tracker.has_right_hand():
                        hand_landmarks = hand_tracker.get_all_landmarks('right')
                        if hand_landmarks:
                            # Get hand center position for moving the exercise tracker
                            wrist = hand_landmarks.landmark[0]
                            index_tip = hand_landmarks.landmark[8]
                            
                            # Use average of wrist and index tip as hand center
                            hand_center_x = (wrist.x + index_tip.x) / 2
                            hand_center_y = (wrist.y + index_tip.y) / 2
                            
                            # Convert to pixel coordinates
                            hand_pixel_x = int(hand_center_x * FRAME_WIDTH)
                            hand_pixel_y = int(hand_center_y * FRAME_HEIGHT)
                            current_hand_pos = (hand_pixel_x, hand_pixel_y)
                            
                            if not touch_dragging:
                                # Start dragging
                                touch_dragging = True
                                touch_start_hand_pos = current_hand_pos
                                touch_start_offset = exercise_offset
                            else:
                                # Continue dragging - calculate relative movement
                                if touch_start_hand_pos is not None:
                                    delta_x = current_hand_pos[0] - touch_start_hand_pos[0]
                                    delta_y = current_hand_pos[1] - touch_start_hand_pos[1]
                                    
                                    # Apply movement with reduced sensitivity
                                    move_factor = 0.8
                                    new_offset_x = touch_start_offset[0] + int(delta_x * move_factor)
                                    new_offset_y = touch_start_offset[1] + int(delta_y * move_factor)
                                    
                                    # Clamp offset to keep panel on screen
                                    max_offset_x = FRAME_WIDTH // 2 - 50  # Keep some margin
                                    max_offset_y = FRAME_HEIGHT // 2 - 50
                                    exercise_offset = (
                                        max(-max_offset_x, min(max_offset_x, new_offset_x)),
                                        max(-max_offset_y, min(max_offset_y, new_offset_y))
                                    )
                else:
                    # Reset drag state when Touch gesture ends
                    touch_dragging = False
                    touch_start_hand_pos = None
                    touch_start_offset = None

                # Cache hand results
                cached_hand_results = hand_results
                cached_gesture_result = gesture_result
            elif hand_tracking_enabled:
                # Use cached hand data
                hand_results = cached_hand_results
                gesture_result = cached_gesture_result
            else:
                # No hand tracking
                hand_results = None
                gesture_result = {'gesture': 'No Hand', 'confidence': 0.0}

            # === VISUALIZATION ===
            image = frame.copy()

            # Apply depth overlay if enabled and available
            try:
                has_depth = hasattr(cap, 'get_last_depth')
            except Exception:
                has_depth = False

            if depth_overlay_enabled and has_depth:
                depth_mm = cap.get_last_depth()
                depth_vis = depth_colormap(depth_mm, max_depth_mm=4000)
                if depth_vis is not None:
                    if depth_vis.shape[:2] != image.shape[:2]:
                        depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))
                    image = cv2.addWeighted(image, 0.6, depth_vis, 0.4, 0)

            if hand_tracking_enabled:
                # Use combined visualizer for both pose and hands
                combined_viz.draw_both(
                    image,
                    pose_results.pose_landmarks if pose_results else None,
                    hand_results
                )
                
                if pose_results and pose_results.pose_landmarks:
                    display_manager.draw_all_angles(image, pose_landmarks, angles_dict)
                else:
                    display_manager.draw_all_angles(image, None, angles_dict)
    
               
                # Draw hand info
                combined_viz.draw_hand_info(image, hand_tracker, y_offset=30)
                
            else:
                # Draw only pose (original behavior)
                if pose_results and pose_results.pose_landmarks:
                    combined_viz.draw_pose(image, pose_results.pose_landmarks)
                    display_manager.draw_all_angles(image, pose_landmarks, angles_dict)
                else:
                    display_manager.draw_all_angles(image, None, angles_dict)

            # Draw exercise information
            if exercise_detection_enabled and exercise_results:
                display_manager.draw_exercise_info(image, exercise_results, exercise_zoom, exercise_offset)

            # Reaction Time Integration
            reaction_result = reaction_detector.update(angles_dict)
            if reaction_result['reaction_time_ms'] is not None:
                if (reaction_result['reaction_time_ms'] != reaction_detector._last_printed_result and
                        reaction_result['angle_change'] is not None):
                    print(f"\n⚡ REACTION TIME: {reaction_result['reaction_time_ms']}ms "
                          f"(angle change: {reaction_result['angle_change']}°)")
                    reaction_detector._last_printed_result = reaction_result['reaction_time_ms']

            # Draw status information
            display_manager.draw_status_info(
                image,
                current_fps,
                pose_recorder.recording,
                saved_filename,
                exercise_detection_enabled,
                current_detector.arm,
                current_exercise
            )

            # If depth available and pose landmarks exist, draw per-landmark depth
            try:
                if has_depth and pose_results and pose_results.pose_landmarks:
                    for lm_idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                        try:
                            px = int(lm.x * FRAME_WIDTH)
                            py = int(lm.y * FRAME_HEIGHT)
                        except Exception:
                            continue

                        # clamp coordinates to image
                        px = max(0, min(px, FRAME_WIDTH - 1))
                        py = max(0, min(py, FRAME_HEIGHT - 1))

                        depth_m = None
                        if hasattr(cap, 'get_depth_at'):
                            try:
                                depth_m = cap.get_depth_at(px, py)
                            except Exception:
                                depth_m = None

                        if depth_m is not None:
                            # draw a small circle and the depth (meters)
                            cv2.circle(image, (px, py), 3, (0, 255, 0), -1)
                            text = f"{depth_m:.2f}m"
                            # offset text slightly to avoid overlapping the marker
                            tx = px + 5
                            ty = py - 5
                            # ensure text is on-screen
                            tx = max(0, min(tx, FRAME_WIDTH - 1))
                            ty = max(10, min(ty, FRAME_HEIGHT - 1))
                            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            except Exception:
                pass

            # Display the frame
            cv2.imshow('MediaPipe Pose + Hand Tracking', image)

            # === KEYBOARD CONTROLS ===
            key = cv2.waitKey(10) & 0xFF

            if key == ord('h'):
                hand_tracking_enabled = not hand_tracking_enabled
                print(f"Hand Tracking: {'ON' if hand_tracking_enabled else 'OFF'}")

            elif key == ord('d'):
                depth_overlay_enabled = not depth_overlay_enabled
                print(f"Depth overlay: {'ON' if depth_overlay_enabled else 'OFF'}")

            elif key == ord('r'):
                pose_recorder.start_recording() 
                saved_filename = None

            elif key == ord('s'):
                if pose_recorder.recording:
                    saved_filename = pose_recorder.stop_recording()

            elif key == ord('e'):
                exercise_detection_enabled = not exercise_detection_enabled
                print(f"Exercise detection: {'ON' if exercise_detection_enabled else 'OFF'}")

            elif key == ord('p'):
                # Reset exercise tracker position
                exercise_offset = (0, 0)
                exercise_zoom = 1.0
                print("Exercise tracker position and zoom reset")

            elif key == ord('a'):
                if current_exercise == 'bicep':
                    if current_detector == right_bicep_detector:
                        current_detector = left_bicep_detector
                        print("Switched to LEFT arm bicep detection")
                    else:
                        current_detector = right_bicep_detector
                        print("Switched to RIGHT arm bicep detection")
                else:
                    if current_detector == right_abduction_detector:
                        current_detector = left_abduction_detector
                        print("Switched to LEFT arm abduction detection")
                    else:
                        current_detector = right_abduction_detector
                        print("Switched to RIGHT arm abduction detection")

            elif key == ord('w'):
                if current_exercise == 'bicep':
                    current_exercise = 'abduction'
                    if current_detector.arm == 'right':
                        current_detector = right_abduction_detector
                    else:
                        current_detector = left_abduction_detector
                    print(f"Switched to SHOULDER ABDUCTION exercise ({current_detector.arm.upper()} arm)")
                else:
                    current_exercise = 'bicep'
                    if current_detector.arm == 'right':
                        current_detector = right_bicep_detector
                    else:
                        current_detector = left_bicep_detector
                    print(f"Switched to BICEP CURL exercise ({current_detector.arm.upper()} arm)")

            elif key == ord('x'):
                current_detector.reset()
                exercise_name = "Bicep Curl" if current_exercise == 'bicep' else "Elbow Abduction"
                print(f"Reset {current_detector.arm} arm {exercise_name} counter")

            elif key == ord('t'):
                stats = current_detector.get_stats()
                exercise_name = "BICEP CURL" if current_exercise == 'bicep' else "ELBOW ABDUCTION"
                print(f"\n== {current_detector.arm.upper()} ARM {exercise_name} STATS ===")
                print(f"Total Reps: {stats['total_reps']}")
                print(f"Average Duration: {stats['avg_duration']}s")
                print(f"Average Range of Motion: {stats['avg_range_of_motion']}°")
                if stats['total_reps'] > 0:
                    print(f"Last Rep Duration: {stats['last_rep_duration']}s")
                print("=" * 50)

            elif key == ord('z'):
                fps_counter.print_detailed_stats()

            elif key == ord('f'):
                fps_filename = fps_counter.save_fps_data()
                if fps_filename:
                    print(f"FPS data saved: {fps_filename}")

            elif key == ord('y'):
                reaction_detector.start_test()

            elif key == ord('v'):
                reaction_detector.print_stats()
                reaction_detector.save_stats_json()

            elif key == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    reaction_detector.close()


if __name__ == "__main__":
    main()