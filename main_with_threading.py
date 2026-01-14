"""
THREADING PATCH WITH PROPER FPS/TIMING TRACKING
Tracks both main loop FPS and background MediaPipe processing time
"""

import cv2
import mediapipe as mp
import time
import threading
from queue import Queue, Empty

from pose_recorder import PoseRecorder
from display_utils import DisplayManager
from fps_threading import FPSCounter
from bicep_curl_detector import BicepCurlDetector
from shoulder_abduction_detector import ShoulderAbductionDetector
from reaction_time_detector import ReactionTimeDetector

try:
    from angle_calculator_cpp import calculate_all_angles
    print(" Using C++ optimized angle calculator")
except ImportError:
    from angle_calculator import calculate_all_angles
    print(" Using Python angle calculator")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class PoseProcessorThread:
    """Separate thread for MediaPipe processing with timing metrics"""
    
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        
        # Timing metrics
        self.processing_times = []
        self.max_processing_times = 1000  # Keep last 1000 measurements
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def put_frame(self, frame):
        """Send frame for processing (non-blocking).

        Can pass either `frame` or `(frame, depth_info)` where `depth_info` is a dict
        compatible with `angle_calculator.calculate_all_angles`.
        """
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
            return True
        return False
    
    def get_result(self):
        """Get processed result (non-blocking, returns None if empty)"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_processing_stats(self):
        """Get statistics about background processing times"""
        if not self.processing_times:
            return None
        
        times = self.processing_times[-100:]  # Last 100 frames
        return {
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'latest_ms': times[-1] if times else None,
            'count': len(self.processing_times)
        }
            
    def _process_loop(self):
        """Background processing loop with timing"""
        with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0,
        ) as pose:
            
            while self.running:
                try:
                    item = self.frame_queue.get(timeout=0.1)
                    # item can be either a frame or (frame, depth_info)
                    if isinstance(item, tuple) and len(item) == 2:
                        frame, depth_info = item
                    else:
                        frame = item
                        depth_info = None
                    
                    # ===== START TIMING =====
                    process_start = time.time()
                    
                    # Process frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    
                    # Calculate angles
                    angles_dict = {
                        "left_elbow": None, "right_elbow": None,
                        "left_shoulder": None, "right_shoulder": None,
                        "left_hip": None, "right_hip": None,
                        "left_knee": None, "right_knee": None
                    }
                    
                    landmarks = None
                    if results and results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        try:
                            angles_dict = calculate_all_angles(landmarks, depth_info)
                        except Exception:
                            angles_dict = calculate_all_angles(landmarks)
                    
                    # ===== END TIMING =====
                    process_time_ms = (time.time() - process_start) * 1000
                    
                    # Store timing
                    self.processing_times.append(process_time_ms)
                    if len(self.processing_times) > self.max_processing_times:
                        self.processing_times.pop(0)
                    
                    # Put result back with timing info
                    result_data = {
                        'results': results,
                        'landmarks': landmarks,
                        'angles_dict': angles_dict,
                        'process_time_ms': process_time_ms,
                        'timestamp': time.time()
                    }
                    
                    # Drop old results if queue full
                    if self.result_queue.full():
                        try:
                            dropped = self.result_queue.get_nowait()
                            # Could log dropped frames here if needed
                        except Empty:
                            pass
                    
                    self.result_queue.put(result_data)
                    
                except Empty:
                    continue
                except Exception as e:
                    print(f"⚠ Processing error: {e}")


def main():
    """Main function with proper FPS tracking"""
    # Configuration
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PROCESS_EVERY_N_FRAMES = 2

    # Set up video capture (prefer Intel RealSense if available)
    try:
        from realsense_adapter import RealSenseCapture
        cap = RealSenseCapture(FRAME_WIDTH, FRAME_HEIGHT)
        print(" Using Intel RealSense camera")
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

    # Frame skipping variables
    frame_counter = 0
    saved_filename = None

    # Cache for results
    cached_results = None
    cached_landmarks = None
    cached_angles_dict = None
    cached_exercise_results = None

    # Start background processing thread
    processor = PoseProcessorThread()
    processor.start()
    print("✓ Threading enabled - MediaPipe running in background")
    
    # Tracking variables for debugging
    last_stats_print = time.time()
    stats_print_interval = 5.0  # Print stats every 5 seconds
    
    try:
        while cap.isOpened():
            loop_start = time.time()  # Time the entire loop
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Frame skipping logic
            frame_counter += 1
            process_current_frame = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)

            if process_current_frame:
                # Send frame to background thread
                try:
                    depth_info = None
                    if hasattr(cap, 'get_depth_info'):
                        depth_info = cap.get_depth_info()
                    frame_sent = processor.put_frame((frame.copy(), depth_info))
                except Exception:
                    frame_sent = processor.put_frame(frame.copy())
                frame_type = 'processing' if frame_sent else 'dropped'
            else:
                frame_type = 'skipped'

            # Try to get results from background thread
            result_data = processor.get_result()
            if result_data:
                # Got new results from thread
                cached_results = result_data['results']
                cached_landmarks = result_data['landmarks']
                cached_angles_dict = result_data['angles_dict']
                
                # Log background processing time to FPS counter
                bg_process_time = result_data.get('process_time_ms')
                if bg_process_time:
                    # Store as a separate metric
                    fps_counter.log_data_point(
                        frame_type='background_processing',
                        custom_time_ms=bg_process_time
                    )
                
                # Exercise detection
                if exercise_detection_enabled:
                    cached_exercise_results = current_detector.update(cached_angles_dict)
                
                # Add frame to recording if active
                if pose_recorder.recording and cached_results and cached_results.pose_landmarks:
                    pose_recorder.add_frame(cached_results.pose_landmarks, cached_angles_dict)
            
            # Use cached data for display
            results = cached_results
            landmarks = cached_landmarks
            angles_dict = cached_angles_dict if cached_angles_dict else {
                "left_elbow": None, "right_elbow": None,
                "left_shoulder": None, "right_shoulder": None,
                "left_hip": None, "right_hip": None,
                "left_knee": None, "right_knee": None
            }
            exercise_results = cached_exercise_results

            # Display logic
            image = frame.copy()

            if results and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                display_manager.draw_all_angles(image, landmarks, angles_dict)
            else:
                display_manager.draw_all_angles(image, None, angles_dict)

            if exercise_detection_enabled and exercise_results:
                display_manager.draw_exercise_info(image, exercise_results)

            # Reaction Time Detector Integration
            reaction_result = reaction_detector.update(angles_dict)

            if reaction_result['reaction_time_ms'] is not None:
                if (reaction_result['reaction_time_ms'] != reaction_detector._last_printed_result and
                        reaction_result['angle_change'] is not None):
                    print(f"\n⚡ REACTION TIME: {reaction_result['reaction_time_ms']}ms "
                          f"(angle change: {reaction_result['angle_change']}°)")
                    reaction_detector._last_printed_result = reaction_result['reaction_time_ms']

            # Update FPS counter for main loop timing
            current_fps = fps_counter.update(frame_type)

            display_manager.draw_status_info(
                image,
                current_fps,
                pose_recorder.recording,
                saved_filename,
                exercise_detection_enabled,
                current_detector.arm,
                current_exercise
            )

            # Print periodic stats for debugging
            if time.time() - last_stats_print >= stats_print_interval:
                bg_stats = processor.get_processing_stats()
                if bg_stats:
                    print(f"\n SYSTEM STATS (last 5s):")
                    print(f"   Main Loop FPS: {current_fps:.1f}")
                    print(f"   Background Processing: {bg_stats['avg_ms']:.1f}ms avg "
                          f"(min: {bg_stats['min_ms']:.1f}ms, max: {bg_stats['max_ms']:.1f}ms)")
                    print(f"   Total processed frames: {bg_stats['count']}")
                last_stats_print = time.time()

            cv2.imshow('MediaPipe Pose with Exercise Detection', image)

            # Keyboard handling
            key = cv2.waitKey(10) & 0xFF

            if key == ord('r'):
                pose_recorder.start_recording()
                saved_filename = None

            elif key == ord('s'):
                if pose_recorder.recording:
                    saved_filename = pose_recorder.stop_recording()

            elif key == ord('e'):
                exercise_detection_enabled = not exercise_detection_enabled
                print(f"Exercise detection: {'ON' if exercise_detection_enabled else 'OFF'}")

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
                print(f"Average Range of Motion: {stats['avg_range_of_motion']}")
                if stats['total_reps'] > 0:
                    print(f"Last Rep Duration: {stats['last_rep_duration']}s")
                print("=" * 50)

            elif key == ord('p'):
                fps_counter.print_detailed_stats()
                bg_stats = processor.get_processing_stats()
                if bg_stats:
                    print(f"\n BACKGROUND PROCESSING STATS:")
                    print(f"   Average: {bg_stats['avg_ms']:.2f}ms")
                    print(f"   Min: {bg_stats['min_ms']:.2f}ms")
                    print(f"   Max: {bg_stats['max_ms']:.2f}ms")
                    print(f"   Total frames processed: {bg_stats['count']}")

            elif key == ord('f'):
                fps_filename = fps_counter.save_fps_data()
                if fps_filename:
                    print(f"✓ FPS data saved: {fps_filename}")

            elif key == ord('y'):
                reaction_detector.start_test()

            elif key == ord('v'):
                reaction_detector.print_stats()
                reaction_detector.save_stats_json()

            elif key == ord('q'):
                break

    finally:
        print("\n Stopping threads...")
        processor.stop()
        
        # Print final stats
        bg_stats = processor.get_processing_stats()
        if bg_stats:
            print(f"\n FINAL BACKGROUND PROCESSING STATS:")
            print(f"   Average: {bg_stats['avg_ms']:.2f}ms ({1000/bg_stats['avg_ms']:.1f} FPS)")
            print(f"   Min: {bg_stats['min_ms']:.2f}ms")
            print(f"   Max: {bg_stats['max_ms']:.2f}ms")
            print(f"   Total frames processed: {bg_stats['count']}")
        
        cap.release()
        cv2.destroyAllWindows()
        reaction_detector.close()


if __name__ == "__main__":
    main()
