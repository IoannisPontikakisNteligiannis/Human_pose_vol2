import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import drawing_styles
mp_pose = mp.solutions.pose

"""
Combined Visualizer - Draws both pose and hand landmarks
Provides clean visualization of the complete body + hands tracking
"""

class CombinedVisualizer:
    """Handles visualization of both pose and hand landmarks"""
    
    def __init__(self):
        # MediaPipe drawing utilities
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for better visibility
        self.pose_landmark_style = self.mp_drawing_styles.get_default_pose_landmarks_style()
            
        
        self.pose_connection_style = self.mp_drawing.DrawingSpec(
            thickness=2,
            circle_radius=2,
            color=(255, 255, 255)  
        )
       
        
        self.hand_connection_style = self.mp_drawing_styles.DrawingSpec(
            thickness=2,
            circle_radius=2,
            color=(255, 255, 255)  
        )
        
        self.hand_landmark_style = self.mp_drawing_styles.DrawingSpec(
            thickness=2,
            circle_radius=3,
            color=(255, 0, 0)  
        )
    
    def draw_pose(self, image, pose_landmarks):
        """
        Draw pose landmarks on image
        
        Args:
            image: BGR image to draw on
            pose_landmarks: MediaPipe pose landmarks
        """
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_landmark_style,
                connection_drawing_spec=self.pose_connection_style
            )
    
    def draw_hands(self, image, hand_results):
        """
        Draw hand landmarks on image
        
        Args:
            image: BGR image to draw on
            hand_results: MediaPipe hands results
        """
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks with custom style
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.hand_landmark_style,
                    connection_drawing_spec=self.hand_connection_style
                )
    
    def draw_both(self, image, pose_landmarks, hand_results):
        """
        Draw both pose and hand landmarks
        
        Args:
            image: BGR image to draw on
            pose_landmarks: MediaPipe pose landmarks
            hand_results: MediaPipe hands results
        """
        # Draw pose first (appears behind hands)
        self.draw_pose(image, pose_landmarks)
        
        # Draw hands on top
        self.draw_hands(image, hand_results)
    
    def draw_hand_info(self, image, hand_tracker, y_offset=30):
        """
        Draw hand detection info on image
        
        Args:
            image: BGR image to draw on
            hand_tracker: HandTracker instance
            y_offset: Vertical offset for text
        """
        hand_count = hand_tracker.get_hand_count()
        
        # Hand count
        text = f"Hands: {hand_count}"
        cv2.putText(
            image, text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),  # Magenta
            2
        )
        
        # Left/Right hand status
        if hand_tracker.has_left_hand():
            cv2.putText(
                image, "L",
                (10, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow
                2
            )
        
        if hand_tracker.has_right_hand():
            cv2.putText(
                image, "R",
                (35, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow
                2
            )
    
    # def draw_gesture_info(self, image, gesture_result, x=10, y=80):
    #     """
    #     Draw gesture detection info
        
    #     Args:
    #         image: BGR image to draw on
    #         gesture_result: Result dict from HandGestureDetector
    #         x: X position
    #         y: Y position
    #     """
    #     if gesture_result['gesture'] != 'Unknown' and gesture_result['gesture'] != 'No Hand':
    #         text = f"Gesture: {gesture_result['gesture']}"
    #         confidence = gesture_result['confidence']
            
    #         # Color based on confidence
    #         if confidence > 0.8:
    #             color = (0, 255, 0)  # Green
    #         elif confidence > 0.6:
    #             color = (0, 255, 255)  # Yellow
    #         else:
    #             color = (0, 165, 255)  # Orange
            
    #         cv2.putText(
    #             image, text,
    #             (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             color,
    #             2
    #         )
            
    #         # Draw confidence bar
    #         bar_width = 100
    #         bar_height = 10
    #         filled_width = int(bar_width * confidence)
            
    #         # Background bar
    #         cv2.rectangle(
    #             image,
    #             (x, y + 5),
    #             (x + bar_width, y + 5 + bar_height),
    #             (100, 100, 100),
    #             -1
    #         )
            
    #         # Filled bar
    #         cv2.rectangle(
    #             image,
    #             (x, y + 5),
    #             (x + filled_width, y + 5 + bar_height),
    #             color,
    #             -1
    #         )
    
    def draw_hand_controls_help(self, image, x=10, y=400):
        """
        Draw hand gesture control help text
        
        Args:
            image: BGR image to draw on
            x: X position
            y: Y position
        """
        help_text = [
            "HAND GESTURES:",
            "Thumbs Up: Start Record",
            "Fist: Stop Record",
            "Peace: Toggle Detection",
            "Open Hand: Reset Counter"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(
                image, text,
                (x, y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )


#############################################################################


class DisplayManager:
    """Handles all display and visualization tasks including exercise information"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def draw_angle_text(self, image, landmarks, angle, angle_name, landmark_index, color=(255, 255, 255), offset=(0, 0)):
        """Draw angle text at landmark position with optional offset"""
        if angle is not None:
            landmark_pixel = tuple(np.multiply([
                landmarks[landmark_index].x,
                landmarks[landmark_index].y
            ], [self.frame_width, self.frame_height]).astype(int))
            
            # Apply offset
            landmark_pixel = (landmark_pixel[0] + offset[0], landmark_pixel[1] + offset[1])
            
            cv2.putText(image, f"{angle_name}: {angle:.1f}", 
                       landmark_pixel, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_8)
        else:
            # Show "Unknown" if angle can't be calculated
            if landmarks[landmark_index].visibility > 0.3:
                landmark_pixel = tuple(np.multiply([
                    landmarks[landmark_index].x,
                    landmarks[landmark_index].y
                ], [self.frame_width, self.frame_height]).astype(int))
                
                # Apply offset
                landmark_pixel = (landmark_pixel[0] + offset[0], landmark_pixel[1] + offset[1])
                
                cv2.putText(image, f"{angle_name}: Unknown", 
                           landmark_pixel, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
    
    def draw_all_angles(self, image, landmarks, angles_dict):
        """Draw all angle measurements on the image"""
        if not landmarks:
            # No landmarks detected - show all as unknown
            positions = [(50, 70), (50, 100), (50, 130), (50, 160), (50, 190), (50, 220), (50, 250), (50, 280)]
            labels = ["LE: Unknown", "RE: Unknown", "LS: Unknown", "RS: Unknown", 
                     "LH: Unknown", "RH: Unknown", "LK: Unknown", "RK: Unknown"]
            
            for pos, label in zip(positions, labels):
                cv2.putText(image, label, pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
            return
        
        # Draw left elbow angle
        if angles_dict["left_elbow"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_elbow"], 
                               "LE", mp_pose.PoseLandmark.LEFT_ELBOW.value)
        else:
            self.draw_angle_text(image, landmarks, None, 
                               "LE", mp_pose.PoseLandmark.LEFT_ELBOW.value)
        
        # Draw right elbow angle
        if angles_dict["right_elbow"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_elbow"], 
                               "RE", mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        else:
            self.draw_angle_text(image, landmarks, None, 
                               "RE", mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        
        # Draw left shoulder angle (offset up from shoulder)
        if angles_dict["left_shoulder"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_shoulder"], 
                               "LS", mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                               offset=(0, -15))
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LS", mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                                   offset=(0, -15))
            else:
                cv2.putText(image, "LS: Unknown", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right shoulder angle (offset up from shoulder)
        if angles_dict["right_shoulder"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_shoulder"], 
                               "RS", mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                               offset=(0, -15))
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RS", mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                                   offset=(0, -15))
            else:
                cv2.putText(image, "RS: Unknown", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw left hip angle
        if angles_dict["left_hip"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_hip"], 
                               "LH", mp_pose.PoseLandmark.LEFT_HIP.value)
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LH", mp_pose.PoseLandmark.LEFT_HIP.value)
            else:
                cv2.putText(image, "LH: Unknown", (50, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right hip angle
        if angles_dict["right_hip"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_hip"], 
                               "RH", mp_pose.PoseLandmark.RIGHT_HIP.value)
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RH", mp_pose.PoseLandmark.RIGHT_HIP.value)
            else:
                cv2.putText(image, "RH: Unknown", (50, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw left knee angle
        if angles_dict["left_knee"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_knee"], 
                               "LK", mp_pose.PoseLandmark.LEFT_KNEE.value)
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LK", mp_pose.PoseLandmark.LEFT_KNEE.value)
            else:
                cv2.putText(image, "LK: Unknown", (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right knee angle
        if angles_dict["right_knee"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_knee"], 
                               "RK", mp_pose.PoseLandmark.RIGHT_KNEE.value)
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RK", mp_pose.PoseLandmark.RIGHT_KNEE.value)
            else:
                cv2.putText(image, "RK: Unknown", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
    
    def draw_distances(self, image, distances_dict):
        """Draw key distance measurements on the image"""
        if not distances_dict:
            return
        
        # Display distances in the top-right corner
        x_pos = image.shape[1] - 200  # Right side
        y_pos = 30
        
        # Shoulder width
        if distances_dict.get("shoulder_width") is not None:
            sw_text = f"SW: {distances_dict['shoulder_width']:.2f}m"
            cv2.putText(image, sw_text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_8)
            y_pos += 25
        
        # Left arm length
        if distances_dict.get("left_arm_length") is not None:
            la_text = f"LA: {distances_dict['left_arm_length']:.2f}m"
            cv2.putText(image, la_text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_8)
            y_pos += 25
        
        # Right arm length
        if distances_dict.get("right_arm_length") is not None:
            ra_text = f"RA: {distances_dict['right_arm_length']:.2f}m"
            cv2.putText(image, ra_text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_8)
            y_pos += 25
        
        # Torso height
        if distances_dict.get("torso_height") is not None:
            th_text = f"TH: {distances_dict['torso_height']:.2f}m"
            cv2.putText(image, th_text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_8)
    
    def draw_exercise_info(self, image, exercise_results):
        """Draw exercise detection information"""
        if not exercise_results:
            return
        
        # Exercise info panel background (semi-transparent)
        overlay = image.copy()
        cv2.rectangle(overlay, (350, 50), (630, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Exercise info panel border 
        border_color = (255, 255, 255)  # Default white
        if 'Abduction' in exercise_results.get('exercise', ''):
            border_color = (255, 255, 255)  
        elif 'Bicep' in exercise_results.get('exercise', ''):
            border_color = (255, 255, 255) 
        
        cv2.rectangle(image, (350, 50), (630, 300), border_color, 2)
        
        # Title with exercise-specific color
        title_color = (255, 255, 255) if 'Abduction' in exercise_results.get('exercise', '') else (255, 255, 255)
        cv2.putText(image, "EXERCISE TRACKER", (360, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_8)
        
        y_pos = 110
        
        # Exercise name and arm
        exercise_text = f"{exercise_results['exercise']} ({exercise_results['arm']})"
        cv2.putText(image, exercise_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_8)
        y_pos += 30
        
        # Rep count 
        rep_text = f"REPS: {exercise_results['rep_count']}"
        cv2.putText(image, rep_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_8)
        y_pos += 35
        
        # Current state
        state_text = f"State: {exercise_results['state']}"
        state_color = self._get_state_color(exercise_results['state'])
        cv2.putText(image, state_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2, cv2.LINE_8)
        y_pos += 25
        
        # Current angle
        if exercise_results['angle'] is not None:
            angle_text = f"Angle: {exercise_results['angle']}"
            cv2.putText(image, angle_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_8)
        else:
            cv2.putText(image, "Angle: N/A", (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_8)
        y_pos += 25
        
        # Peak and valley angles (if available)
        if exercise_results.get('peak_angle') is not None:
            peak_text = f"Peak: {exercise_results['peak_angle']}"
            cv2.putText(image, peak_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_8)
            y_pos += 20
            
        if exercise_results.get('valley_angle') is not None:
            valley_text = f"Valley: {exercise_results['valley_angle']}"
            cv2.putText(image, valley_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_8)
            y_pos += 20            

       
        # Feedback (word-wrapped if too long)
        feedback = exercise_results.get('feedback', '')
        if feedback:
            self._draw_wrapped_text(image, feedback, (360, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _get_state_color(self, state):
        """Get color for exercise state"""
        state_colors = {
            'Neutral': (255, 255, 255),
            'Up Phase': (0, 255, 0),
            'Down Phase': (255, 0, 0),
            'No Detection': (0, 0, 255)
        }
        return state_colors.get(state, (255, 255, 255))
    
    def _draw_wrapped_text(self, image, text, position, font, scale, color, thickness):
        """Draw text with word wrapping"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            
            if text_width < 250:  # Max width for text
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        x, y = position
        for i, line in enumerate(lines):
            cv2.putText(image, line, (x, y + i * 20), font, scale, color, thickness, cv2.LINE_8)
    
    def draw_status_info(self, image, fps, is_recording, filename=None, exercise_enabled=True, current_arm='right', current_exercise='bicep'):
        """Draw FPS, recording status, and instructions"""
        # Display FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (500, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_8)
        
        # Display recording status
        if is_recording:
            cv2.putText(image, "RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_8)
        
        # Display exercise detection status with exercise type
        exercise_name = "Bicep Curl" if current_exercise == 'bicep' else "Shoulder Abduction"
        
        #exercise_status = f"Exercise: {exercise_name} {'ON' if exercise_enabled else 'OFF'} ({current_arm.upper()})"
        #cv2.putText(image, exercise_status, (10, 60), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_8)
        
        # Display instructions 
        instructions = [
            "Press 'r' to start recording",
            "Press 's' to stop recording and save",
            "Press 'e' to toggle exercise detection",
            "Press 'a' to switch arm (L/R)",
            "Press 'w' to switch exercise type",  
            "Press 'x' to reset exercise counter",
            "Press 't' to show exercise stats",
            "Press 'q' to quit",
            "Press y to start reaction time test",
            "Press v to print reaction time stats",
        ]
        
        y_start = 310
        for i, instruction in enumerate(instructions):
            y_pos = y_start + i * 18
             # Draw outline by drawing black text at multiple offset positions
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                cv2.putText(image, instruction, (10 + dx, y_pos + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_8)
             # Draw main white text
            cv2.putText(image, instruction, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_8)