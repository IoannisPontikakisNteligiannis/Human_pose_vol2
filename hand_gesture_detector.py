"""
Hand Gesture Detector - Recognizes common hand gestures
Useful for controlling the application with hand signals
"""

import math


class HandGestureDetector:
    """Detects common hand gestures from hand landmarks"""
    
    def __init__(self):
        self.current_gesture = "Unknown"
        self.gesture_confidence = 0.0
        
    def calculate_distance(self, landmark1, landmark2):
        """Calculate Euclidean distance between two landmarks"""
        dx = landmark1.x - landmark2.x
        dy = landmark1.y - landmark2.y
        dz = landmark1.z - landmark2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def is_finger_extended(self, landmarks, finger_tip_id, finger_pip_id, finger_mcp_id):
        """
        Check if a finger is extended
        
        Args:
            landmarks: Hand landmarks
            finger_tip_id: Tip landmark index
            finger_pip_id: PIP joint landmark index
            finger_mcp_id: MCP joint landmark index
        
        Returns:
            bool: True if finger is extended
        """
        tip = landmarks.landmark[finger_tip_id]
        pip = landmarks.landmark[finger_pip_id]
        mcp = landmarks.landmark[finger_mcp_id]
        
        # Check if tip is higher than pip (for vertical orientation)
        # This is a simplified check - works best when hand is upright
        return tip.y < pip.y and pip.y < mcp.y
    
    def is_thumb_extended(self, landmarks):
        """Check if thumb is extended (different logic than fingers)"""
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        # Thumb extends horizontally, not vertically
        # Check if tip is further from wrist than IP joint
        wrist = landmarks.landmark[0]
        
        dist_tip = self.calculate_distance(thumb_tip, wrist)
        dist_ip = self.calculate_distance(thumb_ip, wrist)
        
        return dist_tip > dist_ip
    
#     def detect_gesture(self, hand_landmarks):
#         """
#         Detect gesture from hand landmarks
        
#         Args:
#             hand_landmarks: MediaPipe hand landmarks
            
#         Returns:
#             dict: {'gesture': str, 'confidence': float}
#         """
#         if not hand_landmarks:
#             return {'gesture': 'No Hand', 'confidence': 0.0}
        
#         landmarks = hand_landmarks
        
#         # Check which fingers are extended
#         thumb_extended = self.is_thumb_extended(landmarks)
#         index_extended = self.is_finger_extended(landmarks, 8, 6, 5)
#         middle_extended = self.is_finger_extended(landmarks, 12, 10, 9)
#         ring_extended = self.is_finger_extended(landmarks, 16, 14, 13)
#         pinky_extended = self.is_finger_extended(landmarks, 20, 18, 17)
        
#         # Count extended fingers
#         extended_count = sum([
#             thumb_extended,
#             index_extended,
#             middle_extended,
#             ring_extended,
#             pinky_extended
#         ])
        
#         # Detect specific gestures
#         gesture = "Unknown"
#         confidence = 0.8
        
#         # Thumbs Up
#         if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
#             gesture = "Thumbs Up"
#             confidence = 0.9
        
#         # Fist (no fingers extended)
#         elif extended_count == 0:
#             gesture = "Fist"
#             confidence = 0.95
        
#         # Open Hand (all fingers extended)
#         elif extended_count == 5:
#             gesture = "Open Hand"
#             confidence = 0.9
        
#         # Peace Sign (index and middle extended)
#         elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
#             gesture = "Peace Sign"
#             confidence = 0.85
        
#         # Pointing (only index extended)
#         elif index_extended and not any([thumb_extended, middle_extended, ring_extended, pinky_extended]):
#             gesture = "Pointing"
#             confidence = 0.85
        
#         # OK Sign (thumb and index forming circle)
#         elif thumb_extended and index_extended:
#             thumb_tip = landmarks.landmark[4]
#             index_tip = landmarks.landmark[8]
#             distance = self.calculate_distance(thumb_tip, index_tip)
            
#             if distance < 0.05:  # Close together
#                 gesture = "OK Sign"
#                 confidence = 0.8
        
#         # Number gestures
#         elif extended_count == 1:
#             gesture = "One"
#         elif extended_count == 2:
#             gesture = "Two"
#         elif extended_count == 3:
#             gesture = "Three"
#         elif extended_count == 4:
#             gesture = "Four"
        
#         self.current_gesture = gesture
#         self.gesture_confidence = confidence
        
#         return {
#             'gesture': gesture,
#             'confidence': confidence,
#             'extended_fingers': {
#                 'thumb': thumb_extended,
#                 'index': index_extended,
#                 'middle': middle_extended,
#                 'ring': ring_extended,
#                 'pinky': pinky_extended
#             }
#         }
    
#     def get_current_gesture(self):
#         """Get the most recently detected gesture"""
#         return self.current_gesture


# # Gesture-based control mappings (example)
# GESTURE_CONTROLS = {
#     'Thumbs Up': 'start_recording',
#     'Fist': 'stop_recording',
#     'Peace Sign': 'toggle_exercise_detection',
#     'Open Hand': 'reset_counter',
#     'Pointing': 'switch_arm'
# }