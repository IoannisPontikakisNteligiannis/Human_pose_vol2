"""
Hand Gesture Detector - Recognizes common hand gestures
Useful for controlling the application with hand signals
"""

import math
import numpy as np


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
        Check if a finger is extended using angle-based detection
        
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
        
        # Calculate vectors
        # Vector from MCP to PIP
        v1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
        # Vector from PIP to tip
        v2 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
        
        # Calculate angle between vectors (cosine similarity)
        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return False
            
        cos_angle = dot_product / (v1_norm * v2_norm)
        # Clamp to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Finger is extended if angle is small (vectors are aligned)
        return angle < np.pi / 2  # ~90 degrees, more lenient
    
    def is_thumb_extended(self, landmarks):
        """Check if thumb is extended using angle-based detection"""
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        wrist = landmarks.landmark[0]
        
        # Calculate vectors
        # Vector from MCP to IP
        v1 = np.array([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y, thumb_ip.z - thumb_mcp.z])
        # Vector from IP to tip
        v2 = np.array([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y, thumb_tip.z - thumb_ip.z])
        
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return False
            
        cos_angle = dot_product / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Thumb is extended if angle is small (vectors are aligned)
        return angle < np.pi / 2.5  # ~72 degrees, more lenient for thumb
    
    def detect_gesture(self, hand_landmarks):
        """
        Detect gesture from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            dict: {'gesture': str, 'confidence': float}
        """
        if not hand_landmarks:
            return {'gesture': 'No Hand', 'confidence': 0.0}
        
        landmarks = hand_landmarks
        
        # Check which fingers are extended
        thumb_extended = self.is_thumb_extended(landmarks)
        index_extended = self.is_finger_extended(landmarks, 8, 6, 5)
        middle_extended = self.is_finger_extended(landmarks, 12, 10, 9)
        ring_extended = self.is_finger_extended(landmarks, 16, 14, 13)
        pinky_extended = self.is_finger_extended(landmarks, 20, 18, 17)
        
        # Count extended fingers
        extended_count = sum([
            thumb_extended,
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended
        ])
        
        # # Detect specific gestures
        # gesture = "Unknown"
        # confidence = 0.8
        
        # # Thumbs Up
        # if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        #     gesture = "Thumbs Up"
        #     confidence = 0.5
        
        # # Fist (no fingers extended)
        # elif extended_count == 0:
        #     gesture = "Fist"
        #     confidence = 0.5
        
        # # Open Hand (all fingers extended)
        # elif extended_count == 5:
        #     gesture = "Open Hand"
        #     confidence = 0.5
        
        # # Peace Sign (index and middle extended)
        # elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
        #     gesture = "Peace Sign"
        #     confidence = 0.85
        
        # # Pointing (only index extended)
        # elif index_extended and not any([thumb_extended, middle_extended, ring_extended, pinky_extended]):
        #     gesture = "Pointing"
        #     confidence = 0.85
        
        # Pinch/Spread/Touch with thumb and index (more lenient)
        if thumb_extended and index_extended:
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            distance = self.calculate_distance(thumb_tip, index_tip)
            
            if distance < 0.08:  # Very close - touch for move
                gesture = "Touch"
                confidence = 0.5
            elif distance < 0.15:  # Close together - pinch for zoom in
                gesture = "Pinch"
                confidence = 0.5
            else:  # Far apart - spread for zoom out
                gesture = "Spread"
                confidence = 0.5
        # Alternative: just thumb and index close (even if not fully extended)
        elif not thumb_extended and not index_extended:
            # Check if thumb and index tips are close together
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            distance = self.calculate_distance(thumb_tip, index_tip)
            
            if distance < 0.06:  # Very close - touch
                gesture = "Touch"
                confidence = 0.7
            elif distance < 0.12:  # Close - pinch
                gesture = "Pinch"
                confidence = 0.6
        
        # # Number gestures
        # elif extended_count == 1:
        #     gesture = "One"
        #     confidence = 0.8
        # elif extended_count == 2:
        #     gesture = "Two"
        #     confidence = 0.8
        # elif extended_count == 3:
        #     gesture = "Three"
        #     confidence = 0.8
        # elif extended_count == 4:
        #     gesture = "Four"
        #     confidence = 0.8
        else:
             gesture = "Unknown"
             confidence = 0.0
        
        self.current_gesture = gesture
        self.gesture_confidence = confidence
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'extended_fingers': {
                'thumb': thumb_extended,
                'index': index_extended,
                'middle': middle_extended,
                'ring': ring_extended,
                'pinky': pinky_extended
            }
        }
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt(
            (point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2 +
            (point1.z - point2.z) ** 2
        )
    
# #     def get_current_gesture(self):
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
