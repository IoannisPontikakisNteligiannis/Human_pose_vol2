"""
Hand Tracker Module - MediaPipe Hands Integration
Handles hand detection, landmark tracking, and gesture recognition
"""

import mediapipe as mp
import time


class HandTracker:
    """Tracks hands using MediaPipe Hands solution"""
    
    def __init__(self, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        """
        Initialize the hand tracker
        
        Args:
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # 0=lite, 1=full (lite is faster)
        )
        
        # Tracking data
        self.results = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.processing_time = 0
        
    def process(self, image_rgb):
        """
        Process frame and detect hands
        
        Args:
            image_rgb: RGB image frame
            
        Returns:
            results: MediaPipe hands results object
        """
        start_time = time.time()
        
        # Make detection
        self.results = self.hands.process(image_rgb)
        
        # Separate left and right hands
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        
        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_landmarks, handedness in zip(
                self.results.multi_hand_landmarks,
                self.results.multi_handedness
            ):
                # handedness.classification[0].label is 'Left' or 'Right'
                label = handedness.classification[0].label
                
                if label == 'Left':
                    self.left_hand_landmarks = hand_landmarks
                elif label == 'Right':
                    self.right_hand_landmarks = hand_landmarks
        
        self.processing_time = time.time() - start_time
        return self.results
    
    def has_hands(self):
        """Check if any hands are detected"""
        return self.results and self.results.multi_hand_landmarks
    
    def has_left_hand(self):
        """Check if left hand is detected"""
        return self.left_hand_landmarks is not None
    
    def has_right_hand(self):
        """Check if right hand is detected"""
        return self.right_hand_landmarks is not None
    
    def get_hand_count(self):
        """Get number of detected hands"""
        if not self.results or not self.results.multi_hand_landmarks:
            return 0
        return len(self.results.multi_hand_landmarks)
    
    def get_landmark(self, hand='right', landmark_id=0):
        """
        Get specific landmark from a hand
        
        Args:
            hand: 'left' or 'right'
            landmark_id: Landmark index (0-20)
            
        Returns:
            landmark object or None
        """
        if hand == 'left' and self.left_hand_landmarks:
            return self.left_hand_landmarks.landmark[landmark_id]
        elif hand == 'right' and self.right_hand_landmarks:
            return self.right_hand_landmarks.landmark[landmark_id]
        return None
    
    def get_all_landmarks(self, hand='right'):
        """
        Get all landmarks for a specific hand
        
        Args:
            hand: 'left' or 'right'
            
        Returns:
            list of landmarks or None
        """
        if hand == 'left':
            return self.left_hand_landmarks
        elif hand == 'right':
            return self.right_hand_landmarks
        return None
    
    def close(self):
        """Clean up resources"""
        if self.hands:
            self.hands.close()


# Hand landmark indices for reference
HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5,
    'INDEX_FINGER_PIP': 6,
    'INDEX_FINGER_DIP': 7,
    'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9,
    'MIDDLE_FINGER_PIP': 10,
    'MIDDLE_FINGER_DIP': 11,
    'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13,
    'RING_FINGER_PIP': 14,
    'RING_FINGER_DIP': 15,
    'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20
}