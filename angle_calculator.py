import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Smoothing and depth reliability parameters
_USE_SMOOTHING = False  # Enable angle smoothing
# _SMOOTHING_ALPHA = 0.4  # EMA alpha (0..1), higher = less smoothing
_DEPTH_STD_THRESHOLD_MM = 80  # stddev in mm within 3x3 window above which depth is unreliable
_MAX_DEPTH_MM = 4000  # ignore depths beyond this (in mm)

# Internal smoothing state: previous smoothed angles per joint
_prev_smoothed_angles = {}

"""  Exponential moving average smoothing of angles (disabled by default)"""
# def _smooth_angles(raw_angles: dict) -> dict:
#     """Apply exponential moving average smoothing to angles in-place and return new dict."""
#     if not _USE_SMOOTHING:
#         return raw_angles
#     smoothed = {}
#     for k, v in raw_angles.items():
#         if v is None:
#             # Keep previous smoothed if exists, otherwise None
#             smoothed[k] = _prev_smoothed_angles.get(k)
#             continue
#         prev = _prev_smoothed_angles.get(k)
#         if prev is None:
#             s = v
#         else:
#             s = prev * (1.0 - _SMOOTHING_ALPHA) + v * _SMOOTHING_ALPHA
#         smoothed[k] = s
#         _prev_smoothed_angles[k] = s
#     return smoothed


def calculate_angle(point1, point2, point3):
    """
    Calculate the angle at point2 (point1-point2-point3).
    Returns angle in 0-180°.
    
    Parameters:
        point1 (list): First point [x, y]
        point2 (list): Middle point (apex) [x, y]
        point3 (list): Third point [x, y]
    
    Returns:
        float: Angle in degrees (0-180), or None if invalid
    """
    try:
        # Check if any of the landmarks has low visibility or invalid position
        if None in (point1, point2, point3):
            return None
            
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        
        # Calculate vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Check for zero distance (landmarks too close or improperly detected)
        vector1_mag = np.linalg.norm(vector1)
        vector2_mag = np.linalg.norm(vector2)

        # If vectors are too small, landmarks are likely misdetected
        MIN_VECTOR_LENGTH = 0.03 # Minimum sensible distance between landmarks (meters or normalized)
        if vector1_mag < MIN_VECTOR_LENGTH or vector2_mag < MIN_VECTOR_LENGTH:
            return None
        
        # Dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        
        # Cosine of the angle
        cos_angle = dot_product / (vector1_mag * vector2_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        
        # Angle in degrees (0-180°)
        angle = np.arccos(cos_angle) * 180.0 / np.pi     
        return angle
    except (TypeError, ValueError) as e:
        print(f"Error calculating angle: {e}")
        return None

def check_landmark_visibility(landmarks, landmark_indices, min_visibility=0.7):
    """
    Check if all landmarks in the specified indices have sufficient visibility.
    
    Parameters:
        landmarks: MediaPipe landmarks object
        landmark_indices: List of landmark indices to check
        min_visibility: Minimum visibility threshold (0-1)
        
    Returns:
        bool: True if all landmarks have sufficient visibility, False otherwise
    """
    if not landmarks:
        return False
        
    for idx in landmark_indices:
        if idx >= len(landmarks) or landmarks[idx].visibility < min_visibility:
            return False
    
    return True

def calculate_all_angles(landmarks, depth_info=None):
    """
    Calculate all joint angles from MediaPipe landmarks.
    
    Parameters:
        landmarks: MediaPipe landmarks object
        
    Returns:
        dict: Dictionary containing all calculated angles
    """
    angles_dict = {
        "left_elbow": None,
        "right_elbow": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "left_hip": None,
        "right_hip": None,
        "left_knee": None,
        "right_knee": None
    }
    
    if not landmarks:
        return angles_dict

    # Determine if depth-based 3D coordinates should be used

    # Determine if depth-based 3D coordinates should be used
    use_depth = False
    depth_arr = None
    intr = None
    img_w = None
    img_h = None
    if depth_info is not None:
        try:
            depth_arr = depth_info.get('depth')
            intr = depth_info.get('intrinsics')
            img_w = depth_info.get('width')
            img_h = depth_info.get('height')
            if depth_arr is not None and intr is not None:
                use_depth = True
        except Exception:
            use_depth = False

    # Compute a local reference center (pelvis mid-point preferred, fall back to shoulders)
    try:
        lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        ref_cx = (lhip.x + rhip.x) / 2.0
        ref_cy = (lhip.y + rhip.y) / 2.0
    except Exception:
        # fallback to shoulder midpoint
        try:
            lsh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rsh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            ref_cx = (lsh.x + rsh.x) / 2.0
            ref_cy = (lsh.y + rsh.y) / 2.0
        except Exception:
            ref_cx = 0.5
            ref_cy = 0.5

    def landmark_to_point(idx):
        lm = landmarks[idx]
        # If depth available, back-project to metric 3D coordinates
        if use_depth and depth_arr is not None and intr is not None:
            px = int(lm.x * (img_w if img_w else intr['width']))
            py = int(lm.y * (img_h if img_h else intr['height']))
            h, w = depth_arr.shape
            # clamp
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            depth_mm = int(depth_arr[py, px])
            if depth_mm == 0:
                return [None]
            z = depth_mm / 1000.0
            fx = intr.get('fx')
            fy = intr.get('fy')
            ppx = intr.get('ppx')
            ppy = intr.get('ppy')
            # Back-project to meters
            X = (px - ppx) * z / fx
            Y = (py - ppy) * z / fy
            Z = z
            return [X, Y, Z]
        else:
            # Use normalized image coordinates (x,y) relative to pelvis/shoulder center
            return [lm.x - ref_cx, lm.y - ref_cy]
    
    # Check visibility for different body parts
    left_arm_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_ELBOW.value,
         mp_pose.PoseLandmark.LEFT_WRIST.value],
        min_visibility=0.7
    )
    
    right_arm_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_ELBOW.value,
         mp_pose.PoseLandmark.RIGHT_WRIST.value],
        min_visibility=0.7
    )
    
    left_shoulder_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_ELBOW.value],
        min_visibility=0.7
    )
    
    right_shoulder_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        min_visibility=0.7
    )
    
    # Check visibility for hip angles
    left_hip_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_KNEE.value],
        min_visibility=0.7
    )
    
    right_hip_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_KNEE.value],
        min_visibility=0.7
    )
    
    # Check visibility for knee angles
    left_knee_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_KNEE.value,
         mp_pose.PoseLandmark.LEFT_ANKLE.value],
        min_visibility=0.7
    )
    
    right_knee_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_KNEE.value,
         mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        min_visibility=0.7
    )
    
    # Calculate LEFT elbow angle if visible
    if left_arm_visible:
        left_shoulder = landmark_to_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_elbow = landmark_to_point(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        left_wrist = landmark_to_point(mp_pose.PoseLandmark.LEFT_WRIST.value)

        if None in (left_shoulder, left_elbow, left_wrist) or left_shoulder==[None] or left_elbow==[None] or left_wrist==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["left_elbow"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT elbow angle if visible
    if right_arm_visible:
        right_shoulder = landmark_to_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_elbow = landmark_to_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        right_wrist = landmark_to_point(mp_pose.PoseLandmark.RIGHT_WRIST.value)

        if None in (right_shoulder, right_elbow, right_wrist) or right_shoulder==[None] or right_elbow==[None] or right_wrist==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["right_elbow"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate LEFT shoulder angle if visible
    if left_shoulder_visible:
        left_hip = landmark_to_point(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_shoulder = landmark_to_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_elbow = landmark_to_point(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        if None in (left_hip, left_shoulder, left_elbow) or left_hip==[None] or left_shoulder==[None] or left_elbow==[None]:
            angles_dict["left_shoulder"] = None
        else:
            angles_dict["left_shoulder"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Calculate RIGHT shoulder angle if visible
    if right_shoulder_visible:
        right_hip = landmark_to_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_shoulder = landmark_to_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_elbow = landmark_to_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        if None in (right_hip, right_shoulder, right_elbow) or right_hip==[None] or right_shoulder==[None] or right_elbow==[None]:
            angles_dict["right_shoulder"] = None
        else:
            angles_dict["right_shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    # Calculate LEFT hip angle if visible
    if left_hip_visible:
        left_shoulder = landmark_to_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_hip = landmark_to_point(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee = landmark_to_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
        if None in (left_shoulder, left_hip, left_knee) or left_shoulder==[None] or left_hip==[None] or left_knee==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        # Convert to anatomical angle (0° = straight leg, increases with flexion)
        angles_dict["left_hip"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT hip angle if visible
    if right_hip_visible:
        right_shoulder = landmark_to_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_hip = landmark_to_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee = landmark_to_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        if None in (right_shoulder, right_hip, right_knee) or right_shoulder==[None] or right_hip==[None] or right_knee==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        # Convert to anatomical angle (0° = straight leg, increases with flexion)
        angles_dict["right_hip"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate LEFT knee angle if visible
    if left_knee_visible:
        left_hip = landmark_to_point(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee = landmark_to_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_ankle = landmark_to_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        if None in (left_hip, left_knee, left_ankle) or left_hip==[None] or left_knee==[None] or left_ankle==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(left_hip, left_knee, left_ankle)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["left_knee"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT knee angle if visible
    if right_knee_visible:
        right_hip = landmark_to_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee = landmark_to_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_ankle = landmark_to_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        if None in (right_hip, right_knee, right_ankle) or right_hip==[None] or right_knee==[None] or right_ankle==[None]:
            vector_angle = None
        else:
            vector_angle = calculate_angle(right_hip, right_knee, right_ankle)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["right_knee"] = 180 - vector_angle if vector_angle is not None else None
    
    # Apply smoothing to reduce jitter/offsets across frames
    try:
        angles_dict = angles_dict #_smooth_angles(angles_dict) If smoothing desired
    except Exception:
        pass

   
    return angles_dict
