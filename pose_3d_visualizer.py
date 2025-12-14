import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp

mp_pose = mp.solutions.pose

# MediaPipe pose connections for drawing the skeleton
POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
    (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
]


def extract_3d_points(landmarks, depth_info=None, img_width=None, img_height=None):
    """
    Extract 3D points from MediaPipe landmarks using depth information.

    Parameters:
        landmarks: MediaPipe landmarks object
        depth_info: Dict with 'depth', 'intrinsics', 'width', 'height'
        img_width: Image width (fallback)
        img_height: Image height (fallback)

    Returns:
        dict: Dictionary mapping landmark indices to 3D points [X, Y, Z] or None
    """
    if not landmarks:
        return {}

    points_3d = {}

    # Setup depth parameters
    use_depth = False
    depth_arr = None
    intr = None
    if depth_info is not None:
        depth_arr = depth_info.get('depth')
        intr = depth_info.get('intrinsics')
        if depth_arr is not None and intr is not None:
            use_depth = True

    for idx in range(len(landmarks)):
        lm = landmarks[idx]

        if use_depth:
            # Convert normalized coordinates to pixel coordinates
            px = int(lm.x * (img_width if img_width else intr['width']))
            py = int(lm.y * (img_height if img_height else intr['height']))

            h, w = depth_arr.shape
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            # Get 3x3 window around the pixel for reliability check
            x0 = max(0, px - 1)
            x1 = min(w - 1, px + 1)
            y0 = max(0, py - 1)
            y1 = min(h - 1, py + 1)
            window = depth_arr[y0:y1+1, x0:x1+1].astype(np.int32)
            valid = window[window > 0]

            if valid.size == 0:
                points_3d[idx] = None
                continue

            mean_depth = int(np.mean(valid))
            std_depth = int(np.std(valid))

            # Depth reliability check
            if std_depth > 80 or mean_depth > 4000:  # 80mm std threshold, 4m max depth
                points_3d[idx] = None
                continue

            z = mean_depth / 1000.0  # Convert mm to meters

            # Back-project to 3D using camera intrinsics
            fx = intr.get('fx')
            fy = intr.get('fy')
            ppx = intr.get('ppx')
            ppy = intr.get('ppy')

            X = (px - ppx) * z / fx
            Y = (py - ppy) * z / fy
            Z = z

            points_3d[idx] = [X, Y, Z]
        else:
            # Use normalized 2D coordinates (set Z=0)
            points_3d[idx] = [lm.x, lm.y, 0.0]

    return points_3d


def calculate_key_distances(points_3d):
    """
    Calculate key distances between landmarks.

    Parameters:
        points_3d: Dict of 3D points from extract_3d_points

    Returns:
        dict: Dictionary of calculated distances
    """
    distances = {}

    # Shoulder width
    left_shoulder = points_3d.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = points_3d.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    if left_shoulder and right_shoulder:
        distances['shoulder_width'] = np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder))

    # Arm lengths
    left_wrist = points_3d.get(mp_pose.PoseLandmark.LEFT_WRIST.value)
    if left_shoulder and left_wrist:
        distances['left_arm_length'] = np.linalg.norm(np.array(left_wrist) - np.array(left_shoulder))

    right_wrist = points_3d.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    if right_shoulder and right_wrist:
        distances['right_arm_length'] = np.linalg.norm(np.array(right_wrist) - np.array(right_shoulder))

    # Leg lengths
    left_hip = points_3d.get(mp_pose.PoseLandmark.LEFT_HIP.value)
    left_ankle = points_3d.get(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    if left_hip and left_ankle:
        distances['left_leg_length'] = np.linalg.norm(np.array(left_ankle) - np.array(left_hip))

    right_hip = points_3d.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
    right_ankle = points_3d.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    if right_hip and right_ankle:
        distances['right_leg_length'] = np.linalg.norm(np.array(right_ankle) - np.array(right_hip))

    # Torso height
    if left_shoulder and left_hip:
        distances['torso_height'] = np.linalg.norm(np.array(left_hip) - np.array(left_shoulder))

    return distances


def visualize_3d_pose(points_3d, distances=None, title="3D Pose Visualization"):
    """
    Visualize the 3D pose with landmarks and connections.

    Parameters:
        points_3d: Dict of 3D points from extract_3d_points
        distances: Dict of distances from calculate_key_distances
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract valid points
    valid_points = {idx: point for idx, point in points_3d.items() if point is not None}

    if not valid_points:
        ax.text(0, 0, 0, "No valid 3D points to display", fontsize=12)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(title)
        plt.show()
        return

    # Plot landmarks
    xs, ys, zs = [], [], []
    for idx, point in valid_points.items():
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])

    ax.scatter(xs, ys, zs, c='red', s=50, alpha=0.8)

    # Add landmark labels
    landmark_names = {
        mp_pose.PoseLandmark.NOSE.value: "Nose",
        mp_pose.PoseLandmark.LEFT_EYE.value: "L_Eye",
        mp_pose.PoseLandmark.RIGHT_EYE.value: "R_Eye",
        mp_pose.PoseLandmark.LEFT_EAR.value: "L_Ear",
        mp_pose.PoseLandmark.RIGHT_EAR.value: "R_Ear",
        mp_pose.PoseLandmark.LEFT_SHOULDER.value: "L_Shoulder",
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value: "R_Shoulder",
        mp_pose.PoseLandmark.LEFT_ELBOW.value: "L_Elbow",
        mp_pose.PoseLandmark.RIGHT_ELBOW.value: "R_Elbow",
        mp_pose.PoseLandmark.LEFT_WRIST.value: "L_Wrist",
        mp_pose.PoseLandmark.RIGHT_WRIST.value: "R_Wrist",
        mp_pose.PoseLandmark.LEFT_HIP.value: "L_Hip",
        mp_pose.PoseLandmark.RIGHT_HIP.value: "R_Hip",
        mp_pose.PoseLandmark.LEFT_KNEE.value: "L_Knee",
        mp_pose.PoseLandmark.RIGHT_KNEE.value: "R_Knee",
        mp_pose.PoseLandmark.LEFT_ANKLE.value: "L_Ankle",
        mp_pose.PoseLandmark.RIGHT_ANKLE.value: "R_Ankle",
    }

    for idx, point in valid_points.items():
        name = landmark_names.get(idx, f"LM{idx}")
        ax.text(point[0], point[1], point[2], name, fontsize=8)

    # Draw skeleton connections
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0].value
        end_idx = connection[1].value

        if start_idx in valid_points and end_idx in valid_points:
            start_point = valid_points[start_idx]
            end_point = valid_points[end_idx]

            ax.plot([start_point[0], end_point[0]],
                   [start_point[1], end_point[1]],
                   [start_point[2], end_point[2]], 'b-', linewidth=2, alpha=0.7)

    # Display distances if provided
    if distances:
        dist_text = "Key Distances:\n"
        for key, value in distances.items():
            if value is not None:
                dist_text += f"{key}: {value:.2f}m\n"
        ax.text2D(0.02, 0.98, dist_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = 0.5  # Default range if no points
    if xs and ys and zs:
        x_range = max(xs) - min(xs) if len(xs) > 1 else 1
        y_range = max(ys) - min(ys) if len(ys) > 1 else 1
        z_range = max(zs) - min(zs) if len(zs) > 1 else 1
        max_range = max(x_range, y_range, z_range)

    ax.set_xlim([np.mean(xs) - max_range/2, np.mean(xs) + max_range/2])
    ax.set_ylim([np.mean(ys) - max_range/2, np.mean(ys) + max_range/2])
    ax.set_zlim([np.mean(zs) - max_range/2, np.mean(zs) + max_range/2])

    plt.tight_layout()
    plt.show()


def create_3d_pose_map(landmarks, depth_info=None, img_width=None, img_height=None, show_plot=True):
    """
    Create and optionally display a 3D map of the pose with distances.

    Parameters:
        landmarks: MediaPipe landmarks object
        depth_info: Depth information dict
        img_width: Image width
        img_height: Image height
        show_plot: Whether to display the plot

    Returns:
        tuple: (points_3d, distances) dictionaries
    """
    # Extract 3D points
    points_3d = extract_3d_points(landmarks, depth_info, img_width, img_height)

    # Calculate distances
    distances = calculate_key_distances(points_3d)

    # Visualize if requested
    if show_plot:
        title = "3D Pose Map with Distances"
        if depth_info:
            title += " (Depth-enhanced)"
        else:
            title += " (2D with Z=0)"

        visualize_3d_pose(points_3d, distances, title)

    return points_3d, distances


# Example usage
if __name__ == "__main__":
    # This would be used with actual landmark data from your pose detection
    print("3D Pose Visualizer")
    print("Use create_3d_pose_map(landmarks, depth_info) to create visualizations")
    print("Example: points_3d, distances = create_3d_pose_map(landmarks, depth_info)")