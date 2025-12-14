"""RealSense demo: overlays colormapped depth on color frames and prints nose depth.

Usage: `python realsense_demo.py`

Press `q` to quit.
"""
import time
import cv2
import numpy as np
import mediapipe as mp

from realsense_adapter import RealSenseCapture


def depth_colormap(depth_mm, max_depth_mm=4000):
    """Convert uint16 depth (mm) to a BGR colormap image (uint8).

    depth_mm: numpy array HxW (uint16) or None
    max_depth_mm: clipping maximum for color scaling
    """
    if depth_mm is None:
        return None
    # Normalize and convert to 8-bit
    depth_clipped = np.clip(depth_mm, 0, max_depth_mm)
    depth_8u = (depth_clipped.astype(np.float32) / max_depth_mm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color


def main():
    FRAME_W = 640
    FRAME_H = 480
    FPS = 30

    try:
        cap = RealSenseCapture(FRAME_W, FRAME_H, FPS)
    except Exception as e:
        print("Failed to open RealSense camera:", e)
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("Press 'q' to quit. Overlaying depth (jet) on color frames.")
    last_print = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break

            depth_mm = cap.get_last_depth()  # uint16 mm or None

            # Draw depth overlay
            depth_vis = depth_colormap(depth_mm, max_depth_mm=4000)
            if depth_vis is not None:
                # Resize colormap if needed
                if depth_vis.shape[:2] != frame.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
                overlay = cv2.addWeighted(frame, 0.6, depth_vis, 0.4, 0)
            else:
                overlay = frame

            # Run MediaPipe to get nose landmark occasionally (cheap)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            nose_depth = None
            if results and results.pose_landmarks:
                lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                nose_depth = cap.get_depth_at(px, py)
                if nose_depth is not None and time.time() - last_print > 0.2:
                    print(f"Nose depth: {nose_depth:.3f} m (px={px},{py})")
                    last_print = time.time()

            # Annotate overlay with text
            status_text = f"Nose depth: {nose_depth:.3f} m" if nose_depth is not None else "Nose depth: N/A"
            cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('RealSense Demo - Depth Overlay', overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
