"""Simple adapter to use Intel RealSense camera as a drop-in replacement
for `cv2.VideoCapture` within this project. It exposes a minimal API:

- `RealSenseCapture(width, height, fps)` -> object with `isOpened()`, `read()`, `release()`, `set()`.

Requires: `pyrealsense2` (and the Intel RealSense SDK/drivers installed on Windows).
"""
from typing import Optional
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    raise ImportError("pyrealsense2 is required for RealSenseCapture: " + str(e))


class RealSenseCapture:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Enable color + depth streams (depth as z16)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        try:
            self.pipeline_profile = self.pipeline.start(self.config)
        except Exception as e:
            self.pipeline = None
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        # Align depth to color so depth map matches color image pixels
        self.align = rs.align(rs.stream.color)

        self._is_opened = True
        self.last_depth = None  # aligned depth numpy array (uint16, millimeters)
        # Cache color intrinsics if available
        try:
            color_stream = self.pipeline_profile.get_stream(rs.stream.color)
            video_profile = color_stream.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            # intr contains: width, height, ppx, ppy, fx, fy, model, coeffs
            self.color_intrinsics = {
                'width': intr.width,
                'height': intr.height,
                'ppx': intr.ppx,
                'ppy': intr.ppy,
                'fx': intr.fx,
                'fy': intr.fy,
                'model': intr.model,
                'coeffs': intr.coeffs
            }
        except Exception:
            self.color_intrinsics = None

    def isOpened(self) -> bool:
        return self._is_opened and (self.pipeline is not None)

    def read(self):
        """Return tuple (ret, color_frame) where color_frame is a BGR numpy array compatible with OpenCV.

        The latest aligned depth frame is stored in `self.last_depth` (uint16, millimeters).
        Use `get_last_depth()` or `get_depth_at(x, y)` to retrieve depth.
        """
        if not self.isOpened():
            return False, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            # Align depth to color
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame:
                return False, None

            color_img = np.asanyarray(color_frame.get_data())

            if depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())
                # Store aligned depth in millimeters (uint16)
                self.last_depth = depth_img
            else:
                self.last_depth = None

            return True, color_img
        except Exception:
            return False, None

    def get_last_depth(self) -> Optional[np.ndarray]:
        """Return the last aligned depth image (uint16 in millimeters) or None."""
        return self.last_depth

    def get_color_intrinsics(self):
        """Return a dict with color intrinsics (fx, fy, ppx, ppy, width, height) or None."""
        return self.color_intrinsics

    def get_depth_info(self):
        """Return a dict suitable for `angle_calculator.calculate_all_angles` when using depth.

        Contains keys: 'depth' (uint16 mm numpy array), 'intrinsics' (dict), 'width', 'height'.
        """
        if self.last_depth is None or self.color_intrinsics is None:
            return None
        return {
            'depth': self.last_depth,
            'intrinsics': self.color_intrinsics,
            'width': self.color_intrinsics.get('width', None),
            'height': self.color_intrinsics.get('height', None)
        }

    def get_depth_at(self, x: int, y: int) -> Optional[float]:
        """Return depth at pixel coordinates (x, y) in meters, or None if unavailable.

        Note: x is column (horizontal), y is row (vertical)."""
        if self.last_depth is None:
            return None
        h, w = self.last_depth.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        depth_mm = int(self.last_depth[y, x])
        if depth_mm == 0:
            return None
        return depth_mm / 1000.0

    def release(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self._is_opened = False

    def set(self, prop_id, value):
        # Minimal implementation: support changing width/height if requested (does not restart pipeline)
        try:
            if prop_id == 3:  # CV_CAP_PROP_FRAME_WIDTH
                self.width = int(value)
                return True
            if prop_id == 4:  # CV_CAP_PROP_FRAME_HEIGHT
                self.height = int(value)
                return True
            return False
        except Exception:
            return False

    def __del__(self):
        self.release()
