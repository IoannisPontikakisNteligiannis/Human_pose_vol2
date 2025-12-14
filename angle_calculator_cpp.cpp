// angle_calculator_cpp.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>

#define _USE_MATH_DEFINES
#include <cmath>


#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;



#include <array>


namespace py = pybind11;

constexpr float MIN_VECTOR_LENGTH = 0.03f;
constexpr float MIN_VISIBILITY = 0.7f;

struct Point2D {
    float x, y;
};

// Fast angle calculation
inline float calculate_angle_fast(const Point2D& p1, const Point2D& p2, const Point2D& p3) {
    // Vectors
    float v1_x = p1.x - p2.x;
    float v1_y = p1.y - p2.y;
    float v2_x = p3.x - p2.x;
    float v2_y = p3.y - p2.y;
    
    // Magnitudes
    float mag1 = std::sqrt(v1_x * v1_x + v1_y * v1_y);
    float mag2 = std::sqrt(v2_x * v2_x + v2_y * v2_y);
    
    // Check minimum vector length
    if (mag1 < MIN_VECTOR_LENGTH || mag2 < MIN_VECTOR_LENGTH) {
        return -1.0f; // Invalid marker
    }
    
    // Dot product and angle
    float dot = v1_x * v2_x + v1_y * v2_y;
    float cos_angle = dot / (mag1 * mag2);
    
    // Clamp to avoid numerical errors
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
    
    return std::acos(cos_angle) * 180.0f / M_PI;
}

// Check visibility for a group of landmarks
inline bool check_visibility(py::object landmarks, const std::array<int, 3>& indices) {
    try {
        for (int idx : indices) {
            auto lm = landmarks.attr("__getitem__")(idx);
            float visibility = lm.attr("visibility").cast<float>();
            if (visibility < MIN_VISIBILITY) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

// Extract 2D point from landmark
inline Point2D get_point(py::object landmarks, int idx) {
    auto lm = landmarks.attr("__getitem__")(idx);
    return {lm.attr("x").cast<float>(), lm.attr("y").cast<float>()};
}

py::dict calculate_all_angles_cpp(py::object landmarks) {
    py::dict angles;
    
    // Initialize all angles to None
    angles["left_elbow"] = py::none();
    angles["right_elbow"] = py::none();
    angles["left_shoulder"] = py::none();
    angles["right_shoulder"] = py::none();
    angles["left_hip"] = py::none();
    angles["right_hip"] = py::none();
    angles["left_knee"] = py::none();
    angles["right_knee"] = py::none();
    
    if (landmarks.is_none()) {
        return angles;
    }
    
    // MediaPipe landmark indices (matches mp_pose.PoseLandmark)
    constexpr int LEFT_SHOULDER = 11;
    constexpr int LEFT_ELBOW = 13;
    constexpr int LEFT_WRIST = 15;
    constexpr int LEFT_HIP = 23;
    constexpr int LEFT_KNEE = 25;
    constexpr int LEFT_ANKLE = 27;
    
    constexpr int RIGHT_SHOULDER = 12;
    constexpr int RIGHT_ELBOW = 14;
    constexpr int RIGHT_WRIST = 16;
    constexpr int RIGHT_HIP = 24;
    constexpr int RIGHT_KNEE = 26;
    constexpr int RIGHT_ANKLE = 28;
    
    try {
        // LEFT ELBOW (shoulder-elbow-wrist)
        if (check_visibility(landmarks, {LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST})) {
            Point2D shoulder = get_point(landmarks, LEFT_SHOULDER);
            Point2D elbow = get_point(landmarks, LEFT_ELBOW);
            Point2D wrist = get_point(landmarks, LEFT_WRIST);
            
            float vector_angle = calculate_angle_fast(shoulder, elbow, wrist);
            if (vector_angle >= 0.0f) {
                // Convert to anatomical angle (180 - vector_angle)
                angles["left_elbow"] = py::cast(180.0f - vector_angle);
            }
        }
        
        // RIGHT ELBOW (shoulder-elbow-wrist)
        if (check_visibility(landmarks, {RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST})) {
            Point2D shoulder = get_point(landmarks, RIGHT_SHOULDER);
            Point2D elbow = get_point(landmarks, RIGHT_ELBOW);
            Point2D wrist = get_point(landmarks, RIGHT_WRIST);
            
            float vector_angle = calculate_angle_fast(shoulder, elbow, wrist);
            if (vector_angle >= 0.0f) {
                angles["right_elbow"] = py::cast(180.0f - vector_angle);
            }
        }
        
        // LEFT SHOULDER (hip-shoulder-elbow)
        if (check_visibility(landmarks, {LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW})) {
            Point2D hip = get_point(landmarks, LEFT_HIP);
            Point2D shoulder = get_point(landmarks, LEFT_SHOULDER);
            Point2D elbow = get_point(landmarks, LEFT_ELBOW);
            
            float angle = calculate_angle_fast(hip, shoulder, elbow);
            if (angle >= 0.0f) {
                angles["left_shoulder"] = py::cast(angle);
            }
        }
        
        // RIGHT SHOULDER (hip-shoulder-elbow)
        if (check_visibility(landmarks, {RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW})) {
            Point2D hip = get_point(landmarks, RIGHT_HIP);
            Point2D shoulder = get_point(landmarks, RIGHT_SHOULDER);
            Point2D elbow = get_point(landmarks, RIGHT_ELBOW);
            
            float angle = calculate_angle_fast(hip, shoulder, elbow);
            if (angle >= 0.0f) {
                angles["right_shoulder"] = py::cast(angle);
            }
        }
        
        // LEFT HIP (shoulder-hip-knee)
        if (check_visibility(landmarks, {LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE})) {
            Point2D shoulder = get_point(landmarks, LEFT_SHOULDER);
            Point2D hip = get_point(landmarks, LEFT_HIP);
            Point2D knee = get_point(landmarks, LEFT_KNEE);
            
            float vector_angle = calculate_angle_fast(shoulder, hip, knee);
            if (vector_angle >= 0.0f) {
                angles["left_hip"] = py::cast(180.0f - vector_angle);
            }
        }
        
        // RIGHT HIP (shoulder-hip-knee)
        if (check_visibility(landmarks, {RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE})) {
            Point2D shoulder = get_point(landmarks, RIGHT_SHOULDER);
            Point2D hip = get_point(landmarks, RIGHT_HIP);
            Point2D knee = get_point(landmarks, RIGHT_KNEE);
            
            float vector_angle = calculate_angle_fast(shoulder, hip, knee);
            if (vector_angle >= 0.0f) {
                angles["right_hip"] = py::cast(180.0f - vector_angle);
            }
        }
        
        // LEFT KNEE (hip-knee-ankle)
        if (check_visibility(landmarks, {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE})) {
            Point2D hip = get_point(landmarks, LEFT_HIP);
            Point2D knee = get_point(landmarks, LEFT_KNEE);
            Point2D ankle = get_point(landmarks, LEFT_ANKLE);
            
            float vector_angle = calculate_angle_fast(hip, knee, ankle);
            if (vector_angle >= 0.0f) {
                angles["left_knee"] = py::cast(180.0f - vector_angle);
            }
        }
        
        // RIGHT KNEE (hip-knee-ankle)
        if (check_visibility(landmarks, {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE})) {
            Point2D hip = get_point(landmarks, RIGHT_HIP);
            Point2D knee = get_point(landmarks, RIGHT_KNEE);
            Point2D ankle = get_point(landmarks, RIGHT_ANKLE);
            
            float vector_angle = calculate_angle_fast(hip, knee, ankle);
            if (vector_angle >= 0.0f) {
                angles["right_knee"] = py::cast(180.0f - vector_angle);
            }
        }
        
    } catch (const std::exception& e) {
        // Return dict with None values on any error
    }
    
    return angles;
}

PYBIND11_MODULE(angle_calculator_cpp, m) {
    m.doc() = "Fast C++ angle calculator for MediaPipe landmarks";
    m.def("calculate_all_angles", &calculate_all_angles_cpp,
          py::arg("landmarks"),
          "Calculate all joint angles from MediaPipe landmarks");
}