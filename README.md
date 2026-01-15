# What's new

A computer vision application for real-time pose estimation, hand tracking, and exercise detection using MediaPipe and OpenCV. 
## Now with Intel RealSense depth camera support for enhanced 3D tracking capabilities.

* Real-time pose detection using MediaPipe
* Intel RealSense Support - Depth-aware tracking with 3D pose visualization
* Hand Tracking & Gesture Recognition - Dual-hand tracking with gesture controls
* Exercise recognition for bicep curls and shoulder abduction
* Bilateral arm tracking (left/right arm support)
* Joint angle calculation for 8 key body points
* Performance optimization for Raspberry Pi 4 (Frame Skipping)
* Data recording and analysis tools
* Interactive controls for live exercise switching
* FPS monitoring and performance analytics

# Controls
* "r" Start recording pose data
* "s" Stop recording and save data as json file
* "e" Toggle exercise detection on/off
* "a" arms (left ↔ right)
* "w" Switch exercises (bicep ↔ abduction)
* "t" Show statistics for current exercise
* "p" Print FPS statistic
* "f" Save FPS data to as json file

* "q" Quit application

* "d" to enable overlay
* "h" to enable/disable hands tracking
Now in the code smoothing can be enabled uncommenting

 "Uncomment below to enable temporal smoothing of angles "
                    
                    # Build temporally-averaged angles (moving mean over recent frames)
                    # avg_angles = {}
                    # for name in angle_names:
                    #     val = angles_dict.get(name)
                    #     # append new valid value if present
                    #     if val is not None:
                    #         angle_history[name].append(float(val))
                    #     # compute mean ignoring None entries
                    #     hist = [v for v in angle_history[name] if v is not None]
                    #     avg_angles[name] = float(sum(hist) / len(hist)) if len(hist) > 0 else None

                      try:
                    cached_angles_dict = angles_dict.copy() # could use avg_angles if desired
                except Exception:
# Old version of the system is available here ( Runs on Rpi4 ): https://github.com/IoannisPontikakisNteligiannis/Rpi4-Human-Analysis-System

