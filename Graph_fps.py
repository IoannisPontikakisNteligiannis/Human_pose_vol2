import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class FPSVisualizer:
    """Visualizer specifically for FPS data from recorded JSON files"""
    
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = None
        self.fps_data = []
        self.frame_times = []
        self.frame_types = []
        
        # Separate background processing data
        self.bg_processing_times = []
        self.bg_timestamps = []
        
    def load_data(self):
        """Load FPS data from JSON file"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} frames from {self.json_file_path}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.json_file_path} not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file {self.json_file_path}")
            return False
    
    def extract_fps_data(self):
        """Extract FPS data from frames, separating background processing"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        self.fps_data = []
        self.frame_times = []
        self.frame_types = []
        self.bg_processing_times = []
        self.bg_timestamps = []
        
        for frame in self.data:
            fps_value = None
            timestamp = None
            frame_type = frame.get('frame_type', 'unknown')
            
            # Get FPS value
            if 'fps' in frame:
                fps_value = frame['fps']
            elif 'angles' in frame and 'fps' in frame['angles']:
                fps_value = frame['angles']['fps']
            
            # Get timestamp
            if 'timestamp' in frame:
                timestamp = frame['timestamp']
            elif 'frame_number' in frame:
                timestamp = frame['frame_number'] / 30.0
            
            # Separate background processing from main loop FPS
            if frame_type == 'background_processing':
                # Store as processing time in ms
                if 'frame_time_ms' in frame and timestamp is not None:
                    self.bg_processing_times.append(frame['frame_time_ms'])
                    self.bg_timestamps.append(timestamp)
            else:
                # Store as FPS for main loop frames
                if fps_value is not None and timestamp is not None:
                    if 0.1 <= fps_value <= 200:  # Reasonable FPS range
                        self.fps_data.append(fps_value)
                        self.frame_times.append(timestamp)
                        self.frame_types.append(frame_type)
        
        print(f"Extracted {len(self.fps_data)} FPS data points (main loop)")
        print(f"Extracted {len(self.bg_processing_times)} background processing measurements")
        
        if len(self.fps_data) == 0:
            print("ERROR: No valid FPS data found!")
            return False
        
        return True
    
    def plot_fps(self, save_plot=True):
        """Create and display FPS plots with proper background processing display"""
        if not self.fps_data:
            print("No FPS data available. Call extract_fps_data() first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FPS Performance Analysis - {Path(self.json_file_path).name}', fontsize=16)
        
        # ========== Plot 1: FPS over time (main loop only) ==========
        processing_mask = np.array(self.frame_types) == 'processing'
        skipped_mask = np.array(self.frame_types) == 'skipped'
        
        if np.any(processing_mask):
            processing_times = np.array(self.frame_times)[processing_mask]
            processing_fps = np.array(self.fps_data)[processing_mask]
            axes[0, 0].scatter(processing_times, processing_fps, 
                             c='blue', alpha=0.6, s=1, label='Processing frames')
        
        if np.any(skipped_mask):
            skipped_times = np.array(self.frame_times)[skipped_mask]
            skipped_fps = np.array(self.fps_data)[skipped_mask]
            axes[0, 0].scatter(skipped_times, skipped_fps, 
                             c='red', alpha=0.6, s=1, label='Skipped frames')
        
        # Add overall trend line
        axes[0, 0].plot(self.frame_times, self.fps_data, 'gray', alpha=0.3, linewidth=0.5)
        
        axes[0, 0].set_title('Main Loop FPS Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add moving average
        if len(self.fps_data) > 10:
            window_size = min(30, len(self.fps_data) // 10)
            moving_avg = np.convolve(self.fps_data, np.ones(window_size)/window_size, mode='valid')
            moving_avg_times = self.frame_times[window_size-1:]
            axes[0, 0].plot(moving_avg_times, moving_avg, 'darkgreen', linewidth=2, 
                           label=f'Moving Average ({window_size} frames)')
            axes[0, 0].legend()
        
        # ========== Plot 2: FPS histogram ==========
        axes[0, 1].hist(self.fps_data, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 1].set_title('Main Loop FPS Distribution')
        axes[0, 1].set_xlabel('FPS')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(self.fps_data), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(self.fps_data):.1f} FPS')
        axes[0, 1].axvline(np.median(self.fps_data), color='orange', linestyle='--', 
                          label=f'Median: {np.median(self.fps_data):.1f} FPS')
        axes[0, 1].legend()
        
        # ========== Plot 3: Frame time analysis WITH background processing ==========
        # Main loop frame times
        frame_times_ms = [1000/fps for fps in self.fps_data]
        
        axes[1, 0].plot(self.frame_times, frame_times_ms, 'purple', alpha=0.5, 
                       linewidth=1, label='Main loop frame time')
        
        # Background processing times (MediaPipe)
        if self.bg_processing_times:
            axes[1, 0].plot(self.bg_timestamps, self.bg_processing_times, 
                            'darkorange', alpha=0.7, linewidth=1, 
                           label='Background processing time (MediaPipe)')
            
            # Add average line for background processing
            avg_bg = np.mean(self.bg_processing_times)
            axes[1, 0].axhline(avg_bg, color='darkorange', linestyle='--', 
                              linewidth=2, alpha=0.8, 
                              label=f'Avg MediaPipe: {avg_bg:.1f}ms')
        
        axes[1, 0].set_title('Frame Time (ms) Over Time')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frame Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add target frame time lines
        axes[1, 0].axhline(1000/30, color='green', linestyle='--', alpha=0.5, 
                          label='30 FPS target (33.3ms)')
        
        axes[1, 0].legend(loc='upper right', fontsize=8)
        
        # ========== Plot 4: Performance metrics ==========
        axes[1, 1].axis('off')
        
        # Calculate statistics
        mean_fps = np.mean(self.fps_data)
        min_fps = np.min(self.fps_data)
        max_fps = np.max(self.fps_data)
        std_fps = np.std(self.fps_data)
        percentile_95 = np.percentile(self.fps_data, 95)
        percentile_5 = np.percentile(self.fps_data, 5)
        
        # Count frame types
        processing_count = sum(1 for ft in self.frame_types if ft == 'processing')
        skipped_count = sum(1 for ft in self.frame_types if ft == 'skipped')
        
        # Background processing stats
        bg_stats_text = ""
        if self.bg_processing_times:
            avg_bg = np.mean(self.bg_processing_times)
            min_bg = np.min(self.bg_processing_times)
            max_bg = np.max(self.bg_processing_times)
            bg_fps_equiv = 1000 / avg_bg
            bg_stats_text = f"""
Background Processing (MediaPipe):
Avg: {avg_bg:.1f}ms ({bg_fps_equiv:.1f} FPS equiv)
Min: {min_bg:.1f}ms
Max: {max_bg:.1f}ms
Samples: {len(self.bg_processing_times)}
"""
        
        stats_text = f"""
Main Loop Performance:

Mean FPS: {mean_fps:.1f}
Median FPS: {np.median(self.fps_data):.1f}
Min FPS: {min_fps:.1f}
Max FPS: {max_fps:.1f}
Std Dev: {std_fps:.1f}

Percentiles:
95th: {percentile_95:.1f} FPS
5th: {percentile_5:.1f} FPS

Frame Types:
Processing: {processing_count} ({processing_count/len(self.fps_data)*100:.1f}%)
Skipped: {skipped_count} ({skipped_count/len(self.fps_data)*100:.1f}%)
{bg_stats_text}
Total Frames: {len(self.fps_data)}
Duration: {self.frame_times[-1]:.1f}s
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',linespacing=1)
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            output_path = Path(self.json_file_path).parent / f"fps_analysis_{Path(self.json_file_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ FPS analysis plot saved to {output_path}")
        
        plt.show()
    
    def print_fps_summary(self):
        """Print detailed FPS statistics"""
        if not self.fps_data:
            print("No FPS data available. Call extract_fps_data() first.")
            return
        
        print("\n=== MAIN LOOP FPS PERFORMANCE ===")
        print(f"Total frames analyzed: {len(self.fps_data)}")
        print(f"Session duration: {self.frame_times[-1]:.1f} seconds")
        print(f"Mean FPS: {np.mean(self.fps_data):.2f}")
        print(f"Median FPS: {np.median(self.fps_data):.2f}")
        print(f"Min FPS: {np.min(self.fps_data):.2f}")
        print(f"Max FPS: {np.max(self.fps_data):.2f}")
        print(f"Standard deviation: {np.std(self.fps_data):.2f}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"  {p}th percentile: {np.percentile(self.fps_data, p):.2f} FPS")
        
        # Frame type breakdown
        processing_count = sum(1 for ft in self.frame_types if ft == 'processing')
        skipped_count = sum(1 for ft in self.frame_types if ft == 'skipped')
        print(f"\nFrame type breakdown:")
        print(f"  Processing frames: {processing_count} ({processing_count/len(self.fps_data)*100:.1f}%)")
        print(f"  Skipped frames: {skipped_count} ({skipped_count/len(self.fps_data)*100:.1f}%)")
        
        # Background processing stats
        if self.bg_processing_times:
            print(f"\n=== BACKGROUND PROCESSING (MediaPipe) ===")
            avg_bg = np.mean(self.bg_processing_times)
            print(f"Average processing time: {avg_bg:.2f}ms")
            print(f"Min processing time: {np.min(self.bg_processing_times):.2f}ms")
            print(f"Max processing time: {np.max(self.bg_processing_times):.2f}ms")
            print(f"Theoretical max FPS: {1000/avg_bg:.1f} FPS")
            print(f"Total measurements: {len(self.bg_processing_times)}")
        
        # Performance analysis
        low_fps_count = sum(1 for fps in self.fps_data if fps < 20)
        good_fps_count = sum(1 for fps in self.fps_data if fps >= 30)
        
        print(f"\nPerformance breakdown:")
        print(f"  Frames below 20 FPS: {low_fps_count} ({low_fps_count/len(self.fps_data)*100:.1f}%)")
        print(f"  Frames at 30+ FPS: {good_fps_count} ({good_fps_count/len(self.fps_data)*100:.1f}%)")


def main():
    """Main function to run the FPS visualizer"""
    
    # Look for FPS JSON files
    pose_dir = Path("exported_poses")
    
    if not pose_dir.exists():
        print(f"Error: Directory '{pose_dir}' doesn't exist!")
        return
    
    # Look for FPS JSON files
    fps_files = list(pose_dir.glob("fps_recording_*.json"))
    
    if not fps_files:
        print(f"No FPS recording JSON files found in '{pose_dir}'")
        print("Available files in directory:")
        all_files = list(pose_dir.iterdir())
        if all_files:
            for file in all_files:
                print(f"  - {file.name}")
        return
    
    # Show available files and use the most recent one
    print("Found FPS files:")
    for i, file in enumerate(fps_files):
        print(f"  {i+1}. {file.name}")
    
    # Use the most recent file
    fps_file = max(fps_files, key=lambda x: x.stat().st_mtime)
    print(f"\n✓ Using most recent file: {fps_file.name}")
    
    # Create visualizer
    visualizer = FPSVisualizer(fps_file)
    
    # Load and process data
    if visualizer.load_data():
        if visualizer.extract_fps_data():
            visualizer.print_fps_summary()
            visualizer.plot_fps(save_plot=True)
        else:
            print("Failed to extract valid FPS data - check file format")
    else:
        print("Failed to load data - check file format")


if __name__ == "__main__":
    main()