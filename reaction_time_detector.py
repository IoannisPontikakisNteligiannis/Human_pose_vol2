import time
from typing import Optional, Dict
import random
import json
from datetime import datetime

# Try to import RPi.GPIO, but don't fail if not available
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False


class ReactionTimeDetector:
    """
    Simple reaction time detector - measures time from LED ON to first movement
    Works on Raspberry Pi with RGB LED or software-only mode.
    """

    def __init__(self, red_pin=17, green_pin=27, blue_pin=22):
        """Initialize detector and setup GPIO for RGB LED (if available)"""
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.connected = False
        self.using_gpio = False

        # Test state
        self.testing = False
        self.waiting_for_led = False
        self.led_on_time = None
        self.baseline_angle = None
        self.movement_threshold = 15  # degrees of change to detect movement

        # Results
        self.last_reaction_time = None
        self.last_angle_change = None
        self.all_results = []

        # Try to setup GPIO if available
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                for pin in [self.red_pin, self.green_pin, self.blue_pin]:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                self.connected = True
                self.using_gpio = True
                print(
                    f"RGB LED initialized on pins R={self.red_pin}, G={self.green_pin}, B={self.blue_pin}")
            except Exception as e:
                print(f"GPIO setup failed: {e}")
                print("Running in software-only mode")
        else:
            print("RPi.GPIO not available - running in software-only mode")
            print("Visual cues will be shown in terminal")

    # LED Control Helpers
    def _led_off(self):
        """Turn off all colors"""
        if self.using_gpio:
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.green_pin, GPIO.LOW)
            GPIO.output(self.blue_pin, GPIO.LOW)

    def _led_blue(self):
        """Set LED to blue (waiting)"""
        if self.using_gpio:
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.green_pin, GPIO.LOW)
            GPIO.output(self.blue_pin, GPIO.HIGH)
        else:
            print("WAITING...")

    def _led_green(self):
        """Set LED to green (go)"""
        if self.using_gpio:
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.blue_pin, GPIO.LOW)
            GPIO.output(self.green_pin, GPIO.HIGH)
        else:
            print("\n" + "="*50)
            print("GO! MOVE YOUR ARM NOW!")
            print("="*50)

    def start_test(self):
        """Start a new reaction time test"""
        print("\n=== Starting Test ===")

        # Random delay before LED turns on (2-10 seconds)
        self.waiting_for_led = True
        delay = random.uniform(2, 10)

        if self.using_gpio:
            print(f"Get ready... Watch for the LED! (~{delay:.1f} seconds)")
            self._led_blue()  # Blue while waiting
        else:
            print(f"Get ready... Watch for GREEN signal! (~{delay:.1f} seconds)")
            self._led_blue()

        # Schedule LED to turn on after delay
        self.led_delay = delay
        self.test_start_time = time.perf_counter()

        self.baseline_angle = None
        self.last_reaction_time = None
        self.last_angle_change = None

    def update(self, angles_dict: Dict[str, Optional[float]]) -> Dict:
        """
        Check if movement detected - call this every frame
        """
        result = {
            'testing': self.testing,
            'waiting': self.waiting_for_led,
            'reaction_time_ms': self.last_reaction_time,
            'angle_change': None
        }

        # Check if it's time to turn on LED/signal
        if self.waiting_for_led:
            elapsed = time.perf_counter() - self.test_start_time
            if elapsed >= self.led_delay:
                self._led_green()  # GO signal
                self.waiting_for_led = False
                self.testing = True
                self.led_on_time = time.perf_counter()

                # Capture baseline RIGHT NOW at green light
                elbow_angle = angles_dict.get('right_elbow')
                if elbow_angle is not None:
                    self.baseline_angle = elbow_angle
                    print(f"Baseline captured: {self.baseline_angle:.1f} degrees")
                else:
                    print("WARNING: No pose detected - test cancelled")
                    self.testing = False
                    self._led_off()
                return result  # Exit after capturing baseline

        # Only check during active test
        if not self.testing:
            return result

        # Get right elbow angle
        elbow_angle = angles_dict.get('right_elbow')
        if elbow_angle is None:
            return result

        # Check if arm moved enough
        angle_change = abs(elbow_angle - self.baseline_angle)
        if angle_change >= self.movement_threshold:
            # Movement detected
            movement_time = time.perf_counter()
            reaction_ms = (movement_time - self.led_on_time) * 1000

            self.last_reaction_time = round(reaction_ms, 2)
            self.last_angle_change = round(angle_change, 1)
            self.all_results.append(self.last_reaction_time)
            self.testing = False

            # Turn off LED
            self._led_off()

            result['reaction_time_ms'] = self.last_reaction_time
            result['angle_change'] = self.last_angle_change
            result['testing'] = False

        return result

    def get_stats(self):
        """Get statistics from all tests"""
        if not self.all_results:
            return None
        return {
            'count': len(self.all_results),
            'average': round(sum(self.all_results) / len(self.all_results), 2),
            'fastest': min(self.all_results),
            'slowest': max(self.all_results)
        }

    def print_stats(self):
        """Print all statistics"""
        stats = self.get_stats()
        if not stats:
            print("No tests completed yet!")
            return

        print("\n" + "="*40)
        print("REACTION TIME STATISTICS")
        print("="*40)
        print(f"Tests completed: {stats['count']}")
        print(f"Average:  {stats['average']} ms")
        print(f"Fastest:  {stats['fastest']} ms")
        print(f"Slowest:  {stats['slowest']} ms")
        print("="*40 + "\n")

    def save_stats_json(self, filename=None):
        """Save all test results and statistics to JSON file"""

        if not self.all_results:
            print("No results to save!")
            return None

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reaction_time_{timestamp}.json"

        stats = self.get_stats()

        data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'all_results_ms': self.all_results,
            'settings': {
                'movement_threshold_degrees': self.movement_threshold,
                'gpio_enabled': self.using_gpio
            }
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Results saved to {filename}")
            return filename
        except Exception as e:
            print(f"Failed to save results: {e}")
            return None

    def close(self):
        """Cleanup GPIO if used"""
        if self.using_gpio:
            self._led_off()
            GPIO.cleanup()
            print("GPIO cleaned up")
        else:
            print("Session ended")