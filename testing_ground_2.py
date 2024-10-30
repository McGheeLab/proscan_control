import time
import math
import matplotlib.pyplot as plt
from modules.m9_serial import spo  # Serial communication with the stage
import numpy as np
from filterpy.kalman import KalmanFilter
import csv
import threading

RADIUS = 1000  # 5 mm in microns
DURATION = 30  # Total time for the circle (seconds)
UPDATE_INTERVAL = 0.3  # Time between updates (300 ms)
NUM_STEPS = int(DURATION / UPDATE_INTERVAL)  # Total updates
Kp = 0.001  # Proportional gain for velocity correction (tune this value)

# Limits for the correction adjustment to prevent overshooting
MAX_CORRECTION = 50  # Max correction velocity in microns/sec

def send_command(command):
    """Send a command to the ProScan III controller."""
    command = f"{command}\r\n".encode('ascii')
    spo.write(command)

def get_current_position():
    """Query the stage for its current position with error handling."""
    send_command("P")
    try:
        response = spo.readline().decode("ascii").strip()
        print(f"Received response: {response}")

        # Clean and split the response to extract x, y, z
        values = response.split(",")
        if len(values) != 3:
            raise ValueError(f"Unexpected response format: {response}")

        # Convert to integers
        x, y, z = map(lambda v: int(v.strip().replace('\r', '').strip('R')), values)
        return x, y, z

    except (ValueError, UnicodeDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None, None, None  # Return invalid values if parsing fails

def move_stage_at_velocity(vx, vy):
    """Move the stage at the specified velocity (vx, vy)."""
    command = f"VS,{vx},{vy}\r\n"
    spo.write(command.encode())
    print(f"Sent velocity command: {command.strip()}")

def initialize_kalman_filter():
    """Initialize the Kalman filter for velocity prediction."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])  # Initial state (x, y, vx, vy)
    kf.F = np.array([[1, 0, UPDATE_INTERVAL, 0],
                     [0, 1, 0, UPDATE_INTERVAL],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement function
    kf.P *= 1000.  # Covariance matrix
    kf.R = 5  # Measurement noise
    kf.Q = np.eye(4) * 0.1  # Process noise
    return kf

def calculate_velocity(base_vx, base_vy, current_x, current_y, target_x, target_y, kf):
    """Calculate the adjusted velocity based on prediction and error correction."""
    # Kernel 2: Kalman Filter Prediction
    z = np.array([current_x, current_y])  # Current position as measurement
    kf.predict()  # Predict next state
    kf.update(z)  # Update with current measurement
    predicted_vx = kf.x[2]
    predicted_vy = kf.x[3]

    # Combine the two kernels for velocity prediction
    combined_vx = 0.5 * (base_vx + predicted_vx)
    combined_vy = 0.5 * (base_vy + predicted_vy)

    # Calculate error between current and target positions
    error_x = target_x - current_x
    error_y = target_y - current_y

    # Apply proportional correction to the velocity based on the error
    correction_vx = Kp * error_x
    correction_vy = Kp * error_y

    # Clamp the correction to avoid excessive velocity adjustments
    correction_vx = max(min(correction_vx, MAX_CORRECTION), -MAX_CORRECTION)
    correction_vy = max(min(correction_vy, MAX_CORRECTION), -MAX_CORRECTION)

    # Adjust the velocity with the correction
    adjusted_vx = combined_vx + correction_vx
    adjusted_vy = combined_vy + correction_vy

    return adjusted_vx, adjusted_vy

def move_and_plot(trajectory_func, kf, plot_title):
    """Move the stage along a given trajectory function and optionally plot the results."""
    # Get the initial position
    x0, y0, _ = get_current_position()
    if x0 is None or y0 is None:
        print("Failed to retrieve initial position. Exiting.")
        return

    # Start time for trajectory calculations
    start_time = time.time()

    # Lists to store actual and ideal path positions
    actual_path_x = []
    actual_path_y = []
    ideal_path_x = []
    ideal_path_y = []

    while True:
        elapsed_time = time.time() - start_time
        target_x, target_y = trajectory_func(elapsed_time, x0, y0)
        if target_x is None and target_y is None:
            break

        # Store ideal positions
        ideal_path_x.append(target_x)
        ideal_path_y.append(target_y)

        # Get the current position of the stage
        current_x, current_y, _ = get_current_position()
        if current_x is None or current_y is None:
            print("Skipping this step due to position retrieval failure.")
            threading.Event().wait(UPDATE_INTERVAL)
            continue

        # Store actual positions
        actual_path_x.append(current_x)
        actual_path_y.append(current_y)

        # Calculate the base velocity for the trajectory function
        base_vx, base_vy = calculate_base_velocity(trajectory_func, elapsed_time, x0, y0)

        # Calculate adjusted velocity
        adjusted_vx, adjusted_vy = calculate_velocity(base_vx, base_vy, current_x, current_y, target_x, target_y, kf)

        # Send the corrected velocity to the stage
        move_stage_at_velocity(adjusted_vx, adjusted_vy)

        # Wait for the next update
        threading.Event().wait(UPDATE_INTERVAL)

    # Stop the stage after completing the trajectory
    move_stage_at_velocity(0, 0)
    print(f"{plot_title} complete.")

    # Plot the actual and ideal paths
    plt.figure(figsize=(10, 8))
    plt.plot(ideal_path_x, ideal_path_y, label='Ideal Path', linestyle='--', color='blue')
    plt.plot(actual_path_x, actual_path_y, label='Actual Path', linestyle='-', color='red')
    plt.xlabel('X Position (microns)')
    plt.ylabel('Y Position (microns)')
    plt.title(f'Stage Movement: {plot_title}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def circular_trajectory(t, x0, y0):
    """Calculate the (x, y) coordinates for time `t` in a circular path."""
    if t > DURATION:
        return None, None
    angle = (2 * math.pi * t) / DURATION
    x = int(RADIUS * math.cos(angle) + x0)
    y = int(RADIUS * math.sin(angle) + y0)
    return x, y

def waypoint_trajectory(elapsed_time, x0, y0, waypoints):
    """Calculate the target waypoint position at a given elapsed time."""
    for waypoint in waypoints:
        target_x, target_y, target_time = waypoint
        if elapsed_time <= target_time:
            return target_x + x0, target_y + y0
    return None, None

def calculate_base_velocity(trajectory_func, elapsed_time, x0, y0):
    """Calculate the base velocity for the given trajectory."""
    if trajectory_func == circular_trajectory:
        omega = (2 * math.pi) / DURATION
        base_vx = -RADIUS * omega * math.sin((2 * math.pi * elapsed_time) / DURATION)
        base_vy = RADIUS * omega * math.cos((2 * math.pi * elapsed_time) / DURATION)
    else:
        # Placeholder for waypoint trajectory velocity calculation
        base_vx = base_vy = 0  # Update as needed
    return base_vx, base_vy

def import_waypoints_from_csv(file_path):
    """Import waypoints from a CSV file."""
    waypoints = []
    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) == 3:
                    x, y, t = map(float, row)
                    waypoints.append((x, y, t))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return waypoints

if __name__ == "__main__":
    # Import waypoints from CSV
    csv_file_path = 'waypoints.csv'
    waypoints = import_waypoints_from_csv(csv_file_path)
    
    # Initialize Kalman filter
    kf = initialize_kalman_filter()
    
    if waypoints:
        move_and_plot(lambda t, x0, y0: waypoint_trajectory(t, x0, y0, waypoints), kf, "Waypoint Following")
    else:
        move_and_plot(circular_trajectory, kf, "Circular Motion")
