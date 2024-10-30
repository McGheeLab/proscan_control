import time
import math
import matplotlib.pyplot as plt
from modules.m9_serial import spo  # Serial communication with the stage

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

def circular_trajectory(t, total_duration, radius):
    """Calculate the (x, y) coordinates for time `t` in a circular path."""
    angle = (2 * math.pi * t) / total_duration
    x = int(radius * math.cos(angle))
    y = int(radius * math.sin(angle))
    return x, y

def move_in_circle():
    """Move the stage in a 5mm circle over 10 seconds using constant velocity commands with error correction."""
    print("Starting circular motion...")
    
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

    for step in range(NUM_STEPS):
        elapsed_time = time.time() - start_time
        if elapsed_time > DURATION:
            break

        # Calculate the desired target position on the circle
        target_x = RADIUS * math.cos((2 * math.pi / DURATION) * elapsed_time) + x0
        target_y = RADIUS * math.sin((2 * math.pi / DURATION) * elapsed_time) + y0

        # Store ideal positions
        ideal_path_x.append(target_x)
        ideal_path_y.append(target_y)

        # Get the current position of the stage
        current_x, current_y, _ = get_current_position()
        if current_x is None or current_y is None:
            print("Skipping this step due to position retrieval failure.")
            time.sleep(UPDATE_INTERVAL)
            continue

        # Store actual positions
        actual_path_x.append(current_x)
        actual_path_y.append(current_y)

        # Calculate the base velocity for circular motion
        omega = (2 * math.pi) / DURATION
        base_vx = -RADIUS * omega * math.sin((2 * math.pi * elapsed_time) / DURATION)
        base_vy = RADIUS * omega * math.cos((2 * math.pi * elapsed_time) / DURATION)

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
        adjusted_vx = base_vx + correction_vx
        adjusted_vy = base_vy + correction_vy

        # Send the corrected velocity to the stage
        move_stage_at_velocity(int(adjusted_vx), int(adjusted_vy))

        # Wait for the next update
        time.sleep(UPDATE_INTERVAL)

    # Stop the stage after completing the circle
    move_stage_at_velocity(0, 0)
    print("Circular motion complete.")

    # Plot the actual and ideal paths
    plt.figure(figsize=(10, 8))
    plt.plot(ideal_path_x, ideal_path_y, label='Ideal Path', linestyle='--', color='blue')
    plt.plot(actual_path_x, actual_path_y, label='Actual Path', linestyle='-', color='red')
    plt.xlabel('X Position (microns)')
    plt.ylabel('Y Position (microns)')
    plt.title('Stage Movement: Ideal vs Actual Path')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    move_in_circle()
