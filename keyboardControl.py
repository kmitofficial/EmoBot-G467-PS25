from dronekit import connect, VehicleMode, mavutil
import time

print("[INFO] Connecting to Mission Planner SITL...")
vehicle = connect('udp:127.0.0.1:14551', wait_ready=True)

# Wait until the vehicle is ready
while not vehicle.is_armable:
    print("Waiting for vehicle to initialize...")
    time.sleep(1)

# Arm and set GUIDED mode
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
while not vehicle.armed:
    print("Arming...")
    time.sleep(1)

print("[INFO] Armed and ready. Use W (forward), S (backward), Q (quit).")

def set_throttle(throttle_value):
    # Channel 3 controls throttle for Rover
    vehicle.channels.overrides = {'1': 1500, '3': throttle_value}
def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)

SOUTH=10
UP=0  
# duration

try:
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == 'w':
            print("Forward")
            send_ned_velocity(SOUTH,0,UP)
            
        elif cmd == 's':
            print("Backward")
            # set_throttle(1400)  # Backward
            send_ned_velocity(-SOUTH,0,UP)
        elif cmd == 'q':
            print("Stopping and exiting")
            set_throttle(1500)
            break
        else:
            print("Invalid command: use W/S/Q")
finally:
    vehicle.channels.overrides = {}
    vehicle.close()