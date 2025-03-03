import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, time
import numpy as np

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()

        # Add "gel" to our list of cameras
        self.camera_names = [
            'cam_high',
            'cam_low',
            'cam_left_wrist',
            'cam_right_wrist',
            'gel',
        ]

        if init_node:
            rospy.init_node('image_recorder', anonymous=True)

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)

            # Select callback based on camera name
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
                topic_name = "/usb_cam_high/image_raw"

            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
                topic_name = "/usb_cam_low/image_raw"

            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
                topic_name = "/usb_cam_left_wrist/image_raw"

            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
                topic_name = "/usb_cam_right_wrist/image_raw"

            elif cam_name == 'gel':
                callback_func = self.image_cb_gel
                # Make sure this matches the topic used in digit.py
                topic_name = "/gel/camera/image_color"

            else:
                # If you add more names later, handle them here
                raise NotImplementedError

            # Subscribe to the correct topic with the selected callback
            rospy.Subscriber(topic_name, Image, callback_func)

            # If debug, store the last 50 timestamps for frequency diagnostics
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        # 转换颜色通道，从 BGR 转为 RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        setattr(self, f'{cam_name}_image', cv_image)
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)

        if self.is_debug:
            ts_list = getattr(self, f'{cam_name}_timestamps')
            ts_list.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_cam_high(self, data):
        return self.image_cb('cam_high', data)

    def image_cb_cam_low(self, data):
        return self.image_cb('cam_low', data)

    def image_cb_cam_left_wrist(self, data):
        return self.image_cb('cam_left_wrist', data)

    def image_cb_cam_right_wrist(self, data):
        return self.image_cb('cam_right_wrist', data)

    def image_cb_gel(self, data):
        return self.image_cb('gel', data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(list_of_timestamps):
            arr = np.array(list_of_timestamps)
            diff = arr[1:] - arr[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            ts_list = getattr(self, f'{cam_name}_timestamps', [])
            if len(ts_list) > 1:
                image_freq = 1 / dt_helper(ts_list)
                print(f'{cam_name} frequency: {image_freq:.2f} Hz')
            else:
                print(f'{cam_name} no timestamps recorded yet.')
        print()

class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_group", JointGroupCommand, self.puppet_arm_commands_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_single", JointSingleCommand, self.puppet_gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')


# Global variables for rectangle adjustment
drawing = False
rect_start = (0, 0)
rect_end = (320, 240)
angle = 0  # Rotation angle in degrees

def main():
    recorder = ImageRecorder(init_node=True)

    # Store rectangle positions
    rect_positions = {}

    # Process each camera
    for cam_name in ['cam_high', 'cam_low', 'gel', 'cam_left_wrist', 'cam_right_wrist']:
        print(f"Processing: {cam_name}")

        while not rospy.is_shutdown():
            # Get the image for the current camera
            images = recorder.get_images()
            image = images.get(cam_name)

            if image is None:
                print(f"No signal for {cam_name}. Skipping...")
                break  # Skip to the next camera if no signal

            # Resize the image to the target size
            target_size = (640, 480)
            image = cv2.resize(image, target_size)

            # Copy the image for drawing
            temp_image = image.copy()

            # Initialize rectangle parameters
            center = (320, 240)  # Center of the rectangle
            width, height = 320, 240  # Initial dimensions
            angle = 0  # Initial rotation angle
            rect_confirmed = False

            # Maximum allowed dimensions
            max_width, max_height = 640, 480

            cv2.namedWindow(f"Adjust Rectangle - {cam_name}")

            while not rect_confirmed:
                # Draw the rotated rectangle
                display_image = temp_image.copy()
                draw_rotated_rectangle(display_image, center, width, height, angle)

                # Show the image with the rectangle
                cv2.imshow(f"Adjust Rectangle - {cam_name}", display_image)

                # Wait for key input
                key = cv2.waitKey(1) & 0xFF

                # Move the rectangle with arrow keys
                if key == ord('w'):  # Move up
                    center = (center[0], max(0, center[1] - 10))
                elif key == ord('s'):  # Move down
                    center = (center[0], min(target_size[1] - 1, center[1] + 10))
                elif key == ord('a'):  # Move left
                    center = (max(0, center[0] - 10), center[1])
                elif key == ord('d'):  # Move right
                    center = (min(target_size[0] - 1, center[0] + 10), center[1])

                # Resize the rectangle (proportional scaling)
                elif key == ord('i'):  # Shrink proportionally
                    # Scale down proportionally by a factor of 0.9
                    if width > 20 and height > 20:  # Minimum size limit
                        width = int(width * 0.9)
                        height = int(height * 0.9)

                elif key == ord('k'):  # Expand proportionally
                    # Scale up proportionally by a factor of 1.1
                    new_width = int(width * 1.1)
                    new_height = int(height * 1.1)

                    # Ensure dimensions do not exceed the maximum limits
                    if new_width > max_width:
                        new_width = max_width
                    if new_height > max_height:
                        new_height = max_height

                    # Check if the new dimensions would cause the rectangle to go out of bounds
                    if center[0] - new_width // 2 < 0 or center[0] + new_width // 2 > target_size[0] or \
                            center[1] - new_height // 2 < 0 or center[1] + new_height // 2 > target_size[1]:
                        print("Cannot expand further, rectangle will go out of bounds.")
                    else:
                        width = new_width
                        height = new_height

                # Rotate the rectangle
                elif key == ord('j'):  # Rotate counterclockwise
                    angle = (angle - 5) % 360
                    # Check if the rotated rectangle is out of bounds
                    x1, y1, x2, y2 = calculate_bounding_box(center, width, height, angle)
                    if x1 < 0 or y1 < 0 or x2 > target_size[0] or y2 > target_size[1]:
                        print("Rotated rectangle exceeds image bounds, reducing size.")
                        # Reduce size proportionally until the rectangle fits within bounds
                        while (x1 < 0 or y1 < 0 or x2 > target_size[0] or y2 > target_size[
                            1]) and width > 20 and height > 20:
                            width = int(width * 0.9)
                            height = int(height * 0.9)
                            x1, y1, x2, y2 = calculate_bounding_box(center, width, height, angle)

                elif key == ord('l'):  # Rotate clockwise
                    angle = (angle + 5) % 360
                    # Check if the rotated rectangle is out of bounds
                    x1, y1, x2, y2 = calculate_bounding_box(center, width, height, angle)
                    if x1 < 0 or y1 < 0 or x2 > target_size[0] or y2 > target_size[1]:
                        print("Rotated rectangle exceeds image bounds, reducing size.")
                        # Reduce size proportionally until the rectangle fits within bounds
                        while (x1 < 0 or y1 < 0 or x2 > target_size[0] or y2 > target_size[
                            1]) and width > 20 and height > 20:
                            width = int(width * 0.9)
                            height = int(height * 0.9)
                            x1, y1, x2, y2 = calculate_bounding_box(center, width, height, angle)

                # Confirm rectangle position
                elif key == ord('\r') or key == ord('\n'):  # Enter key
                    x1, y1, x2, y2 = calculate_bounding_box(center, width, height, angle)
                    print(
                        f"Saved rectangle for {cam_name}: center={center}, width={width}, height={height}, angle={angle}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    rect_positions[cam_name] = {
                        "center": center,
                        "width": width,
                        "height": height,
                        "angle": angle,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                    rect_confirmed = True
                    cv2.destroyWindow(f"Adjust Rectangle - {cam_name}")
                    break

                # Exit program
                elif key == ord('q'):  # Quit
                    print("Exiting...")
                    cv2.destroyAllWindows()
                    return

            # Move to the next camera
            break

    # Print all rectangle positions
    print("Final rectangle positions:")
    for cam_name, rect_data in rect_positions.items():
        print(f"{cam_name}: {rect_data}")
    output_file = "rect_positions.txt"
    with open(output_file, "w") as file:  # Open the file for writing
        for cam_name, rect_data in rect_positions.items():
            print(f"{cam_name}: {rect_data}")
            file.write(f"{cam_name}: {rect_data}\n")  # Write each rectangle's data to the file
    print(f"Rectangle positions saved to {output_file}")
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


def calculate_bounding_box(center, width, height, angle):
    """
    Calculate the bounding box (x1, y1, x2, y2) for a rotated rectangle.
    """
    # Get the rectangle corners
    rect = cv2.boxPoints(((center[0], center[1]), (width, height), angle))
    rect = np.int0(rect)

    # Calculate the bounding box
    x_coords = [point[0] for point in rect]
    y_coords = [point[1] for point in rect]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)

    return x1, y1, x2, y2


def draw_rotated_rectangle(image, center, width, height, angle):
    """
    Draw a rotated rectangle on the image.
    """
    # Get the rectangle corners
    rect = cv2.boxPoints(((center[0], center[1]), (width, height), angle))
    rect = np.int0(rect)

    # Draw the rectangle
    cv2.polylines(image, [rect], isClosed=True, color=(0, 255, 0), thickness=2)


if __name__ == "__main__":
    main()
