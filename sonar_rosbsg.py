#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import holoocean
from threading import Lock, Thread
from pynput import keyboard
import matplotlib.pyplot as plt


#### GET SONAR CONFIG
scenario = "Dam-HoveringCamera"
config = holoocean.packagemanager.get_scenario(scenario)
config = config['agents'][0]['sensors'][-1]["configuration"]
azi = config['Azimuth']
minR = config['RangeMin']
maxR = config['RangeMax']
binsR = config['RangeBins']
binsA = config['AzimuthBins']

#### GET PLOT READY
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi/2)
ax.set_thetamax(azi/2)

theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()


class HolooceanRosbag(Node):
    def __init__(self):
        super().__init__("rosbag")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        self.camera_img_pub = self.create_publisher(Image, "/camera_img", qos_profile)
        self.imu_data = self.create_publisher(Imu,"/imu_data",qos_profile)
        self.sonar_img_pub = self.create_publisher(Image, "/sonar_img", qos_profile)

        self.pressed_keys = list()
        self.pressed_keys_lock = Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.force = 25
        self.bridge = CvBridge()
        self.env = holoocean.make("Dam-HoveringCamera")
        self.running = True

        self.imu_msg = Imu()
        self.imu_msg.header = Header()
        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_msg.header.frame_id = "base_link"
 
        self.holoocean_thread = Thread(target=self.run_holoocean)
        self.holoocean_thread.start()

    def on_press(self, key):
        if hasattr(key, 'char'):
            with self.pressed_keys_lock:
                self.pressed_keys.append(key.char)
                self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            with self.pressed_keys_lock:
                if key.char in self.pressed_keys:
                    self.pressed_keys.remove(key.char)

    def parse_keys(self, keys, val):
        command = np.zeros(8)
        if 'i' in keys:
            command[0:4] += val
        if 'k' in keys:
            command[0:4] -= val
        if 'j' in keys:
            command[[4, 7]] += val
            command[[5, 6]] -= val
        if 'l' in keys:
            command[[4, 7]] -= val
            command[[5, 6]] += val

        if 'w' in keys:
            command[4:8] += val
        if 's' in keys:
            command[4:8] -= val
        if 'a' in keys:
            command[[4, 6]] += val
            command[[5, 7]] -= val
        if 'd' in keys:
            command[[4, 6]] -= val
            command[[5, 7]] += val

        return command

    def run_holoocean(self):
        try:
            with self.env as env:
                while self.running:
                    with self.pressed_keys_lock:
                        if 'q' in self.pressed_keys:
                            self.running = False
                            break
                        command = self.parse_keys(self.pressed_keys, self.force)

                    state = env.tick()
                    env.act("auv0", command)

                    if "LeftCamera" in state:
                        pixels = state["LeftCamera"]
                        self.camera_img_pub.publish(self.bridge.cv2_to_imgmsg(pixels[:, :, 0:3], "bgr8"))

                    if "IMUSensor" in state:
                        imu_data  = state["IMUSensor"]
                        self.imu_msg.linear_acceleration.x = float(imu_data[0][0])
                        self.imu_msg.linear_acceleration.y = float(imu_data[0][1])
                        self.imu_msg.linear_acceleration.z = float(imu_data[0][2])

                        self.imu_msg.angular_velocity.x = float(imu_data[1][0])
                        self.imu_msg.angular_velocity.y = float(imu_data[1][1])
                        self.imu_msg.angular_velocity.z = float(imu_data[1][2])
                        self.imu_data.publish(self.imu_msg)

                    if "ImagingSonar" in state:
                        s = state['ImagingSonar']

                        plot.set_array(s.ravel())
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                                    
        finally:
            plt.ioff()
            plt.show()
            cv2.destroyAllWindows()


    def destroy_node(self):
        self.running = False
        self.listener.stop()
        self.holoocean_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HolooceanRosbag()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()