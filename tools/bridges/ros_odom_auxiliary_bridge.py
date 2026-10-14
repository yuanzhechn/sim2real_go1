#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""将 Go1 ROS1 机身里程计转发为 Flat policy 的 UDP 辅助状态。"""

from __future__ import print_function

import argparse
import json
import socket
import threading
import time

import rospy
from nav_msgs.msg import Odometry


class OdomBridge(object):
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.latest = None
        self.received_at = 0.0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rospy.Subscriber(args.odom_topic, Odometry, self.on_odom, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / args.rate), self.on_timer)

    def on_odom(self, message):
        if message.child_frame_id != self.args.child_frame:
            rospy.logerr_throttle(
                2.0,
                "拒绝里程计: child_frame_id=%s, 期望=%s"
                % (message.child_frame_id, self.args.child_frame),
            )
            return
        linear = message.twist.twist.linear
        with self.lock:
            self.latest = [float(linear.x), float(linear.y), float(linear.z)]
            self.received_at = time.time()

    def on_timer(self, _event):
        with self.lock:
            latest = None if self.latest is None else list(self.latest)
            age = time.time() - self.received_at
        if latest is None or age > self.args.max_age:
            rospy.logwarn_throttle(2.0, "里程计尚未收到或已经超时，不发送辅助状态")
            return
        payload = json.dumps({"base_lin_vel": latest}, separators=(",", ":")).encode("utf-8")
        self.socket.sendto(payload, (self.args.host, self.args.port))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/ros2udp/odom")
    parser.add_argument("--child-frame", default="base_link")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15001)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--max-age", type=float, default=0.10)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("go1_flat_odom_bridge", anonymous=False)
    OdomBridge(args)
    rospy.loginfo(
        "转发 %s(%s) -> udp://%s:%d" % (args.odom_topic, args.child_frame, args.host, args.port)
    )
    rospy.spin()


if __name__ == "__main__":
    main()
