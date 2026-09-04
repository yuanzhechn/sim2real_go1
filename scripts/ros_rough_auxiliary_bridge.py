#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Guarded ROS1 point-cloud/odometry bridge for Rough flat-floor commissioning."""

from __future__ import print_function

import argparse
import imp
import json
import os
import socket
import sys
import threading
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
height_scan_module = imp.load_source(
    "go1_height_scan", os.path.join(ROOT, "src", "go1_sim2real", "height_scan.py")
)
fit_flat_ground_scan = height_scan_module.fit_flat_ground_scan


class RoughBridge(object):
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.velocity = None
        self.velocity_at = 0.0
        self.clouds = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rospy.Subscriber(args.odom_topic, Odometry, self.on_odom, queue_size=1)
        for topic in args.cloud_topic:
            rospy.Subscriber(topic, PointCloud2, self.on_cloud, callback_args=topic, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / args.rate), self.on_timer)

    def on_odom(self, message):
        if message.child_frame_id != self.args.child_frame:
            rospy.logerr_throttle(2.0, "reject odom child frame: %s" % message.child_frame_id)
            return
        value = message.twist.twist.linear
        with self.lock:
            self.velocity = [float(value.x), float(value.y), float(value.z)]
            self.velocity_at = time.time()

    def on_cloud(self, message, topic):
        if message.header.frame_id != self.args.cloud_frame:
            rospy.logerr_throttle(2.0, "reject cloud frame %s from %s" % (message.header.frame_id, topic))
            return
        points = np.asarray(
            list(pc2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float64,
        )
        if points.size:
            with self.lock:
                self.clouds[topic] = (points.reshape((-1, 3)), time.time())

    def on_timer(self, _event):
        now = time.time()
        with self.lock:
            velocity = None if self.velocity is None else list(self.velocity)
            velocity_age = now - self.velocity_at
            fresh = [value[0].copy() for value in self.clouds.values() if now - value[1] <= self.args.max_age]
        if velocity is None or velocity_age > self.args.max_age:
            rospy.logwarn_throttle(2.0, "rough bridge: odometry missing/stale")
            return
        if len(fresh) < self.args.min_cameras:
            rospy.logwarn_throttle(2.0, "rough bridge: only %d fresh cameras" % len(fresh))
            return
        try:
            result = fit_flat_ground_scan(
                np.concatenate(fresh, axis=0),
                height_offset_m=self.args.height_offset,
                max_rms_m=self.args.max_plane_rms,
                max_slope_rad=np.deg2rad(self.args.max_slope_deg),
                min_occupied_cells=self.args.min_ground_cells,
            )
        except ValueError as exc:
            rospy.logwarn_throttle(1.0, "rough bridge safety gate: %s" % exc)
            return
        payload = {
            "base_lin_vel": velocity,
            "height_scan": result.height_scan.tolist(),
            "height_scan_mode": "flat_plane_commissioning",
        }
        self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (self.args.host, self.args.port))
        rospy.loginfo_throttle(
            1.0,
            "rough flat-plane bridge OK: cells=%d rms=%.3fm slope=%.1fdeg scan=[%.3f,%.3f]"
            % (result.occupied_cells, result.rms_m, np.rad2deg(result.slope_rad), result.height_scan.min(), result.height_scan.max()),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/ros2udp/odom")
    parser.add_argument("--child-frame", default="base_link")
    parser.add_argument("--cloud-frame", default="trunk")
    parser.add_argument("--cloud-topic", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15001)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--max-age", type=float, default=0.20)
    parser.add_argument("--min-cameras", type=int, default=3)
    parser.add_argument("--min-ground-cells", type=int, default=45)
    parser.add_argument("--max-plane-rms", type=float, default=0.035)
    parser.add_argument("--max-slope-deg", type=float, default=15.0)
    parser.add_argument("--height-offset", type=float, default=0.5)
    args = parser.parse_args(rospy.myargv()[1:])
    if not args.cloud_topic:
        args.cloud_topic = [
            "/camera1/point_cloud_face", "/camera3/point_cloud_left",
            "/camera4/point_cloud_right", "/camera5/point_cloud_rearDown",
        ]
    rospy.init_node("go1_rough_flat_plane_bridge", anonymous=False)
    RoughBridge(args)
    rospy.logwarn("flat-plane commissioning only; do not use this bridge on rough terrain")
    rospy.spin()


if __name__ == "__main__":
    main()
