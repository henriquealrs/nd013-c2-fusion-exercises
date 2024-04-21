# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame

    # map value range to 8bit

    # focus on +/- 45° around the image center
    pass


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame: dataset_pb2.Frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    lidar_list = [l for l in frame.lasers if l.name == lidar_name]
    lidar = lidar_list[0]
    calib = [cl for cl in frame.context.laser_calibrations if cl.name == lidar_name][0]
    fov_rad = calib.beam_inclination_max - calib.beam_inclination_min

    ang_dg = (fov_rad*180/np.pi)
    print(ang_dg * 60)
    if len(lidar.ri_return1.range_image_compressed) > 0:
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    res = ang_dg / ri.shape[0]
    print(res * 60)


# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame: dataset_pb2.Frame):
    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    values = label_pb2._LABEL_TYPE.values_by_name
    num_vehicles = 0
    for l in frame.laser_labels:
        if l.type == l.TYPE_VEHICLE:
            num_vehicles += 1
    print("number of labeled vehicles in current frame = " + str(num_vehicles))
    return num_vehicles
