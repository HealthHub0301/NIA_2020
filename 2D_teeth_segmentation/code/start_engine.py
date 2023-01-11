from flask import Flask, render_template, request
from datetime import datetime
import logging
import warnings
import os

logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import cv2
import time
import os
import zipfile
import base64
from matplotlib import pyplot as plt

dir_prefix = '/home/boneage/toothapi/code/' #fix your path

api_file_logs_path = '/home/boneage/toothapi/toothapi_file_logs/' #fix your path (save_path)

test_data_path = dir_prefix + 'mrcnn/toothseg/api_test/'
weight_path = dir_prefix + 'mrcnn/toothseg/weights/mask_rcnn_tooth_crop.h5'
yolo_data_path = dir_prefix + "mrcnn/toothseg/yolo/data/custom/test_data/"
yolo_outputdata_path = dir_prefix + "mrcnn/toothseg/yolo/output/"
    

for i, filename in enumerate(os.listdir(test_data_path)):
        os.mkdir(api_file_logs_path + filename[:-4])
        
        # Transfer PNG to YOLO's input
        os.system("cp " + test_data_path + filename + " " + yolo_data_path)

        # Start Yolo engine
        os.system("python3 " + dir_prefix + "mrcnn/toothseg/yolo/detect.py") 
        os.system("cp " + yolo_outputdata_path + filename + " " + api_file_logs_path + filename[:-4] + "/" + filename[:-4] + "_yolo.png") # yolo result copy

        # Start segmentation engine
        os.system("python3 " + dir_prefix + "mrcnn/toothseg/tooth_seg.py splash" + " --weights=" + weight_path + " --image=" + yolo_outputdata_path + filename)
        os.system("cp " + dir_prefix + "mrcnn/toothseg/outputs/tooth_output.png " + api_file_logs_path + filename[:-4] + "/" + filename[:-4] + "_mrcnn.png") # mrcnn result copy

        # Start Putting Cropped Image to make Original Size
        os.system("python3 " + dir_prefix + "mrcnn/toothseg/put_img.py " + "--name=" + filename)
        os.system("cp " + dir_prefix + "mrcnn/toothseg/outputs/tooth_output.png " + api_file_logs_path + filename[:-4] + "/" + filename[:-4] + "_result.png") # mrcnn result copy
        os.system("mv " + dir_prefix + "mrcnn/toothseg/yolo/bbox_txt/" + filename[:-4] + ".txt " + api_file_logs_path + filename[:-4] + "/" + filename[:-4] + "_bbox.txt") # bbox result move
        os.system("rm " + yolo_data_path + "*.png")