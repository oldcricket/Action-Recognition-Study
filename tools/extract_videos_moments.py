#!/usr/bin/env python3

import os
import json
import numpy as np
#import skvideo.io
import cv2
import sys
import concurrent.futures
from shutil import copyfile
import subprocess
from tqdm import tqdm

# input
label_file = "/mnt/glusterfs/GVolStrp1/TEXTGEN/datasets/Moments_in_Time_256x256_30fps/moments_categories.txt"
train_file = "/mnt/glusterfs/GVolStrp1/TEXTGEN/datasets/Moments_in_Time_256x256_30fps/trainingSet.csv"
val_file = "/mnt/glusterfs/GVolStrp1/TEXTGEN/datasets/Moments_in_Time_256x256_30fps/validationSet.csv"
#test_file = "/home/chenrich/dataset/something2something_v2/something-something-v2-test.json"
video_folder = "/mnt/glusterfs/GVolStrp1/TEXTGEN/datasets/Moments_in_Time_256x256_30fps"

# output
train_img_folder = "/nvme0/QFAN/Moments_30fps/training_256"
val_img_folder = "/nvme0/QFAN/Moments_30fps/validation_256"
#test_img_folder = "/home/chenrich/dataset/something2something_v2/testing_256"
train_file_list = "/nvme0/QFAN/Moments_30fps/training_256.txt"
val_file_list = "/nvme0/QFAN/Moments_30fps/validation_256.txt"
#test_file_list = "/home/chenrich/dataset/something2something_v2/testing_256.txt"

def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            label = label.split(',')
            cls_id = int(label[-1])
            id_to_label[cls_id] = label[0]
            label_to_id[label[0]] = cls_id
    return id_to_label, label_to_id

id_to_label, label_to_id = load_categories(label_file)


def load_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            video_id, label_name, _, _= line.split(",")
            label_name = label_name.strip()
            videos.append([video_id, label_name])
    return videos


train_videos = load_video_list(train_file)
val_videos = load_video_list(val_file)
#test_videos = load_test_video_list(test_file)


def resize_to_short_side(h, w, short_side=360):
    newh, neww = h, w
    if h < w:
        newh = short_side
        neww = (w / h) * newh
    else:
        neww = short_side
        newh = (h / w) * neww
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww

def video_to_images_opencv(video, basedir, targetdir, short_side=256):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    filename = os.path.join(basedir, video[0] + ".webm")
    output_foldername = os.path.join(targetdir, video[0])
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        try:
            cap = cv2.VideoCapture(filename)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except:
            print("Can not get video info: {}".format(filename))
            return video[0], cls_id, 0
        newh, neww = resize_to_short_side(height, width, short_side)
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        if not cap.isOpened():
            print("video {} can not be opened.".format(filename))
            return video[0], cls_id, 0

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_filename = os.path.join(output_foldername, "{:04d}.jpg".format(i))
            if not os.path.exists(output_filename):
                frame = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(output_filename, frame)
            i += 1

        cap.release()
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, i))
        return video[0], cls_id, i


def video_to_images(video, basedir, targetdir):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    assert cls_id >= 0
    filename = os.path.join(basedir, video[0])
    video_basename = video[0].split('.')[0]
    output_foldername = os.path.join(targetdir, video_basename)
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '-q:v', '0',
                   '{}/'.format(output_foldername) + '"%05d.jpg"']
        command = ' '.join(command)
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except:
            print("fail to convert {}, {}".format(filename))
            return video[0], cls_id, 0

        # get frame num
        i = 0
        while True:
            img_name = os.path.join(output_foldername + "/{:05d}.jpg".format(i + 1))
            if os.path.exists(img_name):
                i += 1
            else:
                break

        frame_num = i
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video_basename, cls_id, frame_num


def create_train_video():
    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'training'), train_img_folder)
                   for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(train_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video():
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'validation'), val_img_folder)
                   for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(val_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


#create_train_video()
create_val_video()
#create_test_video(256)
