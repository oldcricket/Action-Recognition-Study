#!/usr/bin/env python3

import os
import json
import numpy as np
# import skvideo.io
import cv2
import sys
import concurrent.futures
from shutil import copyfile
import subprocess
# from tqdm import tqdm

# input
label_file = "/share/diva08/data/qfan/UCF101/classInd.txt"
train_file = "/share/diva08/data/qfan/UCF101/trainlist01.txt"
val_file = "/share/diva08/data/qfan/UCF101/testlist01.txt"
video_folder = "/share/diva08/data/qfan/UCF101/video/"

# output
train_img_folder = "/share/diva08/data/qfan/UCF101/training_256/"
val_img_folder = "/share/diva08/data/qfan/UCF101/validation_256/"
train_file_list = "/share/diva08/data/qfan/UCF101/training_256.txt"
val_file_list = "/share/diva08/data/qfan/UCF101/validation_256.txt"

def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            label_id, label = line.split()
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id

id_to_label, label_to_id = load_categories(label_file)

def load_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            video, label_id = line.split()
            label_name, vname = video.split('/')
            videos.append([vname.split('.')[0], label_name])
    return videos


def load_test_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            videos.append([line])
    return videos


train_videos = load_video_list(train_file)
#val_videos = load_video_list(val_file)

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
    filename = os.path.join(basedir, video[0] + ".avi")
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


def video_to_images(video, basedir, targetdir, short_side=256):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    filename = os.path.join(basedir, video[0] + ".avi")
    output_foldername = os.path.join(targetdir, video[0])
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        try:
            video_meta = skvideo.io.ffprobe(filename)
            height = int(video_meta['video']['@height'])
            width = int(video_meta['video']['@width'])
        except:
            print("Can not get video info: {}".format(filename))
            return video[0], cls_id, 0

        if width > height:
            scale = "scale=-1:{}".format(short_side)
        else:
            scale = "scale={}:-1".format(short_side)
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-vf', scale,
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
        return video[0], cls_id, frame_num


# def resize_videos(video, basedir, targetdir, short_side=360):

#     label_name = id_to_label[video[1]]
#     filename = os.path.join(basedir, video[0])
#     print(filename)
#     output_filename = os.path.join(targetdir, video[0])
#     if not os.path.exists(filename):
#         print("{} is not existed.".format(filename))
#         return video[0], video[1], 0
#     else:
#         try:
#             # cap = cv2.VideoCapture(filename)
#             # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             # cap.release()
#             # height = video_meta.shape[1]
#             # width = video_meta.shape[2]
#             video_meta = skvideo.io.ffprobe(filename)
#             height = int(video_meta['video']['@height'])
#             width = int(video_meta['video']['@width'])
#             frame_num = int(video_meta['video']['@nb_frames'])
#         except:
#             print("Can not get video info: {}".format(filename))
#             return video[0], video[1], 0
#         newh, neww = resize_to_short_side(height, width, short_side)
#         # print(height, width, newh, neww, short_side)
#         folder_name = os.path.dirname(output_filename)
#         if not os.path.exists(folder_name):
#             os.makedirs(folder_name)

#         # vid_basename = "{}_{}_{}.mp4".format(video_id,start_time.zfill(6),end_time.zfill(6))
#         # output_filename = os.path.join(folder_name, vid_basename)
#         # print(filename)
#         if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
#             # print("existed {}".format(output_filename))
#             return video[0], video[1], frame_num
#         else:
#             # # if os.path.exists(output_filename):
#             try:
#                 os.remove(output_filename)
#             except:
#                 pass
#             # print("removing {}".format(output_filename))
#             # return
#         # print("start {}".format(output_filename))
#         if newh == short_side and neww == short_side:
#             os.symlink(os.path.abspath(filename), os.path.abspath(output_filename))
#             # copyfile(filename, output_filename)
#         else:
#             if neww < newh:
#                 scale = "scale={}:-1".format(short_side)
#             else:
#                 scale = "scale=-1:{}".format(short_side)
#             # print(neww, newh)
#             if neww % 2 != 0 or newh % 2 != 0:
#                 command = ['ffmpeg',
#                            '-i', '"%s"' % filename,
#                            '-vf', scale,
#                            '-c:v', 'libx264', '-c:a', 'copy',
#                            '-threads', '1',
#                            '-loglevel', 'panic',
#                            '-pix_fmt', 'yuv444p',
#                            '"%s"' % output_filename]
#             else:
#                 command = ['ffmpeg',
#                            '-i', '"%s"' % filename,
#                            '-vf', scale,
#                            '-c:v', 'libx264', '-c:a', 'copy',
#                            '-threads', '1',
#                            '-loglevel', 'panic',
#                            '-pix_fmt', 'yuv420p',
#                            '"%s"' % output_filename]
#             command = ' '.join(command)
#             print(command)
#             try:
#                 output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
#                 print(output)
#             except:
#                 # assert 0, "fail to convert {}, {}".format(filename)
#                 print("fail to convert {}, {}".format(filename))
#                 return video[0], video[1], 0
#         print("Finish {}".format(filename))
#         return video[0], video[1], frame_num


def create_train_video(short_side):
    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, train_img_folder, int(short_side))
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


def create_val_video(short_side):
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, val_img_folder, int(short_side))
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

'''
def create_test_video(short_side):
    with open(test_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, test_img_folder, int(short_side))
                   for video in test_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {}".format(os.path.join(test_img_folder, video_id), frame_num), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")
'''

create_train_video(256)
create_val_video(256)
#create_test_video(256)