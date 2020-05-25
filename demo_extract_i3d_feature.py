'''
We can use torchvision.io.read_video to read videos,
However, it lies in the latest torchvision, in updating.

vframes, _, info = torchvision.io.read_video(vid_file)
print('video frames info', vframes.shape, info)
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from i3d import network_init
from extract_features_from_video import run
import numpy as np
import cv2


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid_dir', help='directory to video file, e.g., *.mp4', default='/disk2/yangle/ExtractVideoFeature/videos_1')
    parser.add_argument('-feature_dir', help='directory to save feature file', default='/disk2/yangle/ExtractVideoFeature/features_1')
    parser.add_argument('-stride', default=1, help='feature extraction stride')
    parser.add_argument('-chunk_size', help='snippet frame number for one feature extraction', default=16)
    parser.add_argument('-batch_size', default=8)
    parser.add_argument('-mode', default='rgb', help='rgb or flow')
    parser.add_argument('-weight_file', default='/disk2/yangle/ExtractVideoFeature/models/rgb_imagenet.pt')

    args = parser.parse_args()
    return args


def video2array(video_file):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Notice: we should perform resize when load frames
    datas = np.zeros((frameCount, 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount) and ret:
        ret, img = cap.read()
        if ret:
            # data = cv2.resize(img, (224, 224), interpolation=cv2.INTER_BITS)
            data = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            datas[fc, :, :, :] = data
            fc += 1
    cap.release()
    return datas


def extract_feature(args):
    model_i3d = network_init(num_devices=1, mode=args.mode, weight_file=args.weight_file)
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    vid_name_set = os.listdir(args.vid_dir)
    vid_name_set.sort(reverse=False)

    for video_name in vid_name_set:
        vid_name = video_name[:-4]

        feature_file = os.path.join(args.feature_dir, vid_name + '.npz')
        if os.path.exists(feature_file):
            print('Video %s already calculated, skip' % video_name)
            continue

        vid_file = os.path.join(args.vid_dir, vid_name + '.mp4')
        vframes = video2array(vid_file)
        print('video frames info', vframes.shape)
        if vframes.shape[0] < args.chunk_size:
            print('video %s contains %d frames, too short, skip' % (video_name, vframes.shape[0]))
            continue

        run(model_i3d, vid_name, vframes, chunk_size=args.chunk_size,
            stride=args.stride,
            output_dir=args.feature_dir, batch_size=args.batch_size)


if __name__ == '__main__':
    args = args_parser()
    extract_feature(args)
