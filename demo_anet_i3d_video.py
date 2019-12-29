'''
We can use torchvision.io.read_video to read videos,
However, it lies in the latest torchvision, in updating.

vframes, _, info = torchvision.io.read_video(vid_file)
print('video frames info', vframes.shape, info)
'''

import argparse
import os
from i3d import network_init
from extract_features_from_video import run
import pickle
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb')  # rgb or flow
parser.add_argument('-weight_file', default='')
parser.add_argument('-use_zip', default=False)
parser.add_argument('-vid_names_file', default='./vids_cal.pkl')
# use stride = 4
parser.add_argument('-stride', default=4, help='feature extraction stride')
parser.add_argument('-chunk_size', help='snippet frame number for one feature extraction', default=16)
parser.add_argument('-batch_size', default=16)
parser.add_argument('-feature_dir', help='directory to save feature file',
                    default='/data/home/v-yale/I3D_feature/result/I3D_rgb')
parser.add_argument('-vid_dir', help='directory to img*.jpg',
                    default='/data/home/v-yale/w_MSM/v-yale/ActivityNet/videos_25fps')
args = parser.parse_args()


def video2array(video_file):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    datas = np.zeros((frameCount, 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount) and ret:
        ret, img = cap.read()
        if ret:
            data = cv2.resize(img, (224, 224), interpolation=cv2.INTER_BITS)
            datas[fc, :, :, :] = data
            fc += 1
    cap.release()
    return datas


if __name__ == '__main__':

    model_i3d = network_init(num_devices=1, mode='rgb', weight_file='./models/rgb_imagenet.pt')
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    vid_name_set = pickle.load(open(args.vid_names_file, 'rb'))
    vid_name_set.sort()
    vid_name_set = vid_name_set[775:840]

    for video_name in vid_name_set:
        vid_name = 'v_' + video_name

        feature_file = os.path.join(args.feature_dir, vid_name+'.npz')
        if os.path.exists(feature_file):
            print('feature calculated', vid_name)
            continue

        vid_file = os.path.join(args.vid_dir, vid_name+'.mp4')
        vframes = video2array(vid_file)
        print('video frames info', vframes.shape)
        if vframes.shape[0] < args.chunk_size:
            continue

        run(model_i3d, vid_name, vframes, mode=args.mode, sample_mode='resize', chunk_size=args.chunk_size, stride=args.stride,
            input_dir=args.vid_dir, output_dir=args.feature_dir, batch_size=args.batch_size, use_zip=args.use_zip)
