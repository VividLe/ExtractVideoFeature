import os

import torch
import math
from PIL import Image
import numpy as np


def load_frame(frame_file, resize=False):
    data = Image.open(frame_file)

    assert (data.size[1] == 256)
    assert (data.size[0] == 340)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert (data.max() <= 1.0)
    assert (data.min() >= -1.0)

    return data


def vframes_pre_process(vframes):
    # Rescale pixel values to [-1, 1]
    vframes = vframes.astype(float)
    datas = (vframes * 2 / 255) - 1
    return datas


def load_rgb_batch(vframes, frame_indices, chunk_size):
    # frame_indices: [batch, chunk_size]
    batch_data = np.zeros(frame_indices.shape + (224, 224, 3))

    # chunk_size = 8
    for i in range(frame_indices.shape[0]):
        indices_list = list(frame_indices[i, :])
        for j in indices_list:
            idx_set = j % chunk_size
            batch_data[i, idx_set, :, :, :] = vframes[j, :, :, :]

    return batch_data  # [batch, chunk_size, 224, 224, 3]


def load_flow_batch(frames_dir, flow_x_files, flow_y_files,
                    frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, 0] = load_frame(os.path.join(frames_dir,
                                                                flow_x_files[frame_indices[i][j]]), resize)

            batch_data[i, j, :, :, 1] = load_frame(os.path.join(frames_dir,
                                                                flow_y_files[frame_indices[i][j]]), resize)

    return batch_data


def run(model, video_name, vframes, chunk_size=16, stride=16, output_dir='', batch_size=1):

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224

        b_data = b_data.cuda().float()
        with torch.no_grad():
            b_features = model(b_data)

        b_features = b_features.data.cpu().numpy()[:, :]
        return b_features

    save_file = '{}.npz'.format(video_name)

    frame_cnt = vframes.shape[0]
    vframes = vframes_pre_process(vframes)

    # Cut frames
    clipped_length = math.floor((frame_cnt - chunk_size) / stride) + 1
    frame_indices = list()

    for i in range(0, frame_cnt-chunk_size, stride):
        indices = [j for j in range(i, i+chunk_size)]
        frame_indices.append(indices)
    frame_indices = np.array(frame_indices)

    chunk_num = frame_indices.shape[0]

    batch_num = int(np.ceil(chunk_num / batch_size))  # Chunks to batches
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    full_features = list()
    for batch_id in range(batch_num):
        batch_data = load_rgb_batch(vframes, frame_indices[batch_id], chunk_size)
        full_features.append(forward_batch(batch_data))

    full_features = np.concatenate(full_features, axis=0)

    np.savez(os.path.join(output_dir, save_file),
             feature=full_features,
             frame_cnt=frame_cnt,
             video_name=video_name)

    print('{} done: {} / {}, {}'.format(video_name, frame_cnt, clipped_length, full_features.shape))
