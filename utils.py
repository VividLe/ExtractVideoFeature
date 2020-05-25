'''
Some useful functions
'''

from PIL import Image
import zipfile
import io
import cv2


def load_zipframe(zipdata, name, resize=False):

    stream = zipdata.read(name)
    data = Image.open(io.BytesIO(stream))

    assert(data.size[1] == 256)
    assert(data.size[0] == 340)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data


def load_ziprgb_batch(rgb_zipdata, rgb_files,
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,:] = load_zipframe(rgb_zipdata,
                rgb_files[frame_indices[i][j]], resize)

    return batch_data


def load_zipflow_batch(flow_x_zipdata, flow_y_zipdata,
                    flow_x_files, flow_y_files,
                    frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_zipframe(flow_x_zipdata,
                flow_x_files[frame_indices[i][j]], resize)

            batch_data[i,j,:,:,1] = load_zipframe(flow_y_zipdata,
                flow_y_files[frame_indices[i][j]], resize)

    return batch_data




