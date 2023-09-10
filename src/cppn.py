import os
import sys

from imageio import imwrite
import librosa
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logging.getLogger().setLevel(logging.ERROR)

RED = 0
GREEN = 1
BLUE = 2


class Generator(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        net,
        c_dim,
        batch_size,
        scale,
        z,
        color_scheme,  
    ):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.net = net
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.scale = scale
        self.z = z
        self.color_scheme = color_scheme

        self.name = 'Generator'
        self.linear_z = nn.Linear(self.z, self.net)
        self.linear_x = nn.Linear(1, self.net, bias=False)
        self.linear_y = nn.Linear(1, self.net, bias=False)
        self.linear_r = nn.Linear(1, self.net, bias=False)
        self.linear_h = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, y, z, r = inputs  # The following shapes are for 256x256
        # x.shape, y.shape, r.shape # ([1, 65536, 1]), z.shape is [8] z_dim, 
        n_points = self.x_dim * self.y_dim  # 256 * 256 = 65536
        ones = torch.ones(n_points, 1, dtype=torch.float)  # [.shape 65536,1]
        # z.shape ([1, 8])
        z_scaled = z.view(self.batch_size, 1, self.z) * ones * self.scale
        # z_scaled.shape ([1, 65336, 8])
        z_pt = self.linear_z(z_scaled.view(self.batch_size*n_points, self.z))  # torch.tensor.view change shape
        x_pt = self.linear_x(x.view(self.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(self.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(self.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = torch.tanh(U)
        x = self.linear_h(H)
        H = F.elu(x)  # Exponential Linear Unit
        H = F.softplus(self.linear_h(H))
        H = torch.tanh(self.linear_h(H))
        x = .5 * torch.sin(self.linear_out(H)) + .5
        img = x.reshape(self.batch_size, self.y_dim, self.x_dim, self.c_dim)
        # print ('G out: ', img.shape)  # [1, 256, 256, 3]
        # Scale by 255
        img *= 255
        if self.color_scheme:  # imageio is RGB
            if self.color_scheme == 'warm':
                # Reduce the green and blue values
                img[:, :, :, RED] *= 1.3
                img[:, :, :, GREEN] *= 1.0
                img[:, :, :, BLUE] *= 0.7
            elif self.color_scheme == 'cool':
                img[:, :, :, RED] *= 0.7
                img[:, :, :, GREEN] *= 1.0
                img[:, :, :, BLUE] *= 1.3
            else:
                print("Invalid Color Scheme. Exiting...")
                sys.exit(0)

        img[img > 255] = 255  # Ensure values are under 255
        return img


def coordinates(
    x_dim,
    y_dim,
    scale,
    batch_size,
):
    """ These represent the x-coords, y-coords, and radial distances of points in 2D space """
    n_points = x_dim * y_dim  # total number of points in 2D space
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5  # (256,)  for 256x256
    # ^ Evenly spaced values
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5  # (256,)
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))  # (256, 256)   -10s along the 1st and last col
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))  # (256, 256)   -10s along the 1st and last row
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    x_mat = torch.from_numpy(x_mat).float()  # ([1, 65536, 1])  Evenly distributed nums -10 to 10
    y_mat = torch.from_numpy(y_mat).float()  # ([1, 65536, 1])  Values -10 to 10  (dif order then x_mat)
    r_mat = torch.from_numpy(r_mat).float()  # ([1, 65536, 1])  Evenly distributed nums 14.1 to ? to 14.1
    return x_mat, y_mat, r_mat


def sample(
    netG,
    z,
    x_dim,
    y_dim,
    scale,
    batch_size,
):
    """ Function is called like this: sample(args, netG, z)[0]*255 """
    x_vec, y_vec, r_vec = coordinates(
        x_dim,
        y_dim,
        scale,
        batch_size,                       
    )
    image = netG((x_vec, y_vec, z, r_vec))
    return image


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


# z1 is a tensor as a starting point, z2 is the ending point
# n_frames = # of intermediate steps -> program inputs args.interpolation
def latent_walk(
    z1,
    z2,
    n_frames,
    netG,
    x_dim,
    y_dim,
    scale,
    batch_size,
):
    delta = (z2 - z1) / (n_frames + 1)
    total_frames = n_frames + 2  # plus 2 for the starting and ending points
    states = []  # holds the imgs
    for i in range(total_frames):
        z = z1 + delta * float(i)  # shape [1,8]
        states.append(
            sample(
                netG,
                z,
                x_dim,
                y_dim,
                scale,
                batch_size,
            )[0]*255
        )
    states = torch.stack(states).detach().numpy()  # concatenates elements in list to a torch tensor
    return states  # Returns multiple imageio imgs


def feature_extraction(file_path, num_mfcc, z):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    start = 0
    end = sample_rate
    t = len(x)
    seconds = round(len(x) / sample_rate)  # Seconds in the video
    features = np.empty(0, dtype=np.float32)
    while end <= t:
        segment = x[start: end]
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        mfcc = np.reshape(mfcc, (1, z))
        features = np.append(features, mfcc)
        start = end
        end += sample_rate
    features = np.reshape(features, (-1, z))

    max_val = np.amax(np.abs(features))
    features /= max_val
    features = torch.from_numpy(features)
    return features, seconds


def cppn(
    interpolation,
    c_dim,
    audio_file,
    scale,
    trials_dir,
    x_dim,
    y_dim,
    color_scheme
):
    seed = np.random.randint(16)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir)

    z = 8
    n_images = 1
    net = 32
    batch_size = 1

    suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(z, scale, c_dim, net)

    netG = init(Generator(
        x_dim,
        y_dim,
        net,
        c_dim,
        batch_size,
        scale,
        z,
        color_scheme,
    ))
    zs = []

    print('args.audio_file', audio_file)
    zs, seconds = feature_extraction(audio_file, z, z)

    n_images = len(zs)
    frames_created = 0
    for i in range(n_images):
        if i+1 not in range(n_images):
            images = latent_walk(
                zs[i],
                zs[0],
                interpolation,
                netG,
                x_dim,
                y_dim,
                scale,
                batch_size,
            )
            break
        images = latent_walk(
            zs[i],
            zs[i+1],
            interpolation,
            netG,
            x_dim,
            y_dim,
            scale,
            batch_size,
        )

        for img in images:
            # Pad with zeros to ensure picutres are in proper order
            save_fn = 'data/trials/{}/{}_{}'.format('.', suff, str(frames_created).zfill(7))
            imwrite(save_fn+'.png', img)  # imageio function
            frames_created += 1
        print('walked {}/{}'.format(i+1, n_images))

    # If inputing audio, return the number of seconds video should last
    print('TOTALFRAMES: ', frames_created)
    print('SECONDS ', seconds)
    return frames_created, seconds
