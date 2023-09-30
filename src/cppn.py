import os
import sys

from imageio import imwrite
import librosa
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.color_scheme import apply_color_scheme

logging.getLogger().setLevel(logging.ERROR)


class Generator(nn.Module):
    def __init__(
        self,
        net,
        batch_size,
        scale,
        z,
        color_scheme,
        x_dim,
        y_dim,
        c_dim,
    ):
        super(Generator, self).__init__()        

        self.net = net        
        self.batch_size = batch_size
        self.color_scheme = color_scheme

        # Latent space width
        self.z = z

        # Multiplier on z
        self.scale = scale
        
        # Output image [x_dim: width, y_dim: height, c_dim: channels]
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim

        self.linear_z = nn.Linear(self.z, self.net)
        self.linear_x = nn.Linear(1, self.net, bias=False)
        self.linear_y = nn.Linear(1, self.net, bias=False)
        self.linear_r = nn.Linear(1, self.net, bias=False)
        self.linear_h = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, y, z, r = inputs
        n_points = self.x_dim * self.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)

        z_scaled = z.view(self.batch_size, 1, self.z) * ones * self.scale
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

        # Scaled by 255
        img *= 255

        if self.color_scheme:
            img = apply_color_scheme(img, self.color_scheme)

        # Ensure values are under 255
        img[img > 255] = 255

        return img


def coordinates(
    batch_size,
    scale,
    x_dim,
    y_dim,
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
    scale,
    batch_size,
    z,
    x_dim,
    y_dim,
):
    """ Function is called like this: sample(args, netG, z)[0]*255 """
    x_vec, y_vec, r_vec = coordinates(
        batch_size=batch_size,
        scale=scale,
        x_dim=x_dim,
        y_dim=y_dim,                      
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
    netG,
    z1,
    z2,
    n_frames,
    scale,
    batch_size,
    x_dim,
    y_dim,
):
    delta = (z2 - z1) / (n_frames + 1)
    total_frames = n_frames + 2  # plus 2 for the starting and ending points

    states = []
    for i in range(total_frames):
        z = z1 + delta * float(i)  # shape [1,8]
        states.append(
            sample(
                netG=netG,
                scale=scale,
                batch_size=batch_size,
                z=z,
                x_dim=x_dim,
                y_dim=y_dim,
            )[0]*255
        )
    states = torch.stack(states).detach().numpy()  # concatenates elements in list to a torch tensor
    return states  # Returns multiple imageio imgs


# Old feature extraction function
# def feature_extraction(file_path, num_mfcc, z):
#     x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
#     start = 0
#     end = sample_rate
#     t = len(x)
#     seconds = round(len(x) / sample_rate)  # Seconds in the video
#     features = np.empty(0, dtype=np.float32)
#     while end <= t:
#         segment = x[start: end]
#         mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
#         mfcc = np.reshape(mfcc, (1, z))
#         features = np.append(features, mfcc)
#         start = end
#         end += sample_rate  # move forward by one second
#     features = np.reshape(features, (-1, z))

#     # Normalize vector by dividing them all by their max value
#     max_val = np.amax(np.abs(features))
#     features /= max_val
#     features = torch.from_numpy(features)
#     return features, seconds


def feature_extraction(audio_segment, z, sample_rate):
    """Extracts mfcc features from each second of the audio segment.
    Returns a list of these feature vectors"""
    start = 0
    end = sample_rate
    t = len(audio_segment)
    features = np.empty(0, dtype=np.float32)
    while end <= t:
        segment = audio_segment[start: end]
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=z).T, axis=0)
        mfcc = np.reshape(mfcc, (1, z))
        features = np.append(features, mfcc)
        start = end
        end += sample_rate  # move forward by one second

    # Normalize vector by dividing them all by their max value
    max_val = np.amax(np.abs(features))
    features /= max_val
    features = torch.from_numpy(features)

    features = np.reshape(features, (-1, z))
    return features


def cppn(
    interpolation,
    c_dim,
    audio_file,
    scale,
    trials_dir,
    x_dim,
    y_dim,
    color_scheme,
    audio_segments,
    sentiments,
    seconds_in_segments,
    sample_rate,
    fps
):
    # seed = np.random.randint(16)
    seed = 16  # make reproducable
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
        net=net,
        batch_size=batch_size,
        scale=scale,  # TODO: Try scaling sentiment here too
        z=z,
        color_scheme=color_scheme,
        x_dim=x_dim,
        y_dim=y_dim,
        c_dim=c_dim,
    ))

    print('args.audio_file: ', audio_file)

    frames_created = 0
    n_images = len(audio_segments)

    audio_sentence_vecs = []
    # for each sentence extract mfcc's for each second
    for i in range(n_images):
        zs = feature_extraction(audio_segments[i], z, sample_rate)
        audio_sentence_vecs.append(zs)
    # for each sentence
    for i in range(n_images):
        zs = audio_sentence_vecs[i]

        zs_length = len(zs)
        sentiment_scale = sentiments[i] - 1  # Range [0,2] scale up if positive. Scale down if negative.
        num_frames = fps * seconds_in_segments[i]  # currently a float
        frames_per_iter = round(num_frames / zs_length)

        # print('Creating imgs for audio_segment', i)
        # print(seconds, 'seconds')
        # print(num_frames, 'num_frames')
        last_vec = None
        # TODO: Interpolate between last vec and this first vec

        # For each second
        for j in range(-1, zs_length - 1):  # Spend the last two seconds interpolating
            z1 = zs[j]
            n_frames = frames_per_iter
            vec_scale = scale * sentiment_scale
            # Interpolate between the last_sentence vec and 1st vec of this sentence
            if j == -1:
                # Skip last_vec if this is the very first
                if last_vec == None:
                    continue
                n_frames *= 2
                vec_scale = scale  # don't scale by sentiment
                z1 = last_vec
                z2 = zs[0]

            elif j+1 not in range(zs_length):
                z2 = zs[0]
            else:
                z2 = zs[j+1]

            images = latent_walk(
                netG=netG,
                z1=z1,
                z2=z2,
                # n_frames=interpolation,
                n_frames=n_frames - 2, # latent_walk adds 2 frames for beginning and end
                scale=vec_scale,
                batch_size=batch_size,
                x_dim=x_dim,
                y_dim=y_dim,
            )

            if j+1 not in range(zs_length):
                last_vec = images[j]
                break  # Unecessary?

            for img in images:
                # Pad with zeros to ensure picutres are in proper order
                save_fn = f'{trials_dir}/./{suff}_{str(frames_created).zfill(7)}'
                imwrite(save_fn+'.png', img)  # imageio function
                frames_created += 1
        print('walked {}/{}'.format(i+1, n_images))


    return frames_created
