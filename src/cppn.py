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
        # img *= 255  # TODO: TRY removing this bc it's done in latent_walk
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
            raise Exception("Invalid Color Scheme. Exiting...")

        # img[img > 255] = 255  # Ensure values are under 255
        img[img > 1] = 1
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


def feature_extraction(audio_segment, z, sample_rate):
    """Extracts mfcc features from each second of the audio segment.
        Returns a list of these feature vectors"""
    start = 0
    end = sample_rate
    total_length = len(audio_segment)
    features = np.empty(0, dtype=np.float32)
    # Move forward one second at a time
    while end <= total_length:
        segment = audio_segment[start: end]
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=z).T, axis=0)
        mfcc = np.reshape(mfcc, (1, z))
        features = np.append(features, mfcc)
        start = end
        end += sample_rate
        # if end > total_length:
        #     print('entering')
        #     end = total_length - 1
        #     segment = audio_segment[start: end]
        #     mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=z).T, axis=0)
        #     mfcc = np.reshape(mfcc, (1, z))
        #     features = np.append(features, mfcc)
        #     break
        # finish the last bit in the next (final iteration)


    # Normalize vector by dividing them all by their max value
    max_val = np.amax(np.abs(features))
    features /= max_val
    features = torch.from_numpy(features)

    for i in range(1, len(features)):
        if i == len(features) - 1:
            magn = torch.norm(features[i] - features[i-1])
            print(f'magn difference between features_extractions: {magn}')

    features = np.reshape(features, (-1, z))
    return features


def cppn(
    interpolation,
    c_dim,
    # audio_file,
    scale,
    trials_dir,
    x_dim,
    y_dim,
    color_scheme,
    audio_segments,
    sentiments,
    seconds_in_segments,
    sample_rate,
    fps,
    total_seconds,
    pause_seconds
):
    seed = np.random.randint(16)
    print('SEED: ', seed)
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
        scale=scale,
        z=z,
        color_scheme=color_scheme,
        x_dim=x_dim,
        y_dim=y_dim,
        c_dim=c_dim,
    ))


    frames_created = 0
    n_images = len(audio_segments)

    total_frames_left = round(total_seconds * fps)
    # print('total_frames_left', total_frames_left)

    last_vec = np.empty([0])  # Work around to replace setting to None
    # for each sentence
    for i in range(n_images):
        # sentiment_scale: range [0,2] scale up if positive. Scale down if negative.
        sentiment_scale = sentiments[i] - 1
        zs = feature_extraction(audio_segments[i], z, sample_rate)
        zs_length = len(zs)
        seconds = seconds_in_segments[i]  # how long this sentence lasts in the audio
        print('the above should have ', seconds_in_segments[i], ' seconds')
        frames_per_sentence_left = round(seconds * fps)

        # for last iteration, use the exact number of frames left
        sentence_iterations_left = n_images - i
        if sentence_iterations_left == 1:
            frames_per_sentence_left = total_frames_left

        total_frames_left -= frames_per_sentence_left

        # For very first iteration, there's no last_vec to interpolate to
        end_range = zs_length - int(pause_seconds)
        if last_vec.any():
            range_list = range(-1, end_range)
        else:
            range_list = range(0, end_range)
        # for each (roughly) second
        for j in range_list:
            # Distribute frames per j iteration evenly as possible
            vec_scale = scale * sentiment_scale
            iterations_left = end_range - j
            frames_per_iter = round(frames_per_sentence_left / iterations_left)

            frames_per_sentence_left -= frames_per_iter

            # Interpolate between last_vec and 1st vec of this sentence
            if j == -1:
                z1 = last_vec
                z2 = zs[0]
                # vec_scale = scale  # don't sentiment scale when internpolating these
            # if last j in iteration
            # elif j+1 not in range_list:  # TODO: DELETE this??
            #     z1 = zs[j]
            #     z2 = zs[0]
            else:
                z1 = zs[j]
                z2 = zs[j+1]
                magnitude_diff = torch.norm(z2 - z1)
                print('magn difference', magnitude_diff)

            images = latent_walk(
                netG=netG,
                z1=z1,
                z2=z2,
                # n_frames=interpolation,
                n_frames=frames_per_iter - 2,  # latent_walk adds two frames
                scale=vec_scale,
                batch_size=batch_size,
                x_dim=x_dim,
                y_dim=y_dim,
            )

            if j+1 not in range_list:
                last_vec = z2
                # print('setting last_vec', last_vec)
            for img in images:
                # Pad with zeros to ensure pictures are in proper order
                save_fn = f'{trials_dir}/./{suff}_{str(frames_created).zfill(7)}'
                imwrite(save_fn+'.png', img)  # imageio function
                frames_created += 1
        print('walked {}/{}'.format(i+1, n_images))


    return frames_created
