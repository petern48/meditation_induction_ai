import os
import sys
import argparse
import numpy as np
import torch
import tifffile
import librosa

from torch import nn
from torch.nn import functional as F
from imageio import imwrite, imsave

RED = 0
GREEN = 1
BLUE = 2

np.set_printoptions(threshold=100)
# Because imageio uses the root logger instead of warnings package...
import logging
logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--z', default=8, type=int, help='latent space width')
    parser.add_argument('--n', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--interpolation', default=10, type=int)
    parser.add_argument('--reinit', default=10, type=int, help='reinit generator every so often')
    parser.add_argument('--exp', default='.', type=str, help='output fn')
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--walk', action='store_true', help='interpolate')
    parser.add_argument('--sample', action='store_true', help='sample n images')
    parser.add_argument('--audio_file', default='', type=str, help='(optional) audio file input')
    parser.add_argument('--color_scheme', default='', type=str, help='(optional) warm or cool')

    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        # try:  # Added this
        #     for k, v in vars(args).items():
        #         setattr(self, k, v)
        # except:  # Input dictionary instead
        #     self.x_dim = args['x_dim']
        #     self.y_dim = args['y_dim']
        #     self.net = args['net']
        #     self.c_dim = args['c_dim']
        #     self.batch_size = args['batch_size']
        #     self.scale = args['scale']
        #     self.z = args['z']
        #     self.color_scheme = args['color_scheme']
        self.name = 'Generator'
        dim = self.x_dim * self.y_dim * self.batch_size
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
        
        img[img>255] = 255  # Ensure values are under 255
        return img


def coordinates(args):
    x_dim, y_dim, scale = args.x_dim, args.y_dim, args.scale
    n_points = x_dim * y_dim  # total number of points in 2D space
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5  # (256,)  for 256x256
    # ^ Evenly spaced values
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5  # (256,)
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))  # (256, 256)   -10s along the 1st and last col
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))  # (256, 256)   -10s along the 1st and last row
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    x_mat = torch.from_numpy(x_mat).float()  # ([1, 65536, 1])  Evenly distributed nums -10 to 10
    y_mat = torch.from_numpy(y_mat).float()  # ([1, 65536, 1])  Values -10 to 10  (dif order then x_mat)
    r_mat = torch.from_numpy(r_mat).float()  # ([1, 65536, 1])  Evenly distributed nums 14.1 to ? to 14.1
    return x_mat, y_mat, r_mat
    # These represent the x-coords, y-coords, and radial distances of points in 2D space


# Function is called like this: sample(args, netG, z)[0]*255
def sample(args, netG, z):
    x_vec, y_vec, r_vec = coordinates(args)
    image = netG((x_vec, y_vec, z, r_vec))
    return image


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


# z1 is a tensor as a starting point, z2 is the ending point
# n_frames = # of intermediate steps -> program inputs args.interpolation
def latent_walk(args, z1, z2, n_frames, netG):
    delta = (z2 - z1) / (n_frames + 1)
    total_frames = n_frames + 2  # plus 2 for the starting and ending points
    states = []  # holds the imgs
    for i in range(total_frames):
        z = z1 + delta * float(i)  # shape [1,8]
        if args.c_dim == 1:
            states.append(sample(args, netG, z)[0]*255)
        else:
            states.append(sample(args, netG, z)[0]*255)
    states = torch.stack(states).detach().numpy()  # concatenates elements in list to a torch tensor
    return states  # Returns multiple imageio imgs


def cppn(args):
    seed = np.random.randint(123456789)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not os.path.exists('./trials/'):
        os.makedirs('./trials/')

    subdir = args.exp
    if not os.path.exists('trials/'+subdir):
        os.makedirs('trials/'+subdir)
    else:
        while os.path.exists('trials/'+subdir):
            response = input('Exp Directory Exists, rename (y/n/overwrite):\t')
            if response == 'y':
                subdir = input('New Exp Directory Name:\t')
            elif response == 'overwrite':
                break
            else:
                sys.exit(0)
        os.makedirs('trials/'+subdir, exist_ok=True)

    if args.name_style == 'simple':
        suff = 'image'
    if args.name_style == 'params':
        suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(args.z, args.scale, args.c_dim, args.net)

    netG = init(Generator(args))
    # print (netG)
    n_images = args.n
    zs = []

    if args.audio_file:
        def feature_extraction(file_path, num_mfcc):
            x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')  # TODO: See if i can choose the sample rate
            start = 0
            end = sample_rate
            t = len(x)
            seconds = len(x) / sample_rate  # Seconds in the video
            features = np.empty(0, dtype=np.float32)
            while end <= t:
                segment = x[start: end]
                mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
                mfcc = np.reshape(mfcc, (1, args.z))
                # stft = librosa.stft(segment).flatten()  # not flattened
                # print('stft', stft.shape)
                # Xdb = librosa.amplitude_to_db(abs(X))
                # feature_segment = np.concatenate(mfcc, stft)
                # features.append(feature_segment)
                # print('feature_segment', feature_segment)
                features = np.append(features, mfcc)
                start = end
                end += sample_rate
            features = np.reshape(features, (-1, args.z))

            # random_scale = np.random.randint(999)
            # features *= random_scale

            max_val = np.amax(np.abs(features))
            features /= max_val
            features = torch.from_numpy(features)
            return features, seconds


        print('args.audio_file', args.audio_file)
        zs, seconds = feature_extraction(args.audio_file, args.z)

        n_images = len(zs)

        # from audio_loader import load_audio
        # sound, fs = load_audio(args.audio_file)
        # print(f'Sample rate is {fs} (44100 is recommended)')
        # print('length of sound array', len(sound))

    # Create z latent vector randomly
    else:
        for _ in range(n_images):
            # Create and append a tensor for each img
            # args.z (default to 8) elements in each tensor
            # each tensor has 1 row, args.z columns
            # initialize to random values in uniform distribution (same likelihood everywhere)
            z_tensor = torch.zeros(1, args.z).uniform_(-1.0, 1.0)
            zs.append(z_tensor)
            # z_tensor.shape  ([1, 8])  [1, args.z]

    if args.walk:
        k = 0
        for i in range(n_images):
            if i+1 not in range(n_images):
                images = latent_walk(args, zs[i], zs[0], args.interpolation, netG)
                break
            images = latent_walk(args, zs[i], zs[i+1], args.interpolation, netG)

            for img in images:

                save_fn = 'trials/{}/{}_{}'.format(subdir, suff, k)
                print ('saving PNG image at: {}'.format(save_fn))
                imwrite(save_fn+'.png', img)  # imageio function
                k += 1
            print ('walked {}/{}'.format(i+1, n_images))

    elif args.sample:
        zs, _ = torch.stack(zs).sort()
        for i, z in enumerate(zs):
            img = sample(args, netG, z).cpu().detach().numpy()
            if args.c_dim == 1:
                img = img[0]
            else:
                img = img[0]
            img = img * 255

            metadata = dict(seed=str(seed),
                            z_sample=str(list(z.numpy()[0])),
                            z=str(args.z), 
                            c_dim=str(args.c_dim),
                            scale=str(args.scale),
                            net=str(args.net))

            save_fn = 'trials/{}/{}_{}'.format(subdir, suff, i)
            print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
            tifffile.imsave(save_fn+'.tif',
                            img.astype('u1'),
                            metadata=metadata)
            imwrite(save_fn+'.png'.format(subdir, suff, i), img)
    else:
        print ('No action selected. Exiting...')
        print ('If this is an error, check command line arguments for ' \
                'generating images')
        sys.exit(0)


if __name__ == '__main__':
    # os.system('rm trials/*')
    # os.system('rm temp.mp4')
    # os.system('rm output.mp4')
    args = load_args()
    # output_file_name = 'output.mp4'
    cppn(args)

    # # Create video from imgs
    # fps = 6  # 7
    # os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i 'trials/*.png' -c:v libx264 -pix_fmt yuv420p -crf 23 temp.mp4")
    # # Overlay music over video
    # if args.audio_file:
    #     # os.system(f'ffmpeg -i temp.mp4 -i {args.audio_file} -c copy -map 0:v:0 -map 1:a:0 {output_file_name}')
    #     os.system(f'ffmpeg -i temp.mp4 -i {args.audio_file} -c:v copy -map 0:v -map 1:a -y {output_file_name}')
    # else:
    #     os.system(f'mv temp.mp4 {output_file_name}')
    # print(f'File created as {output_file_name}')


    # os.system('ffmpeg -r ' + str(fps) + ' -f image2 -s 64x64 -i frames/%06d.png -i ' + audiopath + ' -crf 25 -vcodec libx264 -pix_fmt yuv420p ' + vidpath)
