import sys
import os, glob
import argparse
from text_generation import text_generation
from text_to_audio import text_to_speech, overlay_music_and_speech
# from video_generation import 


DEFAULT_MUSIC = 'music-only1.mp3'

def load_args():
    parser = argparse.ArgumentParser(description='meditation_induction')
    parser.add_argument('--med_type', type=str, help="""type of meditation:
                        '[focused]  [body-scan]  [visualization]  [reflection]  [movement]""")
    parser.add_argument('--script_file', type=str, default='', help='input script to skip text generation')
    parser.add_argument('--accent', type=str, default='co.in', help='[com.au] [co.uk] [us] [ca] [co.in] [ie] [co.za]')
    parser.add_argument('--music_file', default=DEFAULT_MUSIC, type=str, help='background music')
    parser.add_argument('--channels', type=int, default=3, help='3 for RGB, 1 for black/white')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--out_file', default='med_video.mp4', help='output file name')
    parser.add_argument('--color_scheme', default='', type=str, help='(optional) warm or cool')
    parser.add_argument('--no_music', action='store_true', help='(optional) skip background music')

    args = parser.parse_args()
    return args


def main():
    # Load Arguments and check if valid
    args = load_args()

    if args.channels != 1 and args.channels != 3:
        print('Invalid number of channels. Must be (1) for black/white or (3) for RGB')

    # Text generation
    if args.script_file:
        with open(args.script_file, 'r') as f:
            script = f.read()

    else:
        meditation_types = ['focused', 'body-scan', 'visualization', 'reflection', 'movement']
        if args.med_type not in meditation_types:
            print('Invalid input. Provide valid med_type or input own script. Exitting...')
            sys.exit()
        script = text_generation(args.med_type)

        script_file_name = f'{args.med_type}-meditation-script1.txt'
        try:
            with open(script_file_name, 'w') as f:
                f.write(script)
            print(f'Storing meditation script in {script_file_name}')

        except:
            print(f"Couldn't store script in file {script_file_name}. Continuing...")

    # Text To Audio
    audio_file_name = text_to_speech(script, args.accent)

    if args.no_music:
        print('no_music selected ... skipping background music')

    elif args.music_file and os.path.isfile(args.music_file):
        print(f'Adding background music from {args.music_file}')
        audio_file_name = overlay_music_and_speech(audio_file_name, args.music_file)
        print(f'Created {audio_file_name} with background music')


    # Video Generation

    # NOTE: Ideally, a different video generation model could be called for the scripts that
    # involves visualizing realistic images while the CPPN only covers the other ones.

    assert(os.path.isfile('cppn.py'))
    # Call CPPN given audio input

    inter = 25
    scale = 10
    temp_file = 'temp-file.mp4'
    imgs_dir = 'trials'

    # Ensure trials directory is empty
    print(f'Deleting any files in {imgs_dir}/')
    files = glob.glob('trials/*')
    for f in files:
        os.remove(f)
    # os.system('rm trials/*')  # 2>/dev/null


    # Input audio into cppn to create imgs
    cmd = f'echo overwrite | python cppn.py --walk --x_dim {args.x_dim} --y_dim {args.y_dim} \
              --c_dim {args.channels} --interpolation {inter} --audio_file {audio_file_name} \
              --scale {scale}'

    if args.color_scheme:
        cmd += f' --color_scheme {args.color_scheme}'
    cmd += ' > /dev/null'
    print('Creating imgs using cppn')
    os.system(cmd)
    
    # Compile imgs into video
    print('\nCompiling imgs into video')
    os.system(f"ffmpeg -framerate 7 -pattern_type glob -i '{imgs_dir}/*.png' -pix_fmt yuv420p \
              -c:v libx264 -crf 23 {temp_file}")

    # Add audio to video
    print('\nAdding audio to video')
    os.system(f'ffmpeg -i {temp_file} -i {audio_file_name} -c:v copy -map 0:v -map 1:a \
              -y {args.out_file}')  #  > /dev/null 2> /dev/null

    os.system(f'rm {temp_file}')

    print(f'\nSuccessfully created video as {args.out_file}')

    # TODO: Test for best interpolation value. Maybe add --scale

if __name__ == '__main__':
    main()