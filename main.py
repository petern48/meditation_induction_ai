import argparse
import glob
import os

from src import cppn
from src.text_generation import text_generation
from src.text_to_audio import text_to_speech, overlay_music_and_speech


def load_args():
    parser = argparse.ArgumentParser(description='meditation_induction')
    # Non-default arguments
    parser.add_argument('--med_type', type=str,
                        choices=['focused', 'body-scan', 'visualization', 'reflection', 'movement'],
                        help="type of meditation: '[focused] [body-scan] [visualization] [reflection] [movement]")
    # Default arguments
    parser.add_argument('--script_file', type=str, default='',
                        help='input script to skip text generation, hence, there is no script generation')
    parser.add_argument('--accent', type=str, default='co.in',
                        help='[com.au] [co.uk] [us] [ca] [co.in] [ie] [co.za]')
    parser.add_argument('--music_file', default='assets/default_background_music.mp3', type=str,
                        help='background music')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3],
                        help='3 for RGB, 1 for black/white')
    parser.add_argument('--x_dim', default=256, type=int,
                        help='out image width')
    parser.add_argument('--y_dim', default=256, type=int,
                        help='out image height')
    parser.add_argument('--color_scheme', default='warm', type=str, choices=['warm', 'cool'],
                        help='(optional) warm or cool')
    parser.add_argument('--no_music', action='store_true',
                        help='(optional) skip background music')
    # Skip
    parser.add_argument('--skip_cppn_generation', action='store_true',
                        help='skip cppn generation')
    parser.add_argument('--skip_background_music', action='store_true',
                        help='skip background music')

    args = parser.parse_args()
    return args


def main():
    # Load Arguments and check if valid
    args = load_args()
    
    # Creating data and output folders if not exist
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('output'):
        os.makedirs('output')

    if args.channels != 1 and args.channels != 3:
        print('Invalid number of channels. Must be (1) for black/white or (3) for RGB')

    if args.script_file:
        with open(args.script_file, 'r') as f:
            script = f.read()
        print('script_file provided, skipping text generation')
    else:
        meditation_types = ['focused', 'body-scan', 'visualization', 'reflection', 'movement']
        if args.med_type not in meditation_types:
            raise Exception('Invalid input. Provide valid med_type or input own script. Exitting...')

        prompts = {
            'focused': 'write me a focused meditation script designed to enhance focus and attention by noticing all 5 senses',
            'body-scan': 'write me a body scan meditation script to relax and relieve stress by tightening and relaxing muscles',
            'visualization': 'write me a visualization meditation script noticing all 5 senses at the beach/garden and is designed to boost mood, reduce stress, and promote inner peace',
            'reflection': 'write me a reflection meditation script designed to increase self awareness, mindfulness, and gratitude by asking the user about the current day and the recent past',
            'movement': 'write me a movement meditation script designed to improve mind body connection, energy, vitality, and the systems of the body'
        }
        prompt = prompts[args.med_type]

        ###################
        # Text generation #
        ###################
        script = text_generation(prompt)
        script_file_name = f'data/{args.med_type}_meditation_script.txt'
        try:
            with open(script_file_name, 'w') as f:
                f.write(script)
            print(f'Storing meditation script in {script_file_name}')
        except ValueError as error:
            print(f"Couldn't store script in file {script_file_name}: {error}")

    ##################
    # Text-to-speech #
    ##################
    audio_filename = f"data/{args.med_type}_meditation_audio_{args.accent}.mp3"

    sr = 22050  # default librosa sample rate
    pause_seconds = 2.0

    audio_segments, seconds_in_segments, sentiments, seconds = text_to_speech(
        script,
        args.accent,
        audio_filename,
        sr,
        pause_seconds,
        args.music_file
    )


    # Adding background music    
    if not args.skip_background_music:
        # if not os.path.isfile(args.music_file):
        #     raise Exception('Music file not found')
        # audio_output_filename = f"data/{args.med_type}_meditation_audio_background_music.mp3"
        # seconds = overlay_music_and_speech(
        #     audio_filename,
        #     args.music_file,
        #     audio_output_filename
        # )
        audio_output_filename = audio_filename
    else:
        audio_output_filename = audio_filename

    # ####################
    # # Video generation #
    # ####################
    if not args.skip_cppn_generation:

        inter = 25
        scale = 10  # 100
        temp_file = 'temp-file.mp4'
        trials_dir = 'data/trials'

        # Ensure trials directory is empty
        files = glob.glob(f'{trials_dir}/*')
        for f in files:
            os.remove(f)

        x_dim = args.x_dim
        y_dim = args.y_dim
        color_scheme = args.color_scheme

        fps = 25

        print('Creating imgs using cppn')
        frames_created = cppn.cppn(  # removed seconds
            interpolation=inter,
            c_dim=args.channels,
            # audio_file=audio_output_filename,
            scale=scale,
            trials_dir=trials_dir,
            x_dim=x_dim,
            y_dim=y_dim,
            color_scheme=color_scheme,
            audio_segments=audio_segments,
            sentiments=sentiments,
            seconds_in_segments=seconds_in_segments,
            sample_rate=sr,
            fps=fps,
            total_seconds=seconds,
            pause_seconds=int(pause_seconds)
        )
        # fps = round(frames_created / seconds)
        print('TOTALFRAMES: ', frames_created)
        print('SECONDS: ', seconds)
        # print('SECONDS OF VIDEO: ', frames_created / fps)
        print('FPS: ', fps)

        # Compile imgs into video
        print('\nCompiling imgs into video')
        os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{trials_dir}/*.png' -pix_fmt yuv420p \
                -c:v libx264 -crf 23 {temp_file} -loglevel quiet")

        # Add audio to video
        print('Adding audio to video')
        output_filename = f"output/{args.med_type}_meditation_audio_background_music.mp4"
        os.system(f'ffmpeg -i {temp_file} -i {audio_output_filename} -c:v copy -map 0:v -map 1:a \
                -y {output_filename} -loglevel quiet')

        os.system(f'rm {temp_file}')

    print(f'Completed. Video saved at {output_filename}')


if __name__ == '__main__':
    main()
