import argparse
import glob
import os

from src import cppn
from src.text_generation import text_generation, get_prompt
from src.text_to_speech import text_to_audio_and_sentiments


MEDITATION_TYPES = ['mindful-observation', 'body-centered', 'visual-concentration', 'contemplation', 'affect-centered', 'mantra', 'movement']
ACCENT_OPTIONS = ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']
COLOR_SCHEMAS = ['red-orange', 'blue-green', 'blue-yellow', 'black-white', 'warm', 'cool']


def load_args():
    parser = argparse.ArgumentParser(description='meditation_induction')
    # Non-default arguments
    parser.add_argument('--med_type', type=str,
                        choices=MEDITATION_TYPES,
                        help=f"types of meditation: {MEDITATION_TYPES}")
    # Default arguments
    parser.add_argument('--fps', default=20, type=int,
                        help='higher frames per second will generate more frames and take longer to generate')
    parser.add_argument('--script_file', type=str, default='',
                        help='input script to skip text generation, hence, there is no script generation')
    parser.add_argument('--accent', type=str, default='co.in',
                        choices=ACCENT_OPTIONS,
                        help=f"Select one: {ACCENT_OPTIONS}")
    parser.add_argument('--music_file', default='assets/default_background_music.mp3', type=str,
                        help='background music')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3],
                        help='3 for RGB, 1 for black/white')
    parser.add_argument('--x_dim', default=256, type=int,
                        help='out image width')
    parser.add_argument('--y_dim', default=256, type=int,
                        help='out image height')
    parser.add_argument('--color_scheme', default='', type=str,
                        choices=COLOR_SCHEMAS,
                        help=f"Select one: {COLOR_SCHEMAS}")
    parser.add_argument('--show_ffmpeg_output', default=False, action='store_true',
                        help='Show the ffmpeg output instead of supressing it. Good if it runs into some error.')
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
        raise Exception('Invalid number of channels. Must be (1) for black/white or (3) for RGB')

    if args.color_scheme == 'black-white':
        args.channels = 1
        args.color_scheme = False
    else:
        args.channels = 3

    if args.script_file:
        with open(args.script_file, 'r') as f:
            script = f.read()
        script_base_file_name = os.path.basename(args.script_file)
        last_period_idx = script_base_file_name.rfind('.')
        script_base_file_name = script_base_file_name[:last_period_idx].replace(' ', '-')
        print('script_file provided, skipping text generation')
    else:
        if args.med_type not in MEDITATION_TYPES:
            raise Exception('Invalid input. Provide valid med_type or input own script. Exitting...')

        # Get context input from user
        print("How are you feeling today.\nProvide context for your meditation (or press enter to skip): e.g I'm tired and I'm getting ready for bed.")
        context = input()

        prompt = get_prompt(args.med_type.replace('-', ' '), context)  # replace dash with space

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
    if args.script_file:
        base_name = script_base_file_name
        audio_filename = f"data/{base_name}-{args.accent}.mp3"

    else:
        base_name = args.med_type + "-meditation_audio"
        audio_filename = f"data/{args.med_type}-meditation-audio-{args.accent}.mp3"

    sr = 22050  # default librosa sample rate
    pause_seconds = 2.0

    music_file = args.music_file
    if args.skip_background_music:
        music_file = None

    # Apply sentiment analysis, create audio file and overlay background music
    audio_segments, seconds_in_segments, sentiments, seconds = text_to_audio_and_sentiments(
        script,
        args.accent,
        audio_filename,
        sr,
        pause_seconds,
        music_file
    )

    ####################
    # Video generation #
    ####################
    if not args.skip_cppn_generation:

        scale = 10
        temp_file = 'temp-file.mp4'
        trials_dir = 'data/trials'

        # Ensure trials directory is empty
        files = glob.glob(f'{trials_dir}/*')
        for f in files:
            os.remove(f)

        x_dim = args.x_dim
        y_dim = args.y_dim
        color_scheme = args.color_scheme

        print(f'Creating imgs using cppn for {args.fps} fps')
        frames_created = cppn.cppn(  # removed seconds
            c_dim=args.channels,
            scale=scale,
            trials_dir=trials_dir,
            x_dim=x_dim,
            y_dim=y_dim,
            color_scheme=color_scheme,
            audio_segments=audio_segments,
            sentiments=sentiments,
            seconds_in_segments=seconds_in_segments,
            sample_rate=sr,
            fps=args.fps,
            total_seconds=seconds,
            pause_seconds=int(pause_seconds)
        )
        print('TOTALFRAMES: ', frames_created)
        print('SECONDS: ', seconds)
        print('FPS: ', args.fps)

        # Compile imgs into video

        if args.show_ffmpeg_output:
            quiet_output = ''
        else:
            quiet_output = ' -loglevel quiet'  # silences the output of ffmpeg cmds

        print('\nCompiling imgs into video')
        os.system(f"ffmpeg -framerate {args.fps} -pattern_type glob -i '{trials_dir}/*.png' -pix_fmt yuv420p \
                -c:v libx264 -crf 23 {temp_file}{quiet_output}")

        # Add audio to video
        print('Adding audio to video')
        output_filename = f"output/{base_name}.mp4"
        os.system(f'ffmpeg -i {temp_file} -i {audio_filename} -c:v copy -map 0:v -map 1:a \
                -y {output_filename}{quiet_output}')

        os.system(f'rm {temp_file}')

    print(f'Completed. Video saved at {output_filename}')


if __name__ == '__main__':
    main()
