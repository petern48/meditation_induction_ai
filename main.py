import argparse
import glob
import os

# from src.cppn import cppn
from src.text_generation import text_generation
from src.text_to_audio import text_to_speech, overlay_music_and_speech


def load_args():
    parser = argparse.ArgumentParser(description='meditation_induction')
    # Non-default arguments
    parser.add_argument('--med_type', type=str,
                        choices=['focused', 'body-scan', 'visualization', 'reflection', 'movement'],
                        help="type of meditation: '[focused] [body-scan] [visualization] [reflection] [movement]")
    parser.add_argument('--out_file',
                        help='output file name')
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
    # Skip generation
    parser.add_argument('--skip_audio_generation', action='store_false',
                        help='skip audio generation')
    parser.add_argument('--skip_cppn_generation', action='store_false',
                        help='skip cppn generation')

    args = parser.parse_args()
    return args


def main():
    # Load Arguments and check if valid
    args = load_args()

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
    if args.skip_audio_generation:
        audio_filename = f"data/{args.med_type}_meditation_audio_{args.accent}.mp3"
        text_to_speech(script, args.accent, audio_filename)

        # Adding background music
        if not os.path.isfile(args.music_file):
            raise Exception('Music file not found')
        audio_background_filename = f"data/{args.med_type}_meditation_audio_{args.accent}_background_music.mp3"
        overlay_music_and_speech(audio_filename, args.music_file, audio_background_filename)

    ####################
    # Video generation #
    ####################
    if not args.skip_cppn_generation:
        pass

    # imgs_dir = 'data/trials'

    # # Ensure trials directory is empty
    # files = glob.glob(f'{imgs_dir}/*')
    # for f in files:
    #     os.remove(f)

    # # Generations images
    # frames_created, seconds = cppn.cppn(
    #     interpolation=25,
    #     c_dim=args.channels,
    #     audio_file=audio_file_name,
    #     scale=100
    # )
    # fps = round(frames_created / seconds)
    # print('frames_created', frames_created)
    # print('seconds', seconds)
    # print('fps', fps)






    # # Input audio into cppn to create imgs
    # cmd = f'echo overwrite | python cppn.py --walk --x_dim {args.x_dim} --y_dim {args.y_dim} \
    #           --c_dim {args.c_dim} --interpolation {inter} --scale {scale}'
    # # # Input only speech into CPPN
    # #   --audio_file {speech_only_file}
    # if audio_file_name:
    #     cmd += f'--audio_file {audio_file_name}'
    # if args.color_scheme:
    #     cmd += f' --color_scheme {args.color_scheme}'
    # # cmd += ' > /dev/null'
    # # os.system(cmd)




    # Images to video
    # print('\nCompiling imgs into video')
    # os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{imgs_dir}/*.png' -pix_fmt yuv420p \
    #           -c:v libx264 -crf 23 {temp_file}")

    # Add audio to video
    # print('\nAdding audio to video')
    # os.system(f'ffmpeg -i {temp_file} -i {audio_file_name} -c:v copy -map 0:v -map 1:a \
    #           -y {args.out_file}')  #  > /dev/null 2> /dev/null
    # os.system(f'rm {temp_file}')


if __name__ == '__main__':
    main()
