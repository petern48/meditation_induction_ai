# meditation_induction_ai
Clone the repo  
`git clone https://github.com/petern48/meditation_induction_ai.git`  
Change directory  
`cd meditation_induction_ai`

#### Install packages
Create conda environments with proper packages
`conda create --name med_ai --file requirements-conda.txt`

#### Alternative without conda
Create conda environment (optional but recommended)
`conda create -n med_ai python==3.9`  
Activate environment  
`conda activate med_ai`  
Install packages
`pip install -r requirements.txt`

### Explanation of Program
Generate a meditation video with speech (and optionally music) using AI models.
Text Generation: Generate a meditation script by specifying the desired type of meditation.
Text to Audio: Create speech for the text and (optionally add music) to it.
Video Generation: Generate relaxing visuals by inputting audio into a Compositional Pattern Producing Network (CPPN)

Select a type of meditation from the following list:
[focused]  [body-scan]  [visualization]  [reflection]  [movement]
The program will generate a script for the meditation, feed that script to create audio, and feed that
audio into the CPPN to generate a video. The video will come with the audio and (optional) music. 

### Run the program:
To improve speed, remove all files in the trials directory before running the program  
`rm trials/*`

Produce a meditation by providing a *med_type* (see below)  
`python main.py --med_type [med_type]`

Run the program while skipping the text generation  
`python main.py --med_type [med_type] --script_file [text_file]`

Afterwards, the resulting file (`out_file`) will contain the video

### Command Line Options

- `--med_type` _[required]_ select the med_type  
Options: [focused]  [body-scan]  [visualization]  [reflection]  [movement]

- `--script_file` [optional] input a file path to a text file to skip text generation step

- `--text_gen_only` [optional] stop after generating the meditation script

- `--accent` [optional] select an accent for the speech to be spoken in  
Default: indian, co.in   
Options: [com.au] [co.uk] [us] [ca] [co.in] [ie] [co.za]

- `--music_file` [optional] specify a audio file path to use for background music  
Default: use the provided music file

- `--no_music` [optional] don't add background music

- `--channels` [optional] number of channels. 3 will use RGB, 1 will use black/white  
Default: 3

- `--x_dim` [optional] Specify the x size of images  
Default: 256

- `--y_dim` [optional]
Specify the y size of images  
Default: 256

- `--out_file` [optional]
Specify the name of the output file  
Default: med_video.mp4

### Types of Meditation
- `focused` focus on each of the 5 senses  
Benefits: enhance focus and attention

- `body scan` slowly tighten and relax one muscle at a time  
Benefits: relax and reduce tension in the body, unwind before bedtime, sync body and mind

- `visualization` imagine vivid scene using all 5 senses  
Benefits: boost mood, reduce stress, promote inner peace  

- `reflection` pay attention to the feelings and thoughts that arise  
Benefits: increase self-awareness, emotional regulation, mindfulness, gratitude

- `movement` focus by performing various body motions, focus on the fluidity of motions  
Benefits: improve mind body connection, energy, vitality, and systems of the body (e.g digestive, immune)

### References
Original CPPN model taken from https://github.com/neale/CPPN

Inspiration taken from the following articles  
https://nenadmarkus.com/p/visualizing-audio-with-cppns/
https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/