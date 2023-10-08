![Blue Yellow Image](examples/color-scheme-images/blue-yellow.png) &nbsp;
![Gif of Video](/examples/cppn-video.gif) &nbsp;
![Red Orange Image](examples/color-scheme-images/red-orange.png) &nbsp;  
![Black White Image](examples/color-scheme-images/black-white-small.png) &nbsp;
![Program Structure Diagram](assets/meditation-induction-diagram.png) &nbsp;
![Blue Green Image](examples/color-scheme-images/blue-green-small.png)

<!-- ![Cool Image](/examples/color-scheme-images/cool-cppn-img.png) &nbsp; -->
<!-- ![Warm Image](/examples/color-scheme-images/warm-cppn-img.png) -->


# meditation_induction_ai
Clone the repo  
`git clone https://github.com/petern48/meditation_induction_ai.git`  
Change directory  
`cd meditation_induction_ai`

#### Install packages
Install the dependencies:  
`pipenv install`  
Activate the Python virtual environment:  
`pipenv shell`

Activating the virtual environment allows you to work within an isolated environment where the dependencies you installed with pipenv install are available. This ensures that your project uses the correct versions of packages and avoids conflicts with system-wide packages.

Also, it is necessary to install:  
`sudo apt install ffmpeg`  
`pipenv run python -m nltk.downloader punkt vader_lexicon`

### Program Explanation
Generate a meditation video with speech (and optionally music) using AI models.
Text Generation: Generate a meditation script by specifying the desired type of meditation.
Text to Audio: Create speech for the text and (optionally add music) to it.
Video Generation: Generate relaxing visuals by inputting audio into a Compositional Pattern Producing Network (CPPN)

Select a meditation type from the following list:  
`['mindful-observation', 'body-centered', 'visual-concentration', 'contemplation', 'affect-centered', 'mantra-meditation', 'movement-meditation']`

The program will generate a script for the meditation, feed that script to create audio, and feed that
audio into the CPPN to generate a video. The video will come with the audio and (optional) music. 

### Run the program:
Produce a meditation by providing a *med_type* (see below)  
`python main.py --med_type [med_type]`
such as:
`python main.py --med_type focused`

By default, the background music will be added and cppn-based images will be generated unless you specify it as follows:
- Skip background music: `python main.py --med_type [med_type] --skip_background_music`
- Skip cppn-based image generation: `python main.py --med_type [med_type] --skip_cppn_generation`

Run the program while skipping the text generation
`python main.py --med_type [med_type] --script_file [text_file]`

Afterwards, the resulting folder `output` will contain the script and video.

### Command Line Options

- `--med_type` _[required]_ select the med_type  
Options: ['mindful-observation', 'body-centered', 'visual-concentration', 'contemplation', 'affect-centered', 'mantra-meditation', 'movement-meditation']

- `--fps` [optional] select the frames per second the resulting video will be (more fps means more time to generate).  
Default: 20

- `--script_file` [optional] input a file path to a text file to skip text generation step

<!-- - `--text_gen_only` [optional] stop after generating the meditation script -->

- `--accent` [optional] select an accent for the speech to be spoken in  
Default: indian, co.in   
Options: ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']

- `--music_file` [optional] specify a audio file path to use for background music  
Default: use the provided music file

- `--channels` [optional] number of channels. 3 will use RGB, 1 will use black/white  
Default: 3

- `--x_dim` [optional] Specify the x size of images  
Default: 256

- `--y_dim` [optional] Specify the y size of images  
Default: 256

<!-- - `--out_file` [optional] Specify the name of the output file  
Default: med_video.mp4 -->

- `--color_scheme` [optional] Specify `cool` for a cool color scheme (good for relaxation) or `warm` for a warm color schem (good for energy and focus)  
Options: ['red-orange', 'blue-green', 'blue-yellow', 'black-white', 'warm', 'cool']

- `--show_ffmpeg_output` [optional] Do not suppress the output of the ffmpeg commands (useful if some sort of error happens)  

- `--skip_cppn_generation` [optional] Skips the video generation

- `--skip_background_music` [optional] Skip overlaying background music


<!-- 
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
Benefits: improve mind body connection, energy, vitality, and systems of the body (e.g digestive, immune) -->

### References
Original CPPN model taken from https://github.com/neale/CPPN

Inspiration taken from the following articles  
https://nenadmarkus.com/p/visualizing-audio-with-cppns/  
https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/