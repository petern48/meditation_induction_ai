# meditation_induction_ai

### Run the program:
Produce visualization (example) meditation  
`python main.py --med_type visualization`

Run the program while skipping the text generation  
`python main.py --med_type visualization --script_file [text_file]`

Afterwards, the resulting file (med_video.mp4 or provided name) will be viewable

### Command Line Options

- `--med_type` [required] select the med_type  
Options: [focused]  [body-scan]  [visualization]  [reflection]  [movement]

- `--script_file` [optional] input a file path to a text file to skip text generation step

- `--accent` [optional] select an accent for the speech to be spoken in  
Default is indian, co.in   
Options: [com.au] [co.uk] [us] [ca] [co.in] [ie] [co.za]

- `--music_file` [optional] specify a audio file path to use for background music  
Default: use the provided music file

- `--channels` [optional] number of channels. 3 will use RGB, 1 will use black/white  
Default: 3

- `--x_dim` [optional] Specify the x size of images  
Default: 256

- `--y_dim` [optional]
Specify the y size of images  
Default: 256

- `out_file` [optional]
Specify the name of the output file
Default: med_video.mp4