from gtts import gTTS
import math
import os
from pydub import AudioSegment


def text_to_speech(meditation_script, accent, filename):
    """Generates audio file given using gTTS library"""
    # gTTS implementation

    # List of accents available in gTTS
    if accent not in ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']:
        raise Exception('Incorrect accent')

    # Add longer pause after every period
    segments = meditation_script.split('. ')
    pause = AudioSegment.silent(duration=2000)  # milliseconds
    speech_audio = AudioSegment.empty()
    for segment in segments:
        # Create the TTS object
        tts = gTTS(
            segment,
            tld=accent,    # Indian voice sounded the most soothing
            lang="en",
            slow=True
        )
        tts.save('segment.mp3')
        segment_audio = AudioSegment.from_mp3('segment.mp3')
        speech_audio += segment_audio + pause

    # Save the audio file
    print(f"Saving {filename} audio file")
    speech_audio.export(filename, format='mp3')
    os.remove('segment.mp3')  # Remove the temporary file

    return filename


def overlay_music_and_speech(speech_file_path, music_file_path, filename):
    """Add background music to speech given path to file and speech mp3 files"""

    music1 = AudioSegment.from_mp3(music_file_path)
    speech = AudioSegment.from_mp3(speech_file_path)
    times_to_repeat = len(speech) / len(music1)

    # Lengthen music so it is at least the length of speech audio
    longer_music = AudioSegment.empty()
    for _ in range(math.ceil(times_to_repeat)):
        longer_music += music1

    print(f"Saving {filename} audio file")
    combined = speech.overlay(longer_music)
    combined.export(filename, format="mp3")

    return filename
