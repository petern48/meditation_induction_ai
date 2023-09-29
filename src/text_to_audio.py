from gtts import gTTS
import math
import os
from pydub import AudioSegment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import librosa
import soundfile as sf
import numpy as np


def text_to_speech(meditation_script, accent, output_filename, sr):
    """Generates audio file given using gTTS library"""

    # List of accents available in gTTS
    if accent not in ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']:
        raise Exception('Invalid accent')

    audio_segments = []  # List of separate audio arrays
    sentiments = []

    sentences = nltk.sent_tokenize(meditation_script)

    analyzer = SentimentIntensityAnalyzer()
    # pause = AudioSegment.silent(duration=2000)  # milliseconds
    # speech_audio = AudioSegment.empty()
    temp_file = 'segment.mp3'
    pause_duration = 1.0  # seconds
    pause = np.zeros(int(pause_duration * sr), dtype=np.float32)
    combined_audio = np.empty(0)
    seconds_in_segments = []

    for sentence in sentences:
        # Text to speech
        tts = gTTS(
            sentence,
            tld=accent,
            lang="en",
            slow=True
        )
        tts.save(temp_file)

        # Convert to librosa arrays and save in list
        segment, sr = librosa.load(temp_file, sr=sr)
        # Add longer pause after every sentence
        segment = np.concatenate((segment, pause))
        segment_seconds = len(segment) / sr
        audio_segments.append(segment)
        seconds_in_segments.append(segment_seconds)
        combined_audio = np.concatenate((combined_audio, segment))

        # Concatenate audio for pydub
        # segment_audio = AudioSegment.from_mp3(temp_file) #  + pause
        # speech_audio += segment_audio

        # Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(sentence)['compound']
        sentiments.append(sentiment_score)

    os.remove(temp_file)

    # Save the audio file
    # speech_audio.export(output_filename, format='mp3')  # save as pydub
    print(f"Saving {output_filename} audio file")
    sf.write(output_filename, combined_audio, sr)
    seconds = len(combined_audio) / sr

    return audio_segments, seconds_in_segments, sentiments, sentences, seconds


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


# For testing purposes
if __name__ == '__main__':
    filename = 'examples/llama2-chat-body-scan.txt'
    output_filename = 'development_output.mp3'
    sr = 22050  # default librosa value
    with open(filename, 'r') as f:
        script = f.read()
    filename, audio_segments, sentiments, sentences, seconds = text_to_speech(script, 'co.in', output_filename, sr)
    # os.remove(output_filename)
    print(filename)
    for i in range(len(sentences)):
        print(sentences[i])
        print('librosa audio_segment', audio_segments[i])
        print('sentiment', sentiments[i])
        print()