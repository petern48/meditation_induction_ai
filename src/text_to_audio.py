from gtts import gTTS
import math
import os
from pydub import AudioSegment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import librosa
import soundfile as sf
import numpy as np


def text_to_audio_and_sentiments(meditation_script, accent, output_filename, sr, pause_seconds, music_file_path=None):
    """Generates audio file given using gTTS library"""

    # List of accents available in gTTS
    if accent not in ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']:
        raise Exception('Invalid accent')

    audio_segments = []  # List of separate audio arrays
    sentiments = []

    sentences = nltk.sent_tokenize(meditation_script)

    analyzer = SentimentIntensityAnalyzer()
    temp_file = 'segment.mp3'

    combined_audio = np.empty(0)
    seconds_in_segments = []

    if music_file_path != None:
        music1 = AudioSegment.from_mp3(music_file_path)
        start_music = 0

    for sentence in sentences:
        # Text to speech
        tts = gTTS(
            sentence,
            tld=accent,
            lang="en",
            slow=True
        )
        tts.save(temp_file)


        temp_audio = AudioSegment.from_mp3(temp_file)
        # Add longer pause after every sentence
        temp_audio += AudioSegment.silent(duration= pause_seconds * 1000)
        if music_file_path != None:
            end_music = overlay_music_and_speech(temp_audio, music1, start_music, temp_file)
            start_music = end_music

        # Convert to librosa arrays to prepare for audio feature extraction
        segment, _ = librosa.load(temp_file, sr=sr)
        # segment = np.concatenate((segment, pause))
        segment_seconds = len(segment) / sr  # float
        audio_segments.append(segment)
        seconds_in_segments.append(segment_seconds)
        combined_audio = np.concatenate((combined_audio, segment))

        # Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(sentence)['compound']
        sentiments.append(sentiment_score)

    os.remove(temp_file)

    # Save the audio file
    sf.write(output_filename, combined_audio, sr)
    seconds = len(combined_audio) / sr

    return audio_segments, seconds_in_segments, sentiments, seconds


def overlay_music_and_speech(speech_audio, music_audio, start_idx, filename):
    """Add background music to speech given path to file and speech mp3 files"""

    length_music = len(music_audio)
    speech_duration = len(speech_audio)

    end_idx = start_idx + speech_duration
    if end_idx <= length_music - 1:
        specific_audio = music_audio[start_idx:end_idx]

    # Make music wrap around to the beginning
    else:
        second_end = speech_duration - (length_music-1 - start_idx)
        specific_audio = music_audio[start_idx:length_music - 1] + \
        music_audio[0:second_end]
        end_idx = second_end  # for return value

    assert(len(speech_audio) == len(specific_audio))

    combined = speech_audio.overlay(specific_audio)
    combined.export(filename, format="mp3")

    return end_idx
