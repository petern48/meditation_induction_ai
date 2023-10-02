from gtts import gTTS
import math
import os
from pydub import AudioSegment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import librosa
import soundfile as sf
import numpy as np


def text_to_speech(meditation_script, accent, output_filename, sr, pause_seconds, music_file_path):
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
    pause = np.zeros(int(pause_seconds * sr), dtype=np.float32)
    combined_audio = np.empty(0)
    seconds_in_segments = []

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


        # overlay music here?
        temp_audio = AudioSegment.from_mp3(temp_file)
        # Add longer pause after every sentence
        temp_audio += AudioSegment.silent(duration= pause_seconds * 1000)
        end_music = overlay_music_and_speech(temp_audio, music1, start_music, temp_file)
        start_music = end_music


        # Convert to librosa arrays and save in list
        segment, _ = librosa.load(temp_file, sr=sr)
        # # Add longer pause after every sentence
        # segment = np.concatenate((segment, pause))
        segment_seconds = len(segment) / sr  # float
        audio_segments.append(segment)
        seconds_in_segments.append(segment_seconds)
        combined_audio = np.concatenate((combined_audio, segment))

        # Concatenate audio for pydub
        # segment_audio = AudioSegment.from_mp3(temp_file) # + pause
        # speech_audio += segment_audio

        # Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(sentence)['compound']
        sentiments.append(sentiment_score)

    os.remove(temp_file)

    # Save the audio file
    # speech_audio.export(output_filename, format='mp3')  # save as pydub
    # print(f"Saving {output_filename} audio file")
    sf.write(output_filename, combined_audio, sr)
    seconds = len(combined_audio) / sr

    return audio_segments, seconds_in_segments, sentiments, seconds


def overlay_music_and_speech(speech_audio, music_audio, start_idx, filename):
    """Add background music to speech given path to file and speech mp3 files"""

    # music1 = AudioSegment.from_mp3(music_file_path)
    # speech = AudioSegment.from_mp3(speech_file_path)
    length_music = len(music_audio)
    speech_duration = len(speech_audio)
    # If not longer
    end_idx = start_idx + speech_duration
    if end_idx <= length_music - 1:
        specific_audio = music_audio[start_idx:end_idx]

    # Make music wrap around
    else:
        second_end = speech_duration - (length_music-1 - start_idx)
        specific_audio = music_audio[start_idx:length_music - 1] + \
        music_audio[0:second_end]
        end_idx = second_end  # for return value

    # times_to_repeat = math.floor(len(speech) / len(music1))

    # Lengthen music so it matches length of speech file
    # longer_music = AudioSegment.empty()
    # for _ in range(times_to_repeat):
    #     longer_music += music1

    # # Add the remaining bit to make them exactly the same length
    # remaining_time = len(speech) - len(longer_music)
    # longer_music += music1[:remaining_time]

    assert(len(speech_audio) == len(specific_audio))

    # print(f"Saving {filename} audio file")
    combined = speech_audio.overlay(specific_audio)
    combined.export(filename, format="mp3")

    seconds = combined.duration_seconds

    return end_idx


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