from gtts import gTTS
import math
import os
from pydub import AudioSegment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def text_to_speech(meditation_script, accent, filename):
    """Generates audio file given using gTTS library"""

    # List of accents available in gTTS
    if accent not in ['com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']:
        raise Exception('Invalid accent')

    sentiments = []
    audio_segments = []  # Not in original program

    sentences = nltk.sent_tokenize(meditation_script)

    # Add longer pause after every sentence
    pause = AudioSegment.silent(duration=2000)  # milliseconds
    speech_audio = AudioSegment.empty()
    analyzer = SentimentIntensityAnalyzer()

    for sentence in sentences:
        # Text to speech
        tts = gTTS(
            sentence,
            tld=accent,    # Indian voice sounded the most soothing
            lang="en",
            slow=True
        )
        tts.save('segment.mp3')
        segment_audio = AudioSegment.from_mp3('segment.mp3') + pause
        audio_segments.append(segment_audio)
        speech_audio += segment_audio

        # Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(sentence)['compound']
        sentiments.append(sentiment_score)

    # Save the audio file
    print(f"Saving {filename} audio file")
    speech_audio.export(filename, format='mp3')
    os.remove('segment.mp3')  # Remove the temporary file

    return filename, audio_segments, sentiments, sentences


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
    with open(filename, 'r') as f:
        script = f.read()
    filename, audio_segments, sentiments, sentences = text_to_speech(script, 'co.in', 'temp_output.mp3')
    print(filename)
    for i in range(len(sentences)):
        print(audio_segments[i])
        print(sentences[i])
        print(sentiments[i])
        print()