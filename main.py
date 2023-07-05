import torch
import os
from dotenv import load_dotenv
import pyttsx3  # text to speech
from gtts import gTTS
# from lwe import ApiBackend
# from playsound import playsound
import openai


FILE_NAME = "chatgpt-script1.txt"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def chatgbt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100
    )
    # message = response.choices[0].text.strip()
    return response


def generate_text():
    # Just read scripts from .txt file for now
    curr_path = os.getcwd()
    dir_path = os.path.join(curr_path, "med_scripts")
    file_path = os.path.join(dir_path, FILE_NAME)
    with open(file_path, "r") as f:
        meditation_script = f.read()
        # print(meditation_script)

    # Need to clean out the brief bracketed pauses
    return meditation_script


def text_to_speech(meditation_script):
    # gTTS implementation

    # Create the TTS object
    tts = gTTS(meditation_script, lang="en", tld="co.in")  # Indian voice sounded nicer

    # Save the audio file
    print("Saving gtts audio file")
    tts.save("gtts-audio.mp3")

    # pyttsx3 implementation

    engine = pyttsx3.init() # object creation
    # rate = engine.getProperty('rate')   # getting details of current speaking rate
    # engine.setProperty('rate', 125)     # setting up new voice rate
    # volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
    # engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
    # print("Saying script")
    # # engine.say("meditation_script")

    # Listen to different voices (only 2 options??) 
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)  # Female voice sounded nicer
    # for voice in voices:
        # engine.setProperty("voice", voice.id)
    #     print(f"voice id {voice.id}")
    #     engine.say("Hello")
    # engine.runAndWait()

    # # Save to audio file
    print("Saving pyttsx3 audio file")
    engine.save_to_file(meditation_script, 'pyttsx3.mp3')
    engine.runAndWait()


def main():
    meditation_script = generate_text()
    text_to_speech(meditation_script)


if __name__ == "__main__":
    main()
