import random
import json
import os
import pyaudio
import wave


import torch
import speech_recognition as sr  # Import the SpeechRecognition library

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pyttsx3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine = pyttsx3.init()

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Specify the absolute file path
file_path = os.path.join(current_directory, 'intents.json')

with open(file_path, 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


def record_audio(filename, duration):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    #print("Recording...")
    print("Speak something:")

    frames = []

    for i in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    #print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

def recognize_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        user_input = recognizer.recognize_google(audio)
        return user_input

if __name__ == "__main__":
    
    while True:
        try:
            print("Let's chat! (type 'quit' to exit)")
            audio_file = "audio.wav"
            record_duration = 10  # Record for 5 seconds (adjust as needed)

            record_audio(audio_file, record_duration)
            
            sentence = recognize_audio(audio_file)
            print("You : ", sentence)
            if sentence == "quit":
                break

            resp = get_response(sentence)
            
            engine.say(resp)
            print(resp)
            engine.runAndWait()
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


