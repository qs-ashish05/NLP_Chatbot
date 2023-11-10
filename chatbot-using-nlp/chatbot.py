import random
import json
import pickle
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
import pyttsx3

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize the pyttsx3 text-to-speech engine
engine = pyttsx3.init()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    #print(np.array(bag))
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running")

while True:
    try:
        # Listen for speech input
        with sr.Microphone() as source:
            print("Speak something:")
            audio = recognizer.listen(source)

        # Recognize speech input
        user_input = recognizer.recognize_google(audio)
        print("You said:", user_input)

        # Get chatbot response
        ints = predict_class(user_input)
        
        # Debug: Print predicted intents and their probabilities
        for intent in ints:
            print(f"Predicted Intent: {intent['intents']} (Probability: {intent['probability']})")

        chatbot_response = get_response(ints, intents)

        # Output chatbot response as speech
        engine.say(chatbot_response)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")