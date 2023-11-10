import random
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json').read())

# Tokenize and preprocess the data
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tags']))
        if intent['tags'] not in classes:
            classes.append(intent['tags'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Tokenize using TensorFlow's Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
words = tokenizer.word_index
num_words = len(words) + 1

# Prepare training data
X = []
Y = []

max_sequence_length = 179  # Set the maximum sequence length

for doc in documents:
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    sequence = tokenizer.texts_to_sequences([pattern_words])[0]

    # Create bag of words
    bag = [1 if i in sequence else 0 for i in range(1, num_words)]

    # Pad the bag of words to match the expected input shape
    bag = pad_sequences([bag], maxlen=max_sequence_length, padding='post')[0]

    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    X.append(bag)
    Y.append(output_row)

X = np.array(X)
Y = np.array(Y)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    Dense(128, input_shape=(max_sequence_length,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=1, validation_data=(X_test, Y_test))

# Save the model
model.save('chatbot_model.h5')
print('Done')
