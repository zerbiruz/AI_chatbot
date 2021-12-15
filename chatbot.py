import random
import json
import pickle
import numpy as np 

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import WordNetLemmatizer

from tensorflow.keras.models import load_model

ignore_letters = ['?', '!', '.', ',']

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("./resources/intents.json").read())["intents"]

words = pickle.load(open("./resources/words.pkl", 'rb'))
classes = pickle.load(open("./resources/classes.pkl", 'rb'))
model = load_model("./models/chatbot.h5")

print(classes)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]
    return sentence_words

def bag_of_word(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_word(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })
    
    return return_list

def lets_chat(): 
    isQuit = False

    while not isQuit:
        sentence = input("You sey: ")

        predict = predict_class(sentence)[0]
        intent = predict["intent"]
        probability = predict["probability"]

        response_intent = intents[classes.index(intent)]
        response_sentence_list = response_intent["responses"]
        response_sentence = random.choice(response_sentence_list)

        print(f"Bot say: {response_sentence}")

if __name__ == "__main__":
    lets_chat()