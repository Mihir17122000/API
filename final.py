import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import Text, END, scrolledtext
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load pre-trained model and resources
model = load_model('chatassistant_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def send_message(event=None):
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Arial", 12))

        res = chatbot_response(msg)
        chat_log.insert(tk.END, "Prince Solutions: " + res + '\n\n')

        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

base = tk.Tk()
base.title("Prince Solutions Chat Assistant")
base.geometry("400x500")
base.resizable(width=tk.FALSE, height=tk.FALSE)

chat_log = scrolledtext.ScrolledText(base, bd=0, bg="white", height="8", width="50", font=("Arial", 12))
chat_log.config(state=tk.DISABLED)

entry_box = Text(base, bd=0, bg="white", width="29", height="2", font=("Arial", 12), borderwidth=2)
entry_box.bind("<Return>", send_message)

send_button = tk.Button(base, font=("Arial", 12, 'bold'), text="Send", width="12", height=2,
                     bd=0, bg="#5500C2", activebackground="#9C4EFF", fg='#ffffff',
                     command=send_message)

# Place components in the window
chat_log.place(x=5, y=5, height=386, width=390)
entry_box.place(x=5, y=401, height=50, width=300)
send_button.place(x=310, y=401, height=50, width=85)

base.mainloop()
