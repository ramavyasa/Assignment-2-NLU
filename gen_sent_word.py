import re
import nltk
import math
from pickle import load
from random import randint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from pickle import dump

from nltk.corpus import gutenberg
import numpy as np
np.set_printoptions(threshold=np.inf)

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
	return in_text


model = load_model('weights_word.best.hdf5')
mapping = load(open('mapping_words.pkl', 'rb'))

print(generate_seq(model, mapping, 179, 'she did', 10))