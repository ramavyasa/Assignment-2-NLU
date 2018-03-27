from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	for _ in range(n_chars):
		encoded = [mapping[char] for char in in_text]
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		encoded = to_categorical(encoded, num_classes=len(mapping))
		yhat = model.predict_classes(encoded, verbose=0)
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		in_text += char
	return in_text

model = load_model('model.h5')
mapping = load(open('mapping.pkl', 'rb'))

print(generate_seq(model, mapping, 20, 'when she got to the', 100))
print(generate_seq(model, mapping, 20, 'she was in the', 100))