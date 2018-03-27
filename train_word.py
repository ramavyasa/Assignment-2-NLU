import re
import nltk
import math
from random import randint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from pickle import dump

from nltk.corpus import gutenberg
import numpy as np
np.set_printoptions(threshold=np.inf)

train_data = list()
test_data = list()
data = list()
all_words = set()




def extract():
	f = gutenberg.sents("carroll-alice.txt")
	for i in f:
		sen = []
		for j in i:
			k = re.sub(r'[^\w\s]','',j.strip().lower())
			sen.append(k) if k is not '' else None
		data.append(' '.join(sen))
	
	
			
extract()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
encoded = [tokenizer.texts_to_sequences([line])[0] for line in data]				
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
sequences = list()
for line in encoded:	
	for i in range(1, len(line)):
		sequence = line[:i+1]
		sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
max_length = max([len(seq) for seq in encoded])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
sequences = np.array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=max_length-1))
model.add(LSTM(128))	
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

filepath="weights_word.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early_stop =EarlyStopping(monitor='val_acc', min_delta=0.01, patience=20, verbose=2, mode='max')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X, y,validation_split=0.25, epochs=2000, verbose=2,batch_size = 2048,callbacks=callbacks_list)

model.save('model_words.h5')
dump(tokenizer, open('mapping_words.pkl', 'wb'))
f= open("max",'w')
f.write(str(max_length-1))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc_words.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_words.png')



