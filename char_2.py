from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	encoded_seq = [mapping[char] for char in line]
	sequences.append(encoded_seq)

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]

X = np.array(sequences)

y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop =EarlyStopping(monitor='val_acc', min_delta=0.01, patience=20, verbose=2, mode='max')
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X, y,validation_split=0.2, epochs=1000, verbose=2,batch_size = 10000,callbacks=callbacks_list)

model.save('model.h5')
dump(mapping, open('mapping.pkl', 'wb'))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

