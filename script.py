
# Preparing data

# %%

from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
from scipy.io import arff
import pandas as pd

# Loading the dataset
dataset = arff.loadarff('EEG_Eye_State.arff')
dframe = pd.DataFrame(dataset[0])

# veryfy columns
dframe.head()


# %%
# verify row size
dframe.count


# How to pre-process the data?
emb_output_size = 32
max_features = 10000
maxlen = 500
batch_size = 32
# create a vector of size 14
embed_vector = []
# select output column - eyeDetection
actual_output = dframe.columns[14]

print(actual_output)

model = Sequential()
model.add(Embedding(max_features, emb_output_size, input_length=maxlen))
model.add(LSTM(emb_output_size))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(train_x_cutoff, train_y, epochs=10,
                    batch_size=128, validation_split=0.2)
