#%%
# Import 
import pandas as pd
import numpy as np
import os, re, datetime, json, pickle
from tensorflow import keras
from keras.utils import pad_sequences, plot_model
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Embedding
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#%%
#1. Data Loading
df=pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")

#%%
#2. Data Inspection
df.info()
df.describe()

#%%
#2.1 Data Inspection
print(f'Missing Values: {df.isna().sum()}')
print(f'Duplicated text: {df.duplicated().sum()}')

#%%
#3. Data Cleaning
df=df.drop_duplicates()
print(df.duplicated().sum())

features=df['text'] #Features > X
labels=df['category'] #Target > y

#temp=[]

for index, txt in features.items():
    features[index]=re.sub('[^a-zA-Z]',' ',txt).lower()

print('\n', features[10])

#%%
#4. Features Selection

#%%
#5. Data Pre-processing
num_words=5000
oov_token='<OOV>'
trunc_type='post'
pad_type='post'

tokenizer=Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(features)

train_sequences = tokenizer.texts_to_sequences(features)

#Padding + truncating
train_sequences = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')

#%%
# Expand the dimension of the labels and features into 2d array
train_sequences = np.expand_dims(train_sequences, -1)
train_labels = np.expand_dims(labels, -1)

# Pre-processing for the labels - Encode the label with OneHotEncoding
ohe = OneHotEncoder(sparse=False)
train_labels = ohe.fit_transform(train_labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_labels, random_state=123)

#%%
#6. Model Development
nodes=128
model=Sequential()

model = Sequential()
model.add(Embedding(num_words,nodes))
model.add(Bidirectional(LSTM(nodes,return_sequences=True)))
model.add(LSTM(nodes))
model.add(Dropout(0.3))
model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

#callbacks
#early stopping and tensorboard

LOGS_PATH=os.path.join(
    os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%M%d-%H%M%S'))


tensorbaord_callback=TensorBoard(log_dir=LOGS_PATH)

hist=model.fit(X_train, y_train, epochs=5, callbacks=[tensorbaord_callback], validation_data=(X_test,y_test))

# %%
#7. Model Evaluation
y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)

# Evaluate prediction
print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

# %%
#Save Tokenizer
import json

with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(),f)

#Save ohe
import pickle 
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

model.save('text.h5')