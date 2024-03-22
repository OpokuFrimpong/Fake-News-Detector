import streamlit as st
import numpy as np
import pandas as pd
import json
import csv
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
tf.disable_eager_execution()
from gensim.models import KeyedVectors

# Reading the data 
data = pd.read_csv("news.csv") 

data = data.drop(["Unnamed: 0"], axis=1) 


X = st.text_input("Enter text to verify")

if X:	# Check if the file is not empty
	# encoding the labels 
	le = preprocessing.LabelEncoder() 
	le.fit(data['label']) 
	data['label'] = le.transform(data['label']) 

	embedding_dim = 50
	max_length = 54
	trunc_type = 'post'
	padding_type = 'post'
	#This is a token used to represent out-of-vocabulary (OOV) words. These are words that are not found in the vocabulary during tokenization. 
	#Using <OOV> as the token allows the model to handle words that it hasn't seen before.
	oov_tok = "<OOV>"
	training_size = 3000
	test_portion = .1

	title = [] 
	text = [] 
	labels = [] 
	for x in range(training_size): 
		title.append(data['title'][x]) 
		text.append(data['text'][x]) 
		labels.append(data['label'][x]) 


	#Building the embeddings
	tokenizer1 = Tokenizer() 
	tokenizer1.fit_on_texts(title) 
	word_index1 = tokenizer1.word_index 
	vocab_size1 = len(word_index1) 
	sequences1 = tokenizer1.texts_to_sequences(title) 
	padded1 = pad_sequences( 
		sequences1, padding=padding_type, truncating=trunc_type) 
	split = int(test_portion * training_size) 
	training_sequences1 = padded1[split:training_size] 
	test_sequences1 = padded1[0:split] 
	test_labels = labels[0:split] 
	training_labels = labels[split:training_size] 


	# Open the original GloVe vectors file
	with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
		lines = f.readlines()

	# Insert header line with vocabulary size and vector size
	lines.insert(0, f'{len(lines)} 50\n')

	# Save the modified file
	with open('corrected_glove.6B.50d.txt', 'w', encoding='utf-8') as f:
		f.writelines(lines)

	# Load GloVe vectors with explicit encoding
	embeddings_index = {} 
	with open('corrected_glove.6B.50d.txt', encoding='utf-8') as f: 
		for line in f: 
			values = line.split() 
			word = values[0] 
			coefs = np.asarray(values[1:], dtype='float32') 
			embeddings_index[word] = coefs
			
	# Generating embeddings 
	embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim)) 
	for word, i in word_index1.items(): 
		embedding_vector = embeddings_index.get(word) 
		if embedding_vector is not None: 
			embeddings_matrix[i] = embedding_vector 


	#Buildidng architecture 
	model = tf.keras.Sequential([ 
		tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, 
								input_length=max_length, weights=[ 
									embeddings_matrix], 
								trainable=False), 
		tf.keras.layers.Dropout(0.2), 
		tf.keras.layers.Conv1D(64, 5, activation='relu'), 
		tf.keras.layers.MaxPooling1D(pool_size=4), 
		tf.keras.layers.LSTM(64), 
		tf.keras.layers.Dense(1, activation='sigmoid') 
	]) 
	model.compile(loss='binary_crossentropy', 
				optimizer='adam', metrics=['accuracy']) 

	num_epochs = 10

	training_padded = np.array(training_sequences1) 
	training_labels = np.array(training_labels) 
	testing_padded = np.array(test_sequences1) 
	testing_labels = np.array(test_labels) 

	history = model.fit(training_padded, training_labels, 
						epochs=num_epochs, 
						validation_data=(testing_padded, 
										testing_labels), 
						verbose=2) 

	# sample text to check if fake or not 

	# detection 
	sequences = tokenizer1.texts_to_sequences([X])[0] 
	sequences = pad_sequences([sequences], maxlen=54, 
							padding=padding_type, 
							truncating=trunc_type)

	if(model.predict(sequences, verbose=0)[0][0] >= 0.5): 
		st.write("This news is True")
	else:
		st.write("This news is false") 
else:
    st.write("Please input some text")
