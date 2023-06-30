# Nicole Joseph
# NLP Final Project

import tensorflow as tf;
#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import re
import string
import nltk
import argparse
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import csv
#print('Hi')

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--input1',
                    default = max,
                    help = 'Please enter name of the file with training data', 
                    required = True)

#parser.add_argument('--input2',
                    #default = max,
                    #help = 'Please enter name of the file with testing data', 
                    #required = True)

args = parser.parse_args()

# https://www.kaggle.com/c/nlp-getting-started : NLP Disaster Tweets
#trainingFile = open(args.input, 'r')
df = pd.read_csv(args.input1) # use pandas to load the data

# shape attribute stores stores the number of rows and columns as a tuple
#print(df.shape)
# df. head() Returns the first 5 rows of the dataframe
#print(df.head()) 

#print((df.target == 1).sum()) # Disaster 
#print((df.target == 0).sum()) # No Disaster \# remove URL using regular expressions
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

#print(string.punctuation)
# output shows punctuation characters that will be removed: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

#pattern = re.compile(r"https?://(\S+|www)\.\S+")
#for t in df.text:
    #matches = pattern.findall(t)
    #for match in matches:
        #print(t)
        #print(match)
        #print(pattern.sub(r"", t))
    #if len(matches) > 0:
        #break

# remove URL and punctuation from the data
df["text"] = df.text.map(remove_URL) 
df["text"] = df.text.map(remove_punct)

# remove stopwords
stop = set(stopwords.words("english"))

# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)
#print(stop)

df["text"] = df.text.map(remove_stopwords)
#print(df.text)

# Now, prepare the text for the RNN
# Use the Counter object to count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(df.text)
#print(len(counter))
#print(counter)
#print(counter.most_common(5))

num_unique_words = len(counter)

# Split dataset into training and validation/tuning set
train_size = int(df.shape[0] * 0.8) # 80% of the dataset is used for training and 20% of the dataset is for validation

train_df = df[:train_size]
val_df = df[train_size:]

# Get training sentences and training labels
train_sentences = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
# Get validation sentences and validation labels
val_sentences = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()
#print(train_sentences.shape)
#print(val_sentences.shape)

# Use keras module to Tokenize
# Tokenize: vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words = num_unique_words) # create tokenizer object
tokenizer.fit_on_texts(train_sentences) # fit only to training data
# Let each word have a unique index after tokenization
word_index = tokenizer.word_index
#print(word_index)

# Tokenize training data and validation data
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
# compare 5 samples of the training sentences to corresponding training sequences
#print(train_sentences[10:15])
#print(train_sequences[10:15])

# Pad the sequences to have the same length
# Set the max number of words in a sequence
max_length = 20
train_padded = pad_sequences(train_sequences, maxlen = max_length, padding = "post", truncating = "post") # pad with zeroes
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = "post", truncating = "post")
#print(train_padded.shape, val_padded.shape)
#print(train_sentences[10])
#print(train_sequences[10])
#print(train_padded[10])

# Check reversing the indices
# Create dictionary and flip (key is the index, value is the word)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
#print(reverse_word_index)
# Find a corresponding word for each index
def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])
#decoded_text = decode(train_sequences[10])
#print(train_sequences[10])
#print(decoded_text)
#print(train_sentences[10])

# Create LSTM model; import layers from tensoflow.keras
# Word Embeddings: https://www.tensorflow.org/tutorials/text/word_embeddings
model = keras.models.Sequential() # Sequential model
model.add(layers.Embedding(num_unique_words, 32, input_length = max_length)) # 32 is a size that we can specify
# Apply LSTM/RNN layer 
model.add(layers.LSTM(64, dropout = 0.2)) # 64 is the specified number of output units; dropout is 10%
model.add(layers.Dense(1, activation = "sigmoid")) # Add a dense layer with 1 output since we want a 0 or 1 classification at the end
print(model.summary())

loss = keras.losses.BinaryCrossentropy(from_logits = False) # BinaryCrossentropy since w used binary classification
optim = keras.optimizers.Adam(lr = 0.01) # Use an optimizer
metrics = ["accuracy"] # Define the metrics that we want to track
model.compile(loss = loss, optimizer = optim, metrics = metrics)

# During training, automatically use the validation set to fine tune
model.fit(train_padded, train_labels, epochs = 20, validation_data = (val_padded, val_labels), verbose = 2)

predictions = model.predict(train_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]
print(train_sentences[50:75])
print(train_labels[50:75])
print(predictions[50:75])

# Process Testing Data
#df2 = pd.read_csv(args.input2) 

#df2["text"] = df2.text.map(remove_URL) 
#df2["text"] = df2.text.map(remove_punct)
#stop = set(stopwords.words("english"))

#def counter_word2(text_col):
    #count2 = Counter()
    #for text in text_col.values:
        #for word in text.split():
            #count2[word] += 1
    #return count2
#counter2 = counter_word2(df2.text)
#num_unique_words2 = len(counter2)

#test_size = int(df2.shape[0] * 1)
#test_df = df2[:test_size]

#test_sentences = test_df.text.to_numpy()

#tokenizer = Tokenizer(num_words = num_unique_words) 
#tokenizer.fit_on_texts(test_sentences) 
#word_index = tokenizer.word_index
#test_sequences = tokenizer.texts_to_sequences(test_sentences)
#max_length = 20
#test_padded = pad_sequences(test_sequences, maxlen = max_length, padding = "post", truncating = "post") 

#predictions = model.predict(test_padded)
#predictions = [1 if p > 0.5 else 0 for p in predictions]
#output_filename = input("Please enter the name of the output file: ")
#output_file = open(output_filename, "w")
#writer = csv.writer(output_file)
#writer.writerow(predictions[0:20])
#output_file.close()

# print(predictions[0:20])


