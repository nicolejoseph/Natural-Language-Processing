# Text Classification of Tweets Using RNN
The objective of this program is to implement and train a neural net using Python, TensorFlow, and Keras to complete the NLP task of text classification 
of tweets as disaster (1) or no disaster (0). The dataset used was the NLP Disaster Tweets dataset which contains a total of 7,613 tweets. Included is my final project report, 
for a more in-depth look at my final results and how this project was developed!
## Preprocessing
First, the text data was cleaned and preprocessed. URLs were removed using regular expressions. Punctuation was removed, 
and so were stopwords (using the nltk toolkit).

## Tokenization
The Keras module was used to tokenize the training data, or vectorize the text corpus by turning each text into a sequence of integers. Zero padding was utilized 
to ensure that all sequences had the same length. Later on, a dictionary was also created, which checked the reversal of indices.
## Model Architecture
For the machine learning part, a sequential model was set up with imported layers from tensorflow.keras. The model consists of an Embedding layer, 
LSTM layer, and a Dense layer. LSTM (Long Short-Term Memory) are a variety of recurrent neural networks (RNNs) capable of learning long-term dependencies, 
especiallyin sequence prediction problems. Binary cross entropy function for loss was used for this binary classification task. Additionally,
the adam optimizer was employed.

