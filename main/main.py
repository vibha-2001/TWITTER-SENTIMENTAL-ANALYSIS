''''
This code is written and maintained by Vibha Pandey
email: vibhapandey2001@gmail.com
github: https://github.com/vibha-2001
'''


import pandas as pd
import re
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from model import RecurrentNN as RNN
from keras.layers import SimpleRNN
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

#########################################################################################
stopword = set(stopwords.words('english'))
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


# Loading the Dataset
print("Loading input data...")
dataset = pd.read_csv('Dataset/Tweets.csv')

print(dataset.describe())

print(dataset.shape)

#dropping null values
dataset.dropna(inplace=True)
original_df = dataset.copy()

dataset['text'] = dataset['text'].astype(str)

print('Iitializing removing noise from text')


def clean_text(text):
    '''Make text lowercase,
    remove links, remove new line,
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('@\w+', '', text)
    text = re.sub('#\w+', '', text)
    text = re.sub('[^\w\s@]', '', text)
    text = re.sub('\d', '', text)
    text = re.sub('im ', '', text)
    text = re.sub('\s+', ' ', text).strip()
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

#removing noise from text
dataset['text'] = dataset['text'].apply(lambda x: clean_text(x))
print(dataset['text'][2])

#removing unecessary words
frequent_word =["I'm", "-", "****", "&", "mmmmmmmm", "phew", "hehe", "ugh"]
dataset['text']= dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in frequent_word))

#removing stopwords
dataset['text'] = dataset['text'].apply(lambda x:" ".join(term for term in x.split() if term not in stopword)) # clean train

# Using Lemmatization
wn = WordNetLemmatizer()
dataset['text'] = dataset['text'].apply(lambda x:" ".join([wn.lemmatize(word) for word in x.split()]))

#stemming
stemmer = PorterStemmer()
clean_train = dataset['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

print(clean_train.shape)
print("cleaning completed")

print(clean_train[2])

#########################################END OF CLEANING DATASET#################################


X = clean_train
print(X.head())

#convert label string to categorical
dataset['label_id'] = dataset['sentiment'].factorize()[0]
cat_id = dataset[['sentiment', 'label_id']].drop_duplicates().sort_values('label_id')
cat_to_id = dict(cat_id.values)
id_to_cat = dict(cat_id[['label_id', 'sentiment']].values)

print(id_to_cat)

label = dataset['label_id'].values

print(dataset)

max_features = 8000

#tokenizing
tokenizer = Tokenizer(num_words= max_features, oov_token='OOV',filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~')
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index

max_sequence_length = max([len(x) for x in X])
print(max_sequence_length)

X = tokenizer.texts_to_sequences(X)


def pad_sequences(sequences, max_seq_length):
    padded_sequences = np.zeros((len(sequences), max_seq_length))
    for i, seq in enumerate(sequences):
        if len(seq) >= max_seq_length:
            padded_sequences[i, :] = seq[:max_seq_length]
        else:
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences


#padding
X = pad_sequences(X, max_sequence_length)
Y = to_categorical(label, num_classes = 3)

print("Shape of data tensor ['text']:", X.shape)
print("Shape of data tensor ['label']:", Y.shape)


# Split dataset into train and test sets
print("Generate Training and Testing Data")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Loading Model")

# TODO: add model to separate folder #
model = RNN.RecurentNeuralNetwork()

n_batch = 64
n_epochs = 12

history = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=n_batch)

#print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Plot training & validation loss values
# Save loss plot as an image
fig1 = plt.gcf()
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
fig1.savefig('../Loss_Graph.png')
plt.show()
plt.close('all')


# Plot training & validation accuracy values
# Save accuracy plot as an image
fig2 = plt.gcf()
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
fig2.savefig('../Accuracy.png')  # Save the plot as an image
plt.show()

print("Training complete")

model.model.summary()

#saving the model
model.model.save("../Trained model/Tweets.h5")

print("Testing model")
# Testing the model
model.evaluate(X_test, Y_test)

print("Testing complete")