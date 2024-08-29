import pandas as pd
import numpy as np
import seaborn as sborn
import matplotlib.pyplot as mpp

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer as tkz
from tensorflow.keras.preprocessing.sequence import pad_sequences as ps
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('emails.csv')
###print(data.head(3))
###print(data.shape)

sborn.countplot(x = 'enron', data=data)
mpp.show()