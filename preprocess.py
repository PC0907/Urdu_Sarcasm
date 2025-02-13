import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import urduhack

from transformers import AutoTokenizer, XLMRobertaModel

import torch
from urduhack.preprocessing.util import (
    replace_numbers,
    remove_accents,
    normalize_whitespace,
    replace_phone_numbers,
    replace_emails,
    replace_urls,
)

from tqdm import tqdm
tqdm.pandas()

from LughaatNLP import LughaatNLP

df = pd.read_csv(r"C:\Users\mkaab\OneDrive\Desktop\research\Urdu Sarcastic Tweets Dataset\Urdu Sarcastic Tweets Dataset\urdu_sarcastic_dataset.csv", engine='python')
new_df = df[['urdu_text', 'is_sarcastic']].copy()
df = new_df.dropna()



urdu_text_processing = LughaatNLP()


def preprocess_urdu_text(text):
    if isinstance(text, str):
        text = urduhack.preprocessing.util.replace_currency_symbols(text)
        text = urduhack.preprocessing.util.replace_numbers(text)
        text = urduhack.preprocessing.util.remove_accents(text)
        text = urduhack.preprocessing.util.normalize_whitespace(text)
        text = urduhack.preprocessing.util.replace_phone_numbers(text)
        text = urduhack.preprocessing.util.replace_emails(text)
        text = urduhack.preprocessing.util.replace_urls(text)
        text = urdu_text_processing.normalize(text)
        text = urdu_text_processing.lemmatize_sentence(text)
        text = urdu_text_processing.urdu_stemmer(text)
        text = urdu_text_processing.remove_stopwords(text)        
        return text


df['Preprocessed'] = df['urdu_text'].progress_apply(preprocess_urdu_text)
