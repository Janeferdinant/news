"""Turn one article into the feature vector the classical models were trained on.

This repeats the notebook's preprocessing (sections 3 and 5): clean the title and text,
TF-IDF (vectorizer fitted in the notebook on training articles only), three engineered
features, then keep the columns chosen by ACO.
"""
import re

import numpy as np

ENGINEERED = ["text_length", "title_length", "sentiment"]

_stop_words = None


def _nltk_resources():
    global _stop_words
    if _stop_words is None:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def clean_text(text):
    """Lower-case, keep letters only, tokenise, drop English stop words (as in the notebook)."""
    from nltk.tokenize import word_tokenize

    stop_words = _nltk_resources()
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return " ".join(w for w in word_tokenize(text) if w not in stop_words)


def engineered_features(title, text):
    from textblob import TextBlob

    return [len(text.split()), len(title.split()), TextBlob(text).sentiment.polarity]


def full_feature_vector(title, text, vectorizer):
    """TF-IDF plus engineered features for one article, shape (1, n_features)."""
    clean = clean_text(title + " " + text)
    tfidf = vectorizer.transform([clean]).toarray()
    return np.hstack((tfidf, np.array([engineered_features(title, text)])))


def feature_names(vectorizer):
    return np.concatenate((vectorizer.get_feature_names_out(), ENGINEERED))
