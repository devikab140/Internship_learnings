from PyPDF2 import PdfReader
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')


# Read PDF
reader = PdfReader("C:/Users/devik/Downloads/harrypotter.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Lowercase
text = text.lower()

# Remove punctuation & numbers
text = re.sub(r'[^a-z\s]', '', text)

# Tokenization
words = text.split()

# Stopwords removal + Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

processed_words = [
    lemmatizer.lemmatize(word)
    for word in words
    if word not in stop_words
]

# For vectorizers (BoW, TF-IDF)
processed_text = " ".join(processed_words)
# For Word2Vec / GloVe (sentence-wise)
sentences = [processed_words]


#---------- BoW---------------------#
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform([processed_text])

bow_array = X_bow.toarray()
# print(bow_array)
feature_names = bow_vectorizer.get_feature_names_out()

print("Bag of Words:")
for word, count in zip(feature_names, bow_array[0]):
    print(f"{word}: {count}")

#---------- TF-IDF-------------------#

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform([processed_text])

tfidf_array = X_tfidf.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF:")
for word, score in zip(feature_names, tfidf_array[0]):
    print(f"{word}: {score:.5f}")

#------------ word2vec--------------------

from gensim.models import Word2Vec

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4
)

# Example: vector for a word
word = "harry"
if word in w2v_model.wv:
    print("\nWord2Vec vector for 'harry':")
    print(w2v_model.wv[word])
else:
    print("Word not found in vocabulary")

# n gram, glove
