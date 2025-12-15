# from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# # --- OneHotEncoder Example ---
# colors = [['Red'], ['Blue'], ['Green'], ['Red']]
# onehot = OneHotEncoder(sparse_output=False)
# print("OneHot Encoding:\n", onehot.fit_transform(colors))

# --- CountVectorizer Example ---
corpus = ["I love apples", "I hate bananas"]
count_vec = CountVectorizer()
print("\nCount Vectorizer:\n", count_vec.fit_transform(corpus).toarray())
print("Features:", count_vec.get_feature_names_out())

# --- TF-IDF Example ---
tfidf = TfidfVectorizer()
print("\nTF-IDF Vectorizer:\n", tfidf.fit_transform(corpus).toarray())
print("Features:", tfidf.get_feature_names_out())

# --- HashingVectorizer Example ---
hash_vec = HashingVectorizer(n_features=8)
print("\nHashing Vectorizer:\n", hash_vec.transform(corpus).toarray())
