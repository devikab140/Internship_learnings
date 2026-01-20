# BoW on a paragraph
def bag_of_words(paragraph):
    # Importing the libraries
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    import re
    import numpy as np

    # # Download NLTK resources
    # nltk.download('stopwords')
    #printing the stopwords
    print("Stopwords:", stopwords.words('english'))
    # Step 1: Text Preprocessing
    # a. Lowercasing
    paragraph = paragraph.lower()

    # b. Removing Punctuation and Non-alphabetic characters
    paragraph = re.sub(r'[^a-z\s]', '', paragraph)

    # c. Tokenization
    words = paragraph.split()

    # d. Removing Stopwords and lemmatization
    lz = WordNetLemmatizer()
    processed_words = [lz.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]

    # Reconstruct the processed paragraph
    processed_paragraph = ' '.join(processed_words)
    print("Processed Paragraph:", processed_paragraph)
    # Step 2: Creating the Bag of Words model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([processed_paragraph]).toarray()

    return X, vectorizer.get_feature_names_out()
# Example usage
if __name__ == "__main__":
    paragraph = """I only have a day-off on Sunday, so I have only a little free time. 
    Sunday is a wonderful day for me to spend time with my friend. One of the things I 
    really enjoy doing on Sunday morning is to play chess. 
    It is time to relax and talk about the events of the previous week and future plans.
    On Sunday morning, I often sing with my friends at a karaoke restaurant and we 
    all have a good time there. It is especially funny. When I get tired, I stop singing.
      Please don't tell him I said this, but he is a very bad singer!
    Once in a while, I go for a walk on Sundays with my friends. Sometimes, 
    I just stay at home and listen to music, watch television, or read novels. 
    Do you feel bored when you hear about my free time, teacher? """
    
    bow_vector, feature_names = bag_of_words(paragraph)
    # Print each word and its corresponding count
    for word, count in zip(feature_names, bow_vector[0]):
        print(f"{word}: {count}")
    
       