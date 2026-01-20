#using the nltk library
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def preprocess_text(paragraph):
    # Convert to lowercase
    paragraph = paragraph.lower()
    
    # Remove punctuation
    paragraph = paragraph.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text into words
    words = word_tokenize(paragraph)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
   # 6. Stemming
    words = [stemmer.stem(word) for word in filtered_words]
    # 5. Lemmatization
    words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    
    
    return words
   


text="Nature is the beautiful and life-giving force that surrounds us and supports all " \
"living beings on Earth. It includes forests, rivers, mountains, oceans, animals, plants," \
" and even the air we breathe. Nature works in perfect balance, providing food, water," \
" shelter, and energy to humans and other organisms. The green trees give us oxygen and help " \
"purify the air, while rivers and rain supply fresh water essential for life. Animals and " \
"plants depend on each other in complex ecosystems, showing how deeply everything in nature " \
"is connected. Nature also has a calming effect on the human mind; spending time in natural" \
" surroundings reduces stress and brings peace and happiness. However, due to pollution," \
" deforestation, and excessive use of natural resources, nature is being harmed." \
" Climate change, loss of wildlife, and environmental imbalance are serious warnings " \
"that we must act responsibly. Protecting nature is not just an option but a duty. " \
"By conserving forests, reducing pollution, and using resources wisely, we can preserve " \
"nature for future generations and ensure a healthy and sustainable planet."

print("Word after preproces ,nltk",preprocess_text(text))

#using the spacy library
import spacy
def preprocess_text_spacy(paragraph):
    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the paragraph
    doc = nlp(paragraph.lower())
    
    # Remove stop words and punctuation, and lemmatize the words
    filtered_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return filtered_words
print("Word after lemmatization by spacy",preprocess_text_spacy(text))