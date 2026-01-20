# # Text preprocessing on a paragraph without using any external libraries
def preprocess_text(paragraph):
    # Convert to lowercase
    paragraph = paragraph.lower()
    
    # Remove punctuation
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuation:
        paragraph = paragraph.replace(char, "")
    
    # Tokenize the text into words
    words = paragraph.split()
    
    # Remove stop words
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with", 
        "about",  "against","between","into","through","during","before","after",
        "above","below","to","from","up","down","in","out","on","off","over","under",
        "again","further","then","once","here","there","when","where","why","how",
        "all","any","both","each","few","more","most","other","some","such","no",
        "nor","not","only","own","same","so","than","too","very","s","can","t",
        "will","just","don","should","now"])
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


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

print(preprocess_text(text))




# lowered_text = text.lower()
# punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
# for char in punctuation:
#     lowered_text = lowered_text.replace(char, "") 
# tokens = lowered_text.split()
# print(tokens)
# stop_words=set([
#         "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
#         "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
#         "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
#         "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
#         "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
#         "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
#         "because", "as", "until", "while", "of", "at", "by", "for", "with", 
#         "about",  "against","between","into","through","during","before","after",
#         "above","below","to","from","up","down","in","out","on","off","over","under",
#         "again","further","then","once","here","there","when","where","why","how",
#         "all","any","both","each","few","more","most","other","some","such","no",
#         "nor","not","only","own","same","so","than","too","very","s","can","t",
#         "will","just","don","should","now"])    
# filtered_words = [word for word in tokens if word not in stop_words]
# print(filtered_words)

  