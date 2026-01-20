line="The quick brown fox jumps over the lazy dog"
words=line.split()
unigrams=[]
for word in words:
    unigrams.append(word)
print("Unigrams:",unigrams)

#bigrams
bigrams=[]
for i in range(len(words)-1):
    bigrams.append((words[i]+" "+words[i+1]))
print("Bigrams:",bigrams)

#trigrams
trigrams=[]
for i in range(len(words)-2):
    trigrams.append((words[i]+" "+words[i+1]+" "+words[i+2]))
print("Trigrams:",trigrams)


# class NGramGenerator:
#     def __init__ (self,text):
#         self.text=text
#         self.words=self.text.split()
#         self.n=len(self.words)
#     def generate_ngrams(self,n):
#         if n == 1:
#             return self.words
#         elif n == 2:
#             return [self.words[i] + " " + self.words[i+1] for i in range(self.n-1)]
#         elif n == 3:
#             return [self.words[i] + " " + self.words[i+1] + " " + self.words[i+2] for i in range(self.n-2)]
#         else:
#             return []
        
# line="The quick brown fox jumps over the lazy dog"
# print("Using NGramGenerator class:")
# ngram_gen=NGramGenerator(line)
# print(ngram_gen.generate_ngrams(1))  # Unigrams
# print(ngram_gen.generate_ngrams(2))  # Bigrams
# print(ngram_gen.generate_ngrams(3))  # Trigrams
