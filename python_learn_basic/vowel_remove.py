#get a string
#check each word vowels or not
#index from 0 to length of string 
#if vowel count and remove
#not a vowel then keep it

WORD=input("Enter the word: ") # here also we can give --> wrod=input("").lower()
word=WORD.lower()
non_vowel=" "
vowels=["a","e","i","o","u"]
count=0

for i in range(len(word)):     # direct we can give ---> for i in word
    if word[i] in vowels:
        count+=1
        word.replace(word[i],"") # no use , since we using else
    else:
        chrctr=word[i]
        non_vowel+=chrctr          # since we have to add and store in that non_vowel
print(f"word without vowels is {non_vowel}")
print(f"count of vowels {count}")

















word=input("Enter the word: ").lower() # here also we can give --> wrod=input("").lower()
non_vowel=""
vowels=["a","e","i","o","u"]
count=0
for i in range(len(word)):     # direct we can give ---> for i in word
    if word[i]>="0" and word[i]<="9":
       break 
    elif word[i] in vowels:
        count+=1
    else:
        non_vowel+=word[i]   # since we have to add and store in that non_vowel

 
if count >0 or word==non_vowel: 
    print(f"word without vowels is: {non_vowel}")
    print(f"count of vowels: {count}")
else:
    print("Invalid , give a word") 
