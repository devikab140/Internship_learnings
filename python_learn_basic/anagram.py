# get two word from user
# check two words have same letters
# 1) first take each letters of one word and store to a list 
# 2)take second word letter , check it is in the list or not
# 3) take a count of letters match , if count = len(first word) ---> it is anagram
# two have same letters that repete same no of times 

word_1=input("Enter the first word: ").lower()
word_2=input("Enter the second word: ").lower()
letters=[]   # r a c e e,
count=0
for i in word_1:
    letters.append(i)
if len(word_2)==len(letters):
    for x in word_2:
        if x in letters:
            count+=1
if count==len(word_1):
    print(f"{word_1} and {word_2} are anagrams")
else:
    print(f"{word_1} and {word_2} are not anagrams")
# take two count , 

#####################################################################################################33

word_1=input("Enter the first word: ").lower()
word_2=input("Enter the second word: ").lower()
if sorted(word_1)==sorted(word_2):
    print("Anagrams")
else:
    print("Not anagrams")




###################################################


word_1=input("Enter the first word: ").lower()
word_2=input("Enter the second word: ").lower()
if len(word_1)!=len(word_2):
    print("Not anagram")
else:
    for ch in word_1:
        if word_1.count(ch)!=word_2.count(ch):
            print("Not anagram")
            break
    else:
        print("It is anagram")




# select max(salary)
# from table
# where salary < (select max(salary) from table)


# list=[1,2,3,4,3,2]

# unique=[]
# for  x in list:
#     if x not in unique:
#         unique.append(x)
    