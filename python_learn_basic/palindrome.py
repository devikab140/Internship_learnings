# get input from user
# check word from left to right is equal to right to left
# iterate through word , reverse the word and store into a new variable(rev_word)
#check the word and rev_word are same 
#if they same print palindrome

'''
def palindrome(word):
    rev_word=""
    for i in range(len(word)-1,-1,-1):
        rev_word+=word[i]
    if word==rev_word:
        return "It is a palindrome"
    else:
        return "It's not a palindrome"
word=input("Enter the word: ").lower()
print(palindrome(word))


'''
def palindrome(word):
    if word==word[::-1]:
        return "It's palindrome"
    else:
        return "It's not a palindrome"
    

word=input("Enter a word: ")      # .lower() is not used since AMma not same
print(palindrome(word))
    
    