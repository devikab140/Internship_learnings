# clrs=['orange','red','blue','black','pink','white','blue','meroon','violet','blue','peach','blue','green']
# occurence=clrs.count("blue")
# print(occurence)
# count=0
# if clrs[0] == 'blue':
#     count+=1
#     if count == occurence :
#         print("blue's index is zero")



# word="mlayalam"
# print(len(word))
# print(word[::2])
# print(word[1:7:2])
# print(word[0::2])
# print(word[::-1])


#  for finding sqrt of a number
# def sqrt(n):
#     return n**(1/2)
# print(sqrt(24))
numbers=[int(x) for x in input("Enter the numbers with space: ").split()]
max_no=0
for i in range(len(numbers)+1):
    for j in range(i+1,len(numbers)+1):
        if i>j:
            max_no=numbers[i]
        else:
            max_no=numbers[j-1]

min_no=0
for i in range(len(numbers)+1):
    for j in range(i+1,len(numbers)+1):
        if i<j:
            min_no=numbers[i]
        else:
            min_no=numbers[j-1]

print(min_no)