# get a string and create a dict with unique elements of the string as key and occurence as value
# create one empty dict
# check letter present or not . not present add the letter as key
# if present the count increa
strng=input("Enter a string: ")
empty={}
for i in strng:
    if i not in empty:
        empty[i]=1
    else:
        empty[i]=empty.get(i)+1
        #empty[i]+=1 /empty[i]=empty[i]+1        
print(empty)