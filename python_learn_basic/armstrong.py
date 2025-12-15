# first get a number
# count the no:of digit
# each digit^no:of didgit find and add
#if sum==number ---. armstrong


number=input("Enter a number: ")
n=len(number)
sum=0
for i in number:
    sum+=int(i)**n
if sum==int(number):
    print("armstrong")
else:
     print("Not armstrong")


# upto a limit we have to find all anagram numbers

limits=int(input("Enter a number: "))
armstrong=[]
for x in range(limits):
    n=len(str(x))
    sum=0
    for i in str(x):
      sum+=int(i)**n
    if sum==x:
       armstrong.append(x)
    else:
       continue
print(armstrong)