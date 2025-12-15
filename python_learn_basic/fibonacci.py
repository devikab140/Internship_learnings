# get limit from user (how many elements have to print)
# print 0 and 1 then add 0&1 print those value 
# add 3 rd and second value ...
n=int(input("Enter the value: "))
a=0
b=1
for i in range(n):
   c=a+b
   print(a,end=" ")
   a=b
   b=c



n=int(input("Enter the value: "))
a=0
b=1
for i in range(n):
   print(a,end=" ")
   a,b=b,a+b
