# get a number
#check prime or not
#check the number less than a nmber divide or not

n=float(input("Enter a positive number: "))
count=0
if n<1:
    print("Give a valid number , only positive number")
elif n==1:
    print(f"{n} is not a prime number")
elif(int(n)!=n):
    print("Give a valid number , only positive number")
else:
    for i in range(1,int(n+1)):
        if n%i==0:
            count+=1
    if count>2:
        print(f"{n} is not a prime number")  
    else:
        print(f"{n} is a prime number") 


#########################################################################################
# standard method , iterate until the sqrt(n)

n=float(input("Enter a positive number: "))
count=0
if n<=1:
    print("Prime numbers are whole numbers starting from 2 , give another number")
elif(int(n)!=n):
    print("Give an integer number")
else:
    for i in range(1,(int(n**0.5))+1):
        if n%i==0:
           count+=1
    if count>=2:      # here equal To 2 is since we are checking upto square root, so for a prime number only factor is 1, so >=2 is not prime
        print(f"{n} is not a prime number")  
    else:
        print(f"{n} is a prime number")    