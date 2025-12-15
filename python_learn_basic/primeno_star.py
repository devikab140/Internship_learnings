def primelist(n):
    count=0
    primes=[]
    for x in range(2,n+1):
        count=0
        for i in range(1,(int(x**0.5))+1):
            if x%i==0:
                count+=1
        if count>=2:     
            continue
        else:
            primes.append(x)    
    return primes

n=int(input("Enter the number: "))
Primes=primelist(n)
row_width=1
for i in range(1,n+1):
   if i in Primes:
         # print(" "*(n-i),"*"*i,end="")
         for j in range(n-i):
             print(" ",end="")
         for j in range(row_width):
             print("*",end=" ")
             
   else:
       # print(" "*(n-i),str(i)*i,end="")
       for j in range(n-i):
             print(" ",end="")
       for j in range(row_width):
             print(str(i),end=" ")
   row_width+=1
   print()
   



   
   