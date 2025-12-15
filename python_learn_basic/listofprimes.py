n=int(input("Enter a number: "))
primes=[]
for x in range(2,n):
    count=0
    for i in range(1,(int(x**0.5))+1):
        if x%i==0:
           count+=1
    if count>=2:     
        continue
    else:
        primes.append(x)    
print(primes)