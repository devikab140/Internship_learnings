numbers=map(int,input("enter the numbers with space: ").split())    #here function is int, no need of int()
#num=list(map(int,numbers))
n=list(map(lambda x:x**3,numbers))
print(n)