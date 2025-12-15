# n=int(input("Enter the number: "))
# fact=1
# for i in range(1,n+1):
#     fact*=i
# print(f"Factorial of {n} is {fact}")

def factorial(n):
    if n<0:
        return "Negative number have no factorial"

    elif(n==0 or n==1):
        return 1
    else:
        return n*factorial(n-1)
n=n=int(input("Enter the number: "))
print(f"Factorial of {n} is {factorial(n)}")