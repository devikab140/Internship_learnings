# def area(l,b):
#     A=l*b
#     return A

# print(f"Area of the rectangle is {area(5,4)}")

#------------------------------------------------------------------------------------------#


#define functions for calculation
# get input fron user and call the function then do calculation


def addition(a,b):
    sum=a+b
    return sum
def subtraction(a,b):
    diff=a-b
    return diff
def multiplication(a,b):
    product=a*b
    return product
def division(a,b):
    quotient=a/b
    return quotient
def modulus(a,b):
    mod=a%b
    return mod
def power(a,b):
    pwr=a**b
    return pwr
def floor(a,b):
    floor_div=a//b
    return floor_div

a=float(input("Enter the first number: "))
b=float(input("Enter the second number: "))
operation=input("Enter the operation you have to execute: ")
#operation=optn.lower()
if operation==("+"):
    print(addition(a,b))
elif operation== "-":
    print(subtraction(a,b))
elif operation==("/"):
    if b!=0:
        print(division(a,b))
    else:
        print("Cannot divide by zero")
elif operation==("*"):
    print(multiplication(a,b))
elif operation==("%"):
    if b!=0:
        print(modulus(a,b))
    else:
        print("Cannot divide by zero")
elif operation==("**"):
    print(power(a,b))
elif operation==("//"):
    if b!=0:
        print(floor(a,b))
    else:
        print("Not possible")
else:
    print("Incorrect format")



