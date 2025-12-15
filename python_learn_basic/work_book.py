# print(list(range(-5,-10,-1)))

'''
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
optn=input("Enter the operation you have to execute: ")
operation=optn.lower()
if operation==("+") or operation=="addition":
    print(addition(a,b))
elif operation== "-" or operation=="subtraction":
    print(subtraction(a,b))
elif operation==("/")or operation=="division":
    if b!=0:
        print(division(a,b))
    else:
        print("Cannot divide by zero")
elif operation==("*")or operation=="multiplication":
    print(multiplication(a,b))
elif operation==("%")or operation=="modulus":
    if b!=0:
        print(modulus(a,b))
    else:
        print("Cannot divide by zero")
elif operation==("**")or operation=="power":
    print(power(a,b))
elif operation==("//")or operation=="floor division":
    if b!=0:
        print(floor(a,b))
    else:
        print("Not possible")
else:
    print("Incorrect format")

'''


# add=lambda x,y:x+y
# print(add(3,5))

# #square root without using a built in function

# def sqrt(n):
#     return n**0.5
 


  
new_list=[]
for x in lists:
    for y in x:
        if y <=9 and y>=0:
            if int(x)==x:
                new_list.append(x)
print(new_list)



###############################################################

lists=[x for x in input("Enter the input: ").split()]  
#print(lists)
new_l=[]
for x in lists:
    try:
        z=int(x)  
        #print(z)
        new_l.append(z)
    except:
        continue
print(new_l)
  

