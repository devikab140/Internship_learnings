# try:
#     age=int(input("Enter your age: "))
#     print(f"you are {age} years old")
# except ValueError:         # if do not know the name 
#     print("Enter a valid number")





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
try:          # used here since value taking here , we have to check the number also so we started try here
    a=float(input("Enter the first number: "))
    b=float(input("Enter the second number: "))
    operation=input("Enter the operation you have to execute: ")

    if operation==("+"):
        print(addition(a,b))
    elif operation== "-":
        print(subtraction(a,b))
    elif operation==("/"):
        print(division(a,b))
    elif operation==("*"):
        print(multiplication(a,b))
    elif operation==("%"):
        print(modulus(a,b))
    elif operation==("**"):
        print(power(a,b))
    elif operation==("//"):
        print(floor(a,b))
    else:
        print("Invalid operation")
except ZeroDivisionError:                # here we can type only ---> except: ---> it will check with all errors and print the below one
    print("Zero can't be second number")  
except ValueError:
    print("Invalid number")
else:                 # run only when except block don't run
    print("Operation done")
finally:                # run always
    print("End")


#except:            it take all errors 
    # print("Invalid")




a=int(input("Enter the index number: "))
b=input("Enter the key: ")
try:
    numbers=[1,2,3,4,5]
    dict={"a":1,"b":2}
    print(numbers[a])
    print(dict[b])
except IndexError:        # here if invalid index occured even when the keyerror is there which do not proceed further , that is key error not show , if we correct the index then if key error occur then the exception will run
    print("Index out of range")
except KeyError:
    print("Invalid key")

# single line exception
except(IndexError,KeyError):
    print("")


