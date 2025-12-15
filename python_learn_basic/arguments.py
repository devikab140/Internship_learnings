#positional
def area(l,b):
    return l*b
print(area(10,5))

def greetings(name,message):
    print(f"Hello {name} , {message}")
greetings("Devika","Welcome")
greetings("Welcome","Devika")
#keyword arguments
def area(l,b):
    return l*b
print(area(b=5,l=10))

#position and keyword
def area(l,b):
    return l*b
print(area(10,b=5))    # here first thing should be "l"
# if put 10,and l=10 error occur sinc for l 2 values are passing

# default
def greetings(name,message="Welcome"):
    print(f"Hello {name} , {message}")

greetings("Devika")


# Arbitrary positional arguments

# def area(l,b):
#     return l*b
# print(area(10,5,3))  # this will give error


def add_num(*args):
    total=0
    for n in args:
        total+=n
    return total
print(add_num(1,2,3,4,5))


def info(**kwargs):
    for key,value in kwargs.items():
        print(f"{key}:{value}")
info(name="Devika",ph=109108298)