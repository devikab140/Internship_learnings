
clrs=("Black","Blue","Green","Orange","Blue")
# also we can store the index of first blue in any variable
print(clrs.index("Blue",(clrs.index("Blue")+1)))

CLRS=["Black","Blue","Green","Orange","Blue"]
Blue_pop=CLRS.pop(1)
print(Blue_pop)

colours={"Black","Blue","Green","Orange","Blue"}
x=colours.pop()
print(x)
# print(colours.pop())
COLOURS={"Black":1,"Blue":3,"Green":7}
print(COLOURS.pop("Black"))


Name = str(input("Enter your name: "))
print(Name)
Age=int(input("Enter your age: "))
print(Age)


# arithmetic operators
a=int(input("Enter your first number: "))
b=int(input("Enter your second number: "))
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a%b)
print(a**b)
print(a//b)

# comparison operators
print(a==b)
print(a!=b)
print(a>b)
print(a>=b)
print(a<b)
print(a<=b)

#logical operators
a=int(input("Enter your first number: "))
b=int(input("Enter your second number: "))
print(a>b and a!=b)
print(a<b or a==b)
print(not a!=b)

#assignment operators
a=1
a+=2
print(a)
b=10
b-=5
print(b)
c=1
c*=5
print(c)
d=1
d/=4
print(d)

#identity operatos ---> is , is not

a=[1,2,3]
b=[1,2,3]
print(a==b)
print(a is b)
c=a
print(a is c)

print(a is not b)
print(a!=b)
print(a is not c )
# membership operators 
print(2 in a)
print(7 not in a)
print(1 not in a)


#bitwise operator

a=5
b=8
print(a and b)
print (a or b)
print( a | b)
print(a ^ b)

print(a<b | b==a)


clrs=("Black","Blue","Green","Orange","Blue","pink","meroon","Blue")
print(clrs.count("Blue"))
first_b=clrs.index("Blue")

second_b=clrs.index("Blue",first_b+1)
print(clrs.index("Blue",second_b))

third_b=clrs.index("Blue",second_b+1)
print(clrs.index("Blue",third_b))


