# 6
def countdown(n):
    for i in range(n,0,-1):
        yield i
# print(next(countdown(5)))
n=int(input("Enter value for n: "))
for num in countdown(n):
     print(num)

#7
class Laptop:
    def __init__(self,brand,model):
        self.brand=brand
        self.model=model

obj1=Laptop("Dell","XPS 15")
obj2=Laptop("Apple","MacBook pro")
print(obj1.brand,obj1.model)
print(obj2.brand,obj2.model)

#5

def calculate_total(price,*args,tax=0.05):
    total=price+sum(args)
    final_total=total+total*tax
    return final_total
print(calculate_total(100,5,20,tax=0.1))  #5,20 --> *args , we don't need to give valuse in tuple explicitly just give multiple values it will store values like a tuple        