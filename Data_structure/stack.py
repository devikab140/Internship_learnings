stack=[]

def push(item):
    stack.append(item)
    print(f"Pushed element is {item}")
def pop():
    if not stack:
        print("stack is empty")
        return None
    return stack.pop()

# push(10)
# push(20)
n=int(input("How many elements: "))
for i in range(n):
    push(input("Enter a value: "))
print("poped item is ",pop())
#stack.pop() also use 
print(stack)