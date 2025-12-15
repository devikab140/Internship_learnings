import random

number=random.randint(1,100)
for i in range(5):
    num=float(input("Enter a number: "))
    if num==number:
        print("Correct guesss!!!!!!")
        break
    elif(num>number):
        print("The number you entered is greater value type a small value")
    elif(num<number):
        print("The number you enterd is small give a big value")
    print(f"try {i+1} over")
    