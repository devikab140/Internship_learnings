# get the max value from input
# check even or odd for each number
# sum up the even numbers

n=float(input("Enter the value: "))
sum=0
count=0
for x in range(int(n+1)): # for making n integer
    if (x%2)== 0:
        count+=1
        sum+=x
# print("sum of even numbers: ",sum)
print(f"Sum of the even numbers is {sum}")
print("Count of even numbers :",count)