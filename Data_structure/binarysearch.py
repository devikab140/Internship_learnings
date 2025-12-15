# limit 1-100
# take half of limit---check , 
# again check less or big
# less --- half of middle to lower ---check repeat

def binary(t,numbers):
    first=0
    last=(len(numbers))-1  
    # print(numbers[last])    # 100 is length but index start from 0 so last index is 99

    while last>=first:          # here we can also use for i in numbers: -------> but time complexity less for while
        middle=(numbers[last]+numbers[first])//2      # also we can do last+first//2 and  in if ---> if numbers[middle]==target 
        # print(middle)
        if middle==target:
            # print(f"The number is {middle}")
            return middle
            break
        else:
            if middle>target:           # this nested if can be given as elif, if without above else
                last=middle
            else:
                first=middle+1

target=5
num=[x for x in range(1,101)]
print(f"number is {binary(target,num)}")


#########################################################################################

# through recursive function


def bnrysrch(num,t):
    if (len(num))==0:
        return False
    else:
        midpnt=(len(num))//2
        if num[midpnt]==t:
            return True
        elif(num[midpnt]>t):
            return bnrysrch(num[:midpnt],t)
        else:
            return bnrysrch(num[midpnt+1:],t)
target=18
numbers=[x for x in range(1,101)]
print(f"Number is correct: {bnrysrch(numbers,target)}")