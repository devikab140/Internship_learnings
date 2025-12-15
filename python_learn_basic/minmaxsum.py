# numbers=[x for x in int(input("Enter the numbers: ")).split()] 

# this can't be used it store elmnt as str so operations not possible , for single line code we can use map() function





def maxmin(numbers):
    DICT={"max":max(numbers),"min":min(numbers),"sum":sum(numbers),"avg":sum(numbers)/len(numbers)}
    return DICT

numbers=[int(x) for x in input("Enter the numbers with space: ").split()]
print(maxmin(numbers))


# n=int(input("How many numbers are there: "))
# num=[]
# for i in range(n):
#     x=float(input(f"Enter your {i+1}'th number: "))
#     num.append(x)
# print(maxmin(num))





# n=int(input("How many numbers are there: "))
# numbers=[]
# for i in range(n):
#     x=float(input(f"Enter your {i+1}'th number: "))
#     numbers.append(x)
# # print(numbers)
# max_num=max(numbers)
# min_num=min(numbers)
# sum_num=sum(numbers)
# avg_num=sum(numbers)/len(numbers)
# # print(max_num,min_num,sum_num,avg_num)
# num_dict={"Max":max_num,"Min":min_num,"Sum":sum_num,"Avg":avg_num}
# print(num_dict)
# # directly adding to a dictionary
# DICT={"max":max(numbers),"min":min(numbers),"sum":sum(numbers),"avg":sum(numbers)/len(numbers)}
# print(DICT)


