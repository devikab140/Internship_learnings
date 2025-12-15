# for i in range(10):
#     if i==5:
#         break       # terminate here if i==5
#     print(i)

for i in range(10):
    if i==5:
        continue # if value become 5 it will skip this iteration and execute the next
    print(i,end=" ")
else:
    print()
    print("Loop Completed")