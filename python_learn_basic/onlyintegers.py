#create a function to take only integers
#check each value in list is integer or not ----> int(value)==
#
# get a list ( include different type)
# call function & print answers from function

# def integers(lists):
#     new_l=[]
#     for i in lists:
#         if int(i)==i:
#             new_l.append(i)
#         else:
#             continue
#     return new_l

# lists=[input("Enter the strings: ").split()]
# print(integers(lists))

lists=[x for x in input("Enter the input: ").split()]  
#print(lists)
new_l=[]
for x in lists:
    try:
        z=int(x)  
        #print(z)
        new_l.append(z)
    except:
        continue
print(new_l)