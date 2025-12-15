# # Let us say your expense for every month are listed below,
# # January - 2200
# # February - 2350
# # March - 2600
# # April - 2130
# # May - 2190
# # Create a list to store these monthly expenses and using that find out,

# # 1. In Feb, how many dollars you spent extra compare to January?
# # 2. Find out your total expense in first quarter (first three months) of the year.
# # 3. Find out if you spent exactly 2000 dollars in any month
# # 4. June month just finished and your expense is 1980 dollar. Add this item to our monthly expense list
# # 5. You returned an item that you bought in a month of April and
# # got a refund of 200$. Make a correction to your monthly expense list
# # based on this


# expense=[2200,2350,2600,2130,2190]
# #1
# print(f"Extra spent compared to January is {abs(expense[0]-expense[1])}")
# #2
# sum=0
# for i in range(3):
#     sum+=expense[i]
# print(f"Total expense in first quarter is {sum}")
# #3
# for i in range(len(expense)):
#     if expense[i]==2000:
#         print("Month is {i}")
#         break
# else:
#     print("No month has expense 2000")
    
# #4
# expense.append(1980)
# print(expense)

# #5
# expense[3]=expense[3]-200
# print(expense)



# # You have a list of your favourite marvel super heros.
# # heros=['spider man','thor','hulk','iron man','captain america']
# # Using this find out,

# # 1. Length of the list
# # 2. Add 'black panther' at the end of this list
# # 3. You realize that you need to add 'black panther' after 'hulk',
# #    so remove it from the list first and then add it after 'hulk'
# # 4. Now you don't like thor and hulk because they get angry easily :)
# #    So you want to remove thor and hulk from list and replace them with doctor strange (because he is cool).
# #    Do that with one line of code.
# # 5. Sort the heros list in alphabetical order (Hint. Use dir() functions to list down all functions available in list)
# # Solution

# heros=['spider man','thor','hulk','iron man','captain america']
# print(len(heros))
# heros.append("black panther")
# print(heros)
# heros.remove("black panther")
# print(heros)
# heros.insert(3,"black panther")
# print(heros)
# heros[1:3]=["doctor strange"]
# print(heros)
# heros.sort()
# print(heros)
import numpy as np
array=np.array([[1,2,3,4],
                [5,6,7,8]])
# print(np.sum(array))
# # print(np.max(array))

# tple={1:'a'}
# if tple =={}:
#     print(tple ,"is empty")
# else:
#     print(tple,"not empty")

integers=[]
for i in range(1,11):
    integers.append(i)
# print(integers)

# l1=[1,2,3]
# print(l1[::-1])
# l2=[4,5,6]
# print(np.square(l1))
# print(l1+l2)

# stg=input("Enter a string: ")
# vowels="aeiouAEIOU"
# count=0
# # for i in stg:
#     if i in vowels:
#         count+=1
# print(f"count of vowels in {stg} is {count}")


# import pandas as pd
# import pymysql

# # --- Connect to MySQL ---
# connection = pymysql.connect(
#     host="localhost",
#     user="root",
#     password="Devika#513",
#     database="bank"
# )

# # --- Load SQL data ---
# df_sql = pd.read_sql("SELECT * FROM investment", connection)

# # --- Load Excel/CSV data ---
# df_csv = pd.read_csv("C:/Users/devik/Downloads/investment.csv")

# # Suppose CSV has a column 'account_id' that links to SQL’s account_id
# merged_df = pd.merge(df_sql, df_csv, on="account_id", how="outer")

# print(merged_df)


