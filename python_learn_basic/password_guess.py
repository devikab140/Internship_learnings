#create a password
#user have to type password
#check whether these matching 
#count the tries(max 3)
#if match ocuurs in 3 attempts print correct else print finished

password="Devika@1234"
tries=0 
for i in range(3):
   user_pswd=input("Enter the password: ")
   tries+=1
   print(f"Attempt {i+1} is over")    #i+1 instead we can use tries
   if tries<=3:
        if password==user_pswd:
            print("Correct")
            break
        else:
            print("Incorrect password")
   else:
       print("finished") # this can be gve lastafter for without else
 
        
# for i in range(1,4):
#     user_pswd=input("Enter the password: ")
#     if password==user_pswd:
#         print("correct password")
#         break
#     else:
#         print("incorrect password")
#     print(f"Attempt {i} is over")
# print("finished")