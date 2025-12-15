# for triangle shape

n=int(input("Enter the number: "))
for i in range(1,n+1): 
    print(str(i)*i) # any (string*x) --> x times the string will print
    

#pyramid

# get the number from user upto which we have to print
# print first number (firs give equal sapce on L&R and print first no 1 )
# for second number space will decrease by 1 , print prev one then current then prev
# third one 

n=int(input("Enter the number: "))
for i in range(1,n+1):
   print(" "*(n-i),end="")
   for j in range(1,i+1):
        print(j,end="")
   for j in range(i-1,0,-1):
       print(j,end="")
   print()
   
      