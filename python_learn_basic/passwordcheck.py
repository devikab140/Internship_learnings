# get a password
# check condition
# uppercase ,lowercase,number,spcl chrctr,min 12 ,year 2025,sum of nmber125,roman no  sum 50

while True:    # this used for repiting the loop, like repetedly asking to type the password until break occur
    password=input("Enter your password: ")
    roman={"I":1,"X":10,"V":5,"L":50}
    numbers="0123456789"
    alphabet="abcdefghijklmnopqrstuvwxyz"
    spcl_chrctr="@#$%&*?\<>~^+=-_|`,.:;/"
    num_sum=0
    roman_sum=0
    count_upper=0
    count_spcl=0
    count_lower=0
    count_num=0
    if len(password)>=12:
        if "2025" in password:
            for i in password:
                if i in alphabet.upper():
                    count_upper+=1
                    if i in roman:
                        roman_sum+=roman[i] 
                if i in alphabet:
                    count_lower+=1
                if i in spcl_chrctr:
                    count_spcl+=1
                if i in numbers:
                    num_sum+=int(i)
                    count_num+=1
            if (count_lower>=1 and count_upper>=1 and count_num>=1 and count_spcl>=1 and num_sum==125 and roman_sum==50):
                print("Good Password")
                break
            elif(count_lower<1):
                print("Password is missing lowercase letter")
            elif(count_upper<1):
                print("Password is missing uppercase letter")
            elif(count_num<1):
                print("Password is missing numbers")
            elif(count_spcl<1):
                print("Password is missing special character")
            elif(num_sum!=125):
                print("Password's numbers sum is not 125")
            elif(roman_sum!=50):
                print("Password's roman numbers sum is not 50")
        else:
            print("Password does not contain 2025")
    else:
        print("Retype password with minimum 12 characters")



# L2025@mnh155555555555555555555555