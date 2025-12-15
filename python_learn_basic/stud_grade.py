m=float(input("Enter your mark: "))
match m:
    case n if n>95 and n <=100 :  # n if n>95 used 
        print("Grade is A+")
    case n if  n<=95 and n>90:
        print("Grade is A")
    case n if  n<=90 and n>85 :
        print("Grade is B+")
    case n if n<=85 and n>80 :
        print("Grade is B")
    case n if n<=80 and n>70 :
        print("Grade is C")
    case n if n<=70 and n>60 :
        print("Grade is D")
    case n if n<=60 and n>40 :
        print("Grade is E")
    case n if n>100:
        print("invalid number")
    case _:
        print("Grade is F , failed")

