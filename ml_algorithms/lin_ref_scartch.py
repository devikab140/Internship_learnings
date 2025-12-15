import numpy as np

# hrs=np.array([2,3,4,5,6,7])
# marks=np.array([20,35,44,57,71,89])
# n=len(hrs)
# hrs_sum=sum(hrs) #sum_X
# sum_marks=sum(marks) #sum_y
# prod_hrs_marks=np.sum(np.dot(hrs,marks)) #sum_Xy
# hrs_square=sum(hrs**2) #sum_x^2
 
# m=((n*prod_hrs_marks)-(hrs_sum*sum_marks)) / ((n*hrs_square)-(hrs_sum**2))
# print("slope is ",m)

# c=(sum_marks-(m*hrs_sum)) / n
# print("Intercept is ",c)

# x=hrs
# y=m*x+c

# print(y)

# error=((1/n)*np.sum((marks-y)**2))
# print("MSE is ",error)


# area
area=np.array([2600,3000,3200,3600,4000])
price=np.array([550000,565000,610000,680000,725000])


n=len(area)
area_sum=sum(area) #sum_X
sum_price=sum(price) #sum_y
prod_area_price=np.sum(np.dot(area,price)) #sum_Xy
area_square=sum(area**2) #sum_x^2
 
M=((n*prod_area_price)-(area_sum*sum_price)) / ((n*area_square)-(area_sum**2))
print("slope is ",M)

C=(sum_price-(M*area_sum)) / n
print("Intercept is ",C)

X=area
Y_cap=M*X+C
# print(Y_cap)
SSE=sum((price-Y_cap)**2)
print(SSE)
y_bar=(1/n)*sum(price)
SST=sum((price-y_bar)**2)
print(SST)

R2=1-(SSE/SST)
print("R2 score",R2)