import math

X=[2,4,6,8,9]
y=[0,0,1,1,1]
n=len(X)
sum_x=sum([i for i in X])
sum_x2=sum([i**2 for i in X])
sum_y=sum([i for i in y])
sum_xy=sum([i*j for i,j in zip(X,y)])
# print(sum_x2)

m=((n*sum_xy)-(sum_x*sum_y)) / ((n*sum_x2)-(sum_x)**2)
print("coeficient",m)
c=(sum_y-(m*sum_x)) / n
print("intercept",c)

p_x=[1/(1+(math.exp(-(m*i+c)))) for i in X]
print("Probabilities are",p_x)

