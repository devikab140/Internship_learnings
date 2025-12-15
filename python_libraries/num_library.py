
import numpy as np
#creation
x=np.array("h")
y=np.array([4,5,1,2,3,6])
z=np.array([[1,5,6],
            [4,2,3]])
w=np.array([[[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]]])
# l=[1,2,3]
# print(l*2)
'''
print(x)
print(y)
print(w)
print(z)
# print(y*2)


# dim
print(x.ndim)
print(y.ndim)
print(z.ndim)
print(w.ndim)
#shape
print(x.shape)
print(y.shape)
print(z.shape)
print(w.shape) # 2 layers,2-rows,3-columns
#size
print(x.size)
print(y.size)
print(z.size)
print(w.size)


#range array
q=np.arange(1,10,2)  # start ,stope ,step
print(q)
#linspace
a=np.linspace(0,10,5) # default 50 values will print b/w start and stop
print(a)

#zeros ,ones and eye
zero=np.zeros((2,3))
zero_3=np.zeros(((2,2,3)))  # 3 dim--3brackets
print(zero_3)
print(zero_3.ndim)
print(zero)
one=np.ones((1,2))
print(one)
one_3=np.ones(((2,2,3)))
print(one_3)

r=np.eye(3)
print(r)
s=np.eye(2,3)
print(s)


#random numbers
np.random.seed(42)
R=np.random.rand(2)
print(R)
# R2=np.random.rand(2,2)
# print(R2)
# R3=np.random.rand(2,2,3)
# print(R3)

T=np.random.randint(1,10,2)  # 1D dimesion having 2 values
# T2=np.random.randint(1,10,(2,3))  #2d
# T3=np.random.randint(1,10,(3,2,3))  #3d
# print(T3)
# print(T2)
print(T)


u=np.random.randn(2,3)
print(u)
u3=np.random.randn(2,2,3) # 3dim===since 2 layers
print(u3)



#reshape

z=np.array([[1,2,3],
            [4,5,6]])
print(z.reshape(6))  # converting to 1D === give no of elements

print(y.reshape(2,2))  # 2*2 or 1*4 or 4*1 
print(y.reshape(2,-1))  # 2 rows specified and no of columns it will decide
print(y.reshape(-1,2))  # rows auto decide, column =2

p=z.flatten()
print(p)
#ravel also there it convert to 1D but it change the original one


s1=y.sum()
print(s1)
s2=np.sum(z[0:2,0:1])  #slicing taking only our needed elements
print(s2)
s3=w.sum()
print(s3)

#mean
m1=y.mean()
print(m1)

m2=np.mean(z)  # np.sum() is std way
m21=np.sum(z[0:1,0:2])
print(m21)
print(m2)

m3=w.mean()
m31=np.mean(w[1,1])
print(m31)
print(m3)



# dot and cross product


p=z.flatten()
d1=np.dot(y,p)  #1*6 6*1 ---> 1*1 ==one element
print(d1)
d2=np.dot(z,z.reshape(3,-1))
print(d2)
u3=np.random.randn(2,2,3)
d3=np.dot(w,u3.reshape(2,3,2))  # 2 layers , corresponding matrix will multiply
print(d3)



k=np.matmul(z,z.reshape(3,2))
print(k)

h1=np.random.choice([10,15,20,25,30])  # select one number from the list
print(h1)

print(np.sort(y))
print(np.sort(z))
 
print(np.unique(y))
print(np.unique(z))

y=np.array([4,5,1,5,4,1,2,2,2,3,6])
print(np.unique(y))
z=np.array([[1,5,3,4,6],
            [4,2,3,2,3]])
print(np.unique(z))

#slicing in 1D
#step (default 1)and start are not mandatory 
print(y)
print(y[0:4])
print(y[0:4:2])  #step
print(y[::]) # no start ,stop and step ---> if stop not mentioned should add colon
print(y[::-1])     # reverse

print(z)
print(z[1,2]) # indexing
print(z[0:2])  # 0,1 th row full
print(z[0:1,1:2])
print(z[:,0:1])  # all rows and first column
print(z[:,:])  # all rows and columns
print(z[0][1])  # 0th row 1st column

a=np.random.randint(1,10,5)
b=np.random.randint(1,10,5)
print(a)
print(b)
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a%b)
print(a**2)

print(np.pi)

a=np.random.randint(1,10,5)
b=np.random.randint(1,10,5)
print(a)
print(b)
# print(np.vstack((a,b)))
# print(np.hstack((a,b)))

# arr=np.hstack((a,b))
# print(np.split(arr,5))  #equal split

#broadcasting
y=np.array([4,5,1,2])
z=np.array([[1,5,6,1],
            [4,2,3,9]])
print(y+z)
print(y-z)
print(y*z)

a=np.array([1,2,3])
b=np.array([[1],
           [4],
           [8]])
print(b+a)

#filtering

print(z)
print([z<=4])
print([(z<=4)&(z>3)]) #multiple with symbols(symbols only) (&,|)
print(z[~(z<3)])
print(z[z>3])  # o\p is 1D

print(np.where(z>3,z,0)) #condition,original array, which value give if condition false(.3 values print as it is rest all becomes zero)

#aggregate functions:- we get one value output like sum,mean
#

print(w)
print(np.sum(w,axis=0)) #first block first emnt(1)+second block first elemnt(7)
print(np.sum(w,axis=1)) #first block first colmn(1+4)
print(np.sum(w,axis=2)) # row wise sum


M=np.array([[1,2,3],
             [14,5,6],
             [7,8,7]])
print(M)
# print(M.transpose())
# print(np.transpose(M))
print(np.linalg.det(M))
print(np.linalg.inv(M))
'''
#splitig
