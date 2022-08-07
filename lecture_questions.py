import numpy as np


def hinge_loss(x):
    return x


x_1_t=np.array([1,0,1]).T
x_2_t=np.array([1,1,1]).T
x_3_t=np.array([1,1,-1]).T
x_4_t=np.array([-1,1,1]).T

y_1=2
y_2=2.7
y_3=-0.7
y_4=2

X=[x_1_t,x_2_t,x_3_t,x_4_t]
Y=[y_1,y_2,y_3,y_4]

theta=np.array([0,1,2]).T

def hinge_loss(X,Y,R=None ):
    if R is None:
        R = []
    for i in range(len(X)):
        z=Y[i]-np.matmul(X[i],theta)
        if (z)>=1:
            R.append(0)
        else:
            R.append(1-z)

    return sum(R)/len(R)

def square_error_loss(X,Y,R=None ):
    if R is None:
        R = []
    for i in range(len(X)):
        z=np.square(Y[i]-np.matmul(X[i],theta))/2
        R.append(z)

    return sum(R)/len(R)

print(hinge_loss(X,Y))
print(square_error_loss(X,Y))