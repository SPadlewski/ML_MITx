# perceptron learning algorithm
# let w be the weight vector
# w=[theta0,theta1,theta2,theta3]
# theta=[theta1,theta2,theta3]
# thet0 is the bias bias term

# perceptron learning algorithm:
# for each example x
# update w if y_pred=sign(wTx)!=y where y is the true label for x
# then w<-w+y*x update w

import numpy as np

dataset = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])

print("The dataset:\n", dataset)
print(" ")
# now transform the dataset using quadratic kernel

# define the kernel using lambda function
k = lambda x: np.array([x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2])

# transformed data
data_trans = []
for x in map(k, dataset):
    data_trans.append(x)

# convert data_trans to numpy


data_trans = np.array(dataset)
print(data_trans)
# combined ones and data_trans to get the new data
# ones vector got placed as the first column in the transformed dataset

data_trans = np.hstack(
    [np.ones(dataset.shape[0]).reshape(dataset.shape[0], 1), data_trans])  # this is done for to calculate the bias term
print("Transformed dataset:\n", data_trans)

w = np.zeros(3)  # intialize the parameter vector to be zeros

# store the mistakes in a list
mistakes = [1, 9, 10, 5, 9, 11, 0, 3, 1, 1]

label = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

for i in range(dataset.shape[0]):
    # mistakes got multiplied as according to the perceptron algorithm
    # the weight vector gets updated only if for each example the algorithm makes
    # mistake
    w = w + mistakes[i] * label[i] * data_trans[i]

print(" ")
theta0 = w[0]
print("Theta0 is given by ", theta0)
print(" ")
theta = w[1:]
print("Theta vector is given by :\n", theta, theta.shape)