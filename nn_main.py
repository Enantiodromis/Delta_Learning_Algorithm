from sklearn import datasets
from sequential_delta_learning import sequential_delta_learning_run

# Neural Networks
# The Delta Learning Algorithm is to be applied in order to learn the weights of a linear 
# threshold unit. Below, the Delta Learning Algorithm is called and applied to the dataset:

# Feature Vector, xT:  Class: 
#      (0.0, 2.0)         1   
#      (1.0, 2.0)         1   
#      (2.0, 1.0)         1   
#     (-3.0, 1.0)         0   
#    (-2.0, -1.0)         0   
#    (-3.0, -2.0)         0   

# QUESTION ONE: 
# Two epochs of the Sequential Delta Learning Algorithm is applied to the training data above.
# The initial parameters values:
#   - θ=−6.5
#   - w1=−7.5  
#   - w2=1.5 
#   - learning rate of 1.0 
# A linear threshold unit has a transfer function which is a linear weighted sum of its inputs and an activation
# function that is the Heaviside function. For the Heaviside function H(0) = 0.5.
#
# Values reported are to at least 2 decimal places, and a precision of at least 4 decimal places for all calculations.
iris = datasets.load_iris()

w = [6.5,-7.5,1.5]
n = 1.0
iterations = 2
X = [[0.0,2.0],[1.0,2.0],[2.0,1.0],[-3.0,1.0],[-2.0,-1.0],[-3.0,-2.0]]
t = [1, 1, 1, 0, 0, 0]

sequential_delta_learning_run(X,n,w,t,iterations)

w = [-0.5,-2.5,3.5,1.5,0.5]
n = 0.10
iterations = 2
X = iris.data
t = iris.target

w_new = sequential_delta_learning_run(X,n,w,t,iterations)

sequential_delta_learning_run(X,n,w_new,t,iterations)