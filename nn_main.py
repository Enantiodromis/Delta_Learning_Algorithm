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
#
# We need to add a value of 1 to all input data, therefore X_augmented = [1,X]
# w_with_threshold = [-theta, w] wX - theta = w_with_threshold.X_augmented.
# Once wx has been computed we then apply the heavside function, of wx = 0 H(wx) = 0.5
# for all other cases H(wx) is 0 if wx < 0, 1 if wx > 0
