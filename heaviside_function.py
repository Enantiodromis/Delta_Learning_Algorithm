import numpy as np

# The Heaviside function often written as H(x), is a non continuous function whose value is zero for a negative input and one for a posotive input.
# In the implementation below we are use the following definition of the Heaviside Function.
# 
#         0, x < 0
#   H(x)= 0.5, x = 0
#         1, x > 0
#
def heaviside_function_run(w, x):
    wx = np.dot(w,x)

    if wx > 0: return 1 
    elif wx == 0: return 0.5
    else: return 0 