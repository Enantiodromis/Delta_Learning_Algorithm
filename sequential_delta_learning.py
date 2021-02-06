import numpy as np
from prettytable import PrettyTable
from heaviside_function import heaviside_function_run

# Implementation of the Sequential Delta Learning algorithm. Using the update: w←w+η( t−y ) xt
def sequential_delta_learning_run(X, n, w, t, epoch):
    if not isinstance(X, list) and not isinstance(t, list): X,t = X.tolist(), t.tolist() # Converting datasets to list if not already
    
    for el in X: el.insert(0,1) 
    results = []
    counter = 0

    for iter in range(epoch):
        for iter1 in range(len(X)):
            x = X[iter1]

            y = heaviside_function_run(x,w)

            # Calculating: η( t−y ) xt, the sequential update
            update = np.zeros(len(x))
            for iter2 in range(len(x)): update[iter2] = n * (t[iter1] - y) * x[iter2] # η( t−y ) xt

            # Calculating: w+η( t−y ) xt the new value of w by adding the current w with η( t−y ) xt
            w = np.add(w, update)

            # Formatting and storing output
            results.append((str(iter1 + 1 + (len(X) * iter)),np.round(t[iter1]), X[iter1], np.round(y, 4), np.round(w,2)))
            
            # Calculating the percentage of samples for which the linear threshold unit produces the desired output.
            # Using the initial parameter values, and those calculated after learning for 2 epochs
            if (t[iter1] == 0 and y == 1):
                counter = counter + 1
            if (t[iter1] == 1 and y == 0):
                counter = counter + 1
            if (t[iter1] == 2 and y == 0):
                counter = counter + 1

    percentage = (counter / (len(X) * epoch))*100
    print("COUNTER: " + str(counter) + " PERCENTAGE: " + str(percentage)+"%")
    pt = PrettyTable(('Iteration','Class', 'X original', 'y = H(wx)', ' w = w+η( t−y ) xt'))
    for row in results: pt.add_row(row)

    pt.align['Iteration'] = 'c'
    pt.align['Class'] = 'l'
    pt.align['X original'] = 'l'  
    pt.align['y = H(wx)'] = 'l'
    pt.align['w_new'] = 'l'

    print(pt)

    return w