import numpy as np
from prettytable import PrettyTable
from heaviside_function import heaviside_function_run

# Implementation of the Sequential Delta Learning algorithm. Using the update: w←w+η( t−y ) xt
def sequential_delta_learning_run(X, n, w, t, epoch):
    for el in X: el.insert(0,1) 
    results = []
    for iter in range(epoch):
        for iter1 in range(len(X)):
            previous_w = w 
            x = X[iter1]
            y = heaviside_function_run(w,x)

            # Calculating: η( t−y ) xt, the sequential update
            update = np.zeros(len(x))
            for iter2 in range(len(x)): update[iter2] = n * (t[iter1] - y) * x[iter2] # η( t−y ) xt

            # Calculating: w+η( t−y ) xt the new value of w by adding the current w with η( t−y ) xt
            w = np.add(w, update)

            # Formatting and storing output
            results.append((str(iter1 + 1 + (len(X) * iter)),np.round(t[iter1]), X[iter1], np.round(y, 4), np.round(w,2)))
    
    pt = PrettyTable(('Iteration','Class', 'X original', 'y = H(wx)', ' w = w+η( t−y ) xt'))
    for row in results: pt.add_row(row)

    pt.align['Iteration'] = 'c'
    pt.align['Class'] = 'l'
    pt.align['X original'] = 'l'  
    pt.align['y = H(wx)'] = 'l'
    pt.align['w_new'] = 'l'

    print(pt)


