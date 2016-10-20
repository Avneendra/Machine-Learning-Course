import numpy as np 

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w. 
    """
    w = np.zeros(len(x[0]))
    k=0
    for ai in alpha:
        if ai>0:
            w+=y[k]*ai*x[k]
        k+=1
    
    # TODO: IMPLEMENT THIS FUNCTION
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices 
    of all of the support vectors
    """

    support = set()
    # TODO: IMPLEMENT THIS FUNCTION
    k=0
    for ex in x:
        
        val=(y[k]*(sum(np.transpose(w)*ex)+b))
        if(abs(val-1)<tolerance): 
            support.add(k)
        k+=1
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    k=0
    for ex in x:
        
        val=1-(y[k]*(sum(np.transpose(w)*ex)+b))
        if(val>0):
            slack.add(k)
        k+=1
    return slack
