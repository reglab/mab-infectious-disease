import numpy as np 


def sum_lambd(test_hist, alpha=1, beta=1):
    
    return np.sum([
        np.exp(-alpha*beta*te)*(np.exp(alpha*(t+1)) - np.exp(alpha*t))
        for t, te in enumerate(test_hist)
    ])


def lambd(t, total_tests, alpha=1, beta=1):
    return np.exp(alpha*(t - beta*total_tests))


def Ct(t, test_hist, alpha=1, beta=1, N=1000, c0=1):
    
    num = c0 * N * lambd(t, np.sum(test_hist), alpha, beta)
    denom = N - c0 + (c0 * sum_lambd(test_hist, alpha, beta))
    
    return num / denom



