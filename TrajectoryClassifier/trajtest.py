import numpy as np
import math
from stochastic.processes.continuous import WienerProcess
from stochastic.processes.continuous import FractionalBrownianMotion
from scipy.optimize import minimize_scalar


def maximal_excursion_test_stats(traj, d):
    """ 
    Input parameter:
    -------
    traj : single trajectory from a stochatsic process
    d : dimension of the stochastic process

    Output result:
    -------
    This function will return the maximal excursion test statistics of the input trajectory

    """
    traj = traj.reshape(-1,d)
    stats = (np.sqrt(d) * np.max(np.linalg.norm(traj[1:] - traj[0], axis = 1))) / (np.sqrt(np.sum(np.square(np.linalg.norm(np.diff(traj, axis = 0), axis = 1)))))
    return stats



def MSD_estimate1d(traj, traj_length) : 
    """
    Input parameter:
    -------
    traj: single trajectory from a 1d stochastic process
    traj_length: the length of the trajectory

    Output result:
    -------
    This function will return the MSD estimates of the input trajectory

    """
    output_array = np.zeros(traj_length + 1)
    for lag in range(1,traj_length + 1):
        squared_distances = [(traj[i] - traj[i-lag]) ** 2 for i in range(lag,traj_length + 1)]
        output_array[lag] = np.sum(squared_distances) / (traj_length - lag + 1)
    return output_array



def MSD_estimate2d(traj,traj_length): 
    """
    Input parameter:
    -------
    traj: single trajectory from a 2d stochastic process
    traj_length: the length of the trajectory

    Output result:
    -------
    This function will return the MSD estimates of the input trajectory

    """
    traj = traj.reshape(-1,2)
    output_array = np.zeros(traj_length + 1)
    for lag in range(1,traj_length + 1):
        squared_distances = [((traj[i,0] - traj[i-lag,0]) ** 2 + (traj[i,1] - traj[i-lag,1]) ** 2) for i in range(lag,traj_length + 1)]
        output_array[lag] = np.sum(squared_distances) / (traj_length - lag + 1)
    return output_array



def F_asym1d(x, r = 20000):
    """
    Input parameter:
    -------
    x: test statistics
    r: number of truncated terms in the analytic expression of the asymptotic CDF. By default, r = 20000

    Output result: 
    -------
    The (approximate) value of asymptotic CDF evaluated at x. 

    Remarks:
    -------
    This function only computes the CDF of test stats when dimension = 1. For dimension = 2 and 3, the analytic 
    expression of CDF will be different. 

    """
    a = np.arange(r)
    b = a[::2]
    s = 0
    for i in range(len(b)):
        s = s + ((-1)**(b[i]/2))*(4/((b[i]+1)*(np.pi)))*(math.exp((-1/8)*(((np.pi)*(b[i]+1))/x)**2))
    return s



def F_hat2d(n, x, func, num_of_MC = 10000):
    """
    Input parameters:
    -------
    n: This is an array of consecutive integers, starts from 2, ends with 1+len(n).
    x: Usually x = np.arange(0,5.1,0.1).
    func: This is a function parameter. It should compute the maximal excursion test stats of the input trajectories.
    num_of_MC: Number of Monte Carlo Simulations. By default, num_of_MC = 10000.

    Output result:
    -------
    This function will return a table, with ith row being the empirical CDF of T_{i+1} evaluated at x.
    Here, T_{i+1} is the maximal excursion test stats of trajectories with size i+1

    """
    table = np.zeros([len(n), len(x)])
    for i in range(len(n)):
        std_bm = WienerProcess(t = int(i+2))
        std_bm_traj = np.dstack([[std_bm.sample(int(i+2)) for k in range(num_of_MC)], [std_bm.sample(int(i+2)) for k in range(num_of_MC)]])
        stats = func(std_bm_traj)
        for j in range(len(x)):
            table[i, j] = np.sum(np.where(stats <= x[j], 1, 0)) / num_of_MC
    return table



def num_of_null(p,m):
    """
    Input parameters:
    -------
    p: Sorted pvalues of the m null hypotheses
    m: Number of null hypothese that we are testing

    Output result:
    -------
    This function will give an estimate of m_0, the number of true null hypotheses.
    Knowing \widehat{m_0} enables us to carry out adaptive BH procedure. 

    """
    S = (1-p) / (m+1-np.arange(m))
    j = np.where(np.diff(S)<0)[-1][0]
    result = min(1/S[j] + 1, m)
    return result



def dfBB(traj,n):
    """
    Input parameters:
    -------
    traj: A 1D fractional Brownian trajectory
    n: Size of input trajectory

    Output result:
    -------
    This function will return the differenced fractional Brownian bridge trajectory of the input fBM trajectory.

    """
    first_element = traj[0]
    last_element = traj[-1]
    indices = np.arange(len(traj))
    fBB = traj - first_element - (indices / n) * (last_element - first_element)
    DFBB = np.diff(fBB)
    return DFBB



def sample_autocorrelation(traj,n) : 
    """
    Input parameters:
    -------
    traj: A 1D diffrenced fractional Brownian bridge trajectory
    n: Size of input trajectory + 1

    Output result:
    -------
    This function will return the sample autocorrelation of the input dfBB trajectory. 

    """
    output_array = np.zeros(n) 
    for lag in range(0,n): 
        squared_distances = [traj[i] * traj[i-lag] for i in range(lag,n)]
        output_array[lag] = np.sum(squared_distances)
        autocorrelation = output_array / output_array[0]
    return autocorrelation



def objective_hurst_estimate(traj,h,n):
    """
    Input parameters:
    ------
    traj: A 1D fractional Brownian trajectory
    h: Hurst exponent of that trajectory
    n: Size of the input trajectory

    Output result:
    ------
    This is the objective function when we do the hurst exponent estimation. 
    The hurst exponent estimate \widehat{h} is the global minimum of this function on (0,1)

    """
    pho_hat = sample_autocorrelation(dfBB(traj,n),n)
    func = np.zeros(n-1)
    for i in range(n-1):
        func[i] = (pho_hat[i+1]-(0.5 * ((i+2)**(2*h)-2*((i+1)**(2*h))+i**(2*h)) + n**(2*h-2) + ((i+1)**(2*h)-(n-i-1)**(2*h)-n**(2*h))/(n*(n-i-1)))/(1-n**(2*h-2)))**2
    return np.sum(func)



def hurst_exponent_estimate(traj,n):
    """
    Input parameter:
    ------
    traj: A set of 1D fractional Brownian trajectories.
    n: Size of input trajectories
    
    Output result:
    ------
    This function will return the hurst exponent estimate of the input trajectories. 

    """
    hurst = np.zeros(len(traj))
    for i in range(len(traj)):
        given_traj = traj[i,:]
        fixed_traj_function = lambda h: objective_hurst_estimate(given_traj,h,n)
        result = minimize_scalar(fixed_traj_function, bounds = (0,1), method = 'bounded')
        hurst[i] = result.x
    return hurst 



def hurst_threshold(size, num_of_MC = 100):
    """
    Input parameter:
    ------
    size: This is an array of consecutive integers, starts from 2, ends with 1+len(n). For example, size could be np.arange(2,101).
    num_of_MC: Number of Monte Carlo Simulations. By default, it is 100.


    Output result:
    ------
    This function will give the thresholds of using hurst exponent estimates to do classification of different size of trajectories.


    """
    result = np.zeros([len(size), 2])
    for i in range(len(size)):
        std_bm = FractionalBrownianMotion(hurst = 0.5, t = int(i+2))
        std_bm_traj = np.stack([std_bm.sample(int(i+2)) for k in range(num_of_MC)])
        hurst_exponent = hurst_exponent_estimate(std_bm_traj, i+2)
        sort_hurst_exponent = np.sort(hurst_exponent)
        result[i,:] = np.array([sort_hurst_exponent[2], sort_hurst_exponent[97]])
    return result











































