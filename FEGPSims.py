import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
import numpy.linalg as la
import random

# Constructing the desired matrices

def get_Matern_covariance(data, kappa, s):
    """
    data: N x D array of N vectors x_i in R^D
    kappa: correlation parameter in SPDE formulation
    s: smoothness parameter in SPDE formulation

    Returns a N x N matrix M with M_ij = C_Mat(x_i, x_j)
    """

    dim = data.shape[1]
    nu = s - dim/2
    sigma2 = gamma(nu)/(gamma(s)*(4*np.pi)**(dim/2))
    scalar = sigma2*(2**(1-nu)/gamma(nu))

    # matrix of pairwise distances
    D = kappa*np.linalg.norm(data[:, None] - data[None, :], axis = -1)

    M = np.zeros(D.shape)
    off_diag = (D > 0)
    M[off_diag] = scalar * (D[off_diag] ** nu) * kv(nu, D[off_diag])
    M[~off_diag] = sigma2

    return M

def get_mass_matrix(n_h, L):
    h = L/(n_h-1)
    off_diag = np.repeat(1, n_h-1)
    diag = np.full(n_h, 4)
    diag[0] = diag[-1] = 2
    M = np.diag(off_diag, k=1) + np.diag(diag, k=0) + np.diag(off_diag, k=-1)
    return (h/6)*M

"""
def get_lumped_mass_matrix(M, n_h):
    M = get_mass_matrix(n_h)
    return np.diag(M.sum(axis = 1))
"""


def get_stiffness_matrix(n_h, L):
    h = L/(n_h-1)
    off_diag = np.repeat(-1, n_h-1)
    diag = np.full(n_h, 2)
    diag[0] = diag[-1] = 1
    G = np.diag(off_diag, k=1) + np.diag(diag, k=0) + np.diag(off_diag, k=-1)
    return (1/h)*G

def get_FE_precision_matrix(M, G, kappa, s):
    """
    M: FE mass matrix with M_ij = <e_{h,i}, e_{h,j}>_2
    M_tilde: Lumped FE mass matrix
    G: FE stiffness matrix with G_{ij} = <grad(e_{h,i}), grad(e_{h,j})>_2
    kappa: correlation parameter in SPDE formulation
    s: smoothness parameter in SPDE formulation

    Returns the precision matrix (k^2M + G)[M_tilde^{-1}(k^2M+G)]^{s-1}
    """

    M_tilde = np.diag(M.sum(axis = 1))
    block1 = kappa**2 * M + G
    #print("rank of block1 is {r} and dimension is {d}".format(r = la.matrix_rank(block1), d = block1.shape))
    block2 = la.solve(M_tilde, block1)

    #return la.matmul(block1, block2**(s-1))
    return block1 @ la.matrix_power(block2, s-1)

def get_FE_design_matrix(data, n_h, L):
    """
    data: N x D array of N vectors x_i in R^D

    Return N x n_h finite element design matrix S with S_ij = e_j(X_i)
    """
    
    K = n_h - 1
    h = L/K
    S = np.zeros((data.shape[0], n_h))
    for i in range(data.shape[0]):
        x_i = data[i,:].item()
        j = int(np.floor(x_i/h)) # get index in the partition
        S[i,j] = -x_i/h + j + 1
        S[i, j+1] = x_i/h - j
    return S


# ground truth and comparison vectors

def get_ground_truth_vector(data, L, kappa, s, M):
    zeta = np.random.multivariate_normal(mean = np.zeros(2*(M+1)), cov = np.eye(2*(M+1)))
    alpha = zeta[:M+1]
    beta = zeta[M+1:]
    cnst = alpha[-1]/(np.sqrt(kappa*L))
    scalar = (np.sqrt(2)*(kappa**(s-0.5)))/np.sqrt(L)

    f_0 = []
    for x in data:
        sum_x = 0
        for i in range(M+1):
            sum_x += ((kappa + ((i*np.pi)/L)**2)**(-s/2))*(alpha[i]*np.cos((i*np.pi*x)/L) + beta[i]*np.sin((i*np.pi*x)/L))
        f_x = cnst + scalar*sum_x
        f_0.append(f_x)
    
    return np.array(f_0)


random.seed(666)

N = 100
L = 5
kappa = 5
s = 2
tau = 2
M = 100
n_h = 50

data = np.random.uniform(0, L, N)
data = data.reshape((N, 1))

def generate_data(N, L, n_h, kappa, s, tau):
    """
    N: sample size
    L: upper limit of domain
    n_h: number of finite element basis functions
    kappa: correlation parameter in SPDE formulation
    s: smoothness parameter in SPDE formulation
    tau: standard deviation of likelihood for GP regression

    Returns posterior vectors F_cf, F_fe in R^N with data generated from Uniform(0,L)
    """

    # Setting 1: Matern covariance
    I_N = np.eye(N)
    Sigma = get_Matern_covariance(data, kappa, s)
    f_N = np.random.multivariate_normal(mean = np.zeros(N), cov = Sigma)
    y_cf = np.random.multivariate_normal(mean = f_N, cov = (tau^2)*I_N)
    block1 = Sigma + (tau^2)*I_N
    F_cf = Sigma @ la.solve(block1, y_cf)

    # Setting 2: Finite element
    M, G = get_mass_matrix(n_h, L), get_stiffness_matrix(n_h, L)
    Q = get_FE_precision_matrix(M, G, kappa, s)
    
    evals_Q = la.eigvalsh(Q)
    
    #print("smallest eigenvalue of Q is {v}".format(v = evals_Q[0]))

    S = get_FE_design_matrix(data, n_h, L)

    #print("max row sum in S is {s}".format(s = np.max(S.sum(axis=1))))
    #print("min row sum in S is {s}".format(s = np.min(S.sum(axis=1))))

    w = np.random.multivariate_normal(mean = np.zeros(Q.shape[0]), cov = la.inv(Q))
    y_fe = np.random.multivariate_normal(mean = S @ w, cov = (tau^2)*I_N)
    block2 = S.T @ S + (tau**2)*Q
    
    #print(np.allclose(block2, S.T @ S + (tau**2)*Q))
    #print("rank of block2 is {r} and dimension is {d}".format(r = la.matrix_rank(block2), d = block2.shape)) # PROBLEM HERE

    evals_block2 = la.eigvalsh(block2)
    #print("smallest eigenvalue of block2 is {v}".format(v = evals_block2[0]))
    #F_fe = S @ la.inv(block2) @ S.T @ y_fe
    F_fe = S @ la.solve(block2, S.T @ y_fe)

    return np.array([F_cf, F_fe])

reps = 20

nums = np.arange(2, 200, 5)
nums = [int(num) for num in nums]
errs = []
for n_h in nums:
    print(n_h)
    avg_error = 0
    for rep in range(reps):
        F = generate_data(N, L, n_h, kappa, s , tau)
        F_cf = F[0,:]
        F_fe = F[1,:]
        f_0 = get_ground_truth_vector(data, L, kappa, s, M)
        avg_error += la.norm(F_cf - f_0)/N

    errs.append(avg_error/reps)
    print(errs[-1])

plot_data = np.array([nums, errs])
plt.plot(plot_data[0], plot_data[1])
