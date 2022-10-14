import jax
import jax.numpy as jnp
import jax.random as rd
import numpy as np
from functools import partial
from tqdm import tqdm
import optax as optim
import warnings
warnings.simplefilter('ignore')

def k_RBF(x,y,h=1.0):
    return jnp.exp(-1/h*jnp.sum((x-y)**2))

def k_lap(x,y,h=1.0):
    return jnp.exp(-1/h*jnp.sum(jnp.abs(x-y)))

def k_IMQ(x,y,h=1.0):
    return 1/jnp.sqrt(1+jnp.sum((x-y)**2)/(2*h))

def k_linear(x,y,h=1.0):
    return jnp.sum(x*y)

def k_pol(x,y,h=1.0,r=1,n=2):
    return (jnp.sum(x*y)+r)**n


def gaussian(x,mu,Sigma):
    '''
    mu: mean
    Sigma: covariance matrix
    '''
    k = x.shape[0]
    Sigma_inv = jnp.linalg.inv(Sigma)
    return 1/jnp.sqrt((2*jnp.pi)**k * jnp.abs(jnp.linalg.det(Sigma))) * jnp.exp(-0.5*(x-mu)@Sigma_inv@(x-mu))

def logp_gaussian_1D(x,mu,sigma2):
    '''
    R^p --> R
    mu: mean
    sigma: variance. 
    '''
    return - jnp.sum((x-mu)**2/(2*sigma2)) - 0.5*jnp.log(sigma2) - jnp.log(jnp.sqrt(2*jnp.pi))


def logp_gaussian_nD(x,mu,Sigma):
    '''
    mu: mean
    Sigma: covariance matrix
    '''
    return -0.5*(x-mu) @ jnp.linalg.inv(Sigma) @ (x-mu)


def logp_neal(x, sigma=6):
    return logp_gaussian_1D(x[0], sigma**2/4, sigma**2) + logp_gaussian_1D(x[1], 0, jnp.exp(0.5*x[0]))




def logp_cross(x):
    '''
        Problem: numerical unstable.
    '''
    m1 = jnp.array([0,2])*1.0
    m2 = jnp.array([-2,0])*1.0
    m3 = jnp.array([2,0])*1.0
    m4 = jnp.array([0,-2])*1.0

    s1 = jnp.array([[0.15**2,0],[0,1]])
    s2 = jnp.array([[1,0],[0,0.15**2]])

    eps = 1e-8
    return jnp.log(1/4*(gaussian(x,m1,s1) + gaussian(x,m2,s2) + gaussian(x,m3,s2) + gaussian(x,m4,s1)) + eps)

def logp_banana(x,b=0.1):
    y = jnp.array([x[0], -b*x[0]**2+x[1]+100*b])
    mu = jnp.array([0,0])
    Sigma = jnp.array([[100,0],[0,1]])*1.0
    return logp_gaussian_nD(y, mu, Sigma)


def logp_warp(x):
    x1 = x[0]
    x2 = x[1]
    norm = jnp.sqrt(x1**2 + x2**2)
    theta = jnp.arctan2(x2,x1) + 0.5*norm
    y1 = norm*jnp.cos(theta)
    y2 = norm*jnp.sin(theta)
    y = jnp.array([y1,y2])
    mu = jnp.array([0,0])
    Sigma = jnp.array([[1,0],[0,0.12**2]])
    return logp_gaussian_nD(y, mu, Sigma)

def logp_cauchy(x,x0,gamma):
    return jnp.log(gamma/jnp.pi/((x-x0)**2 + gamma**2))


logp_gaussian1d_vmap = jax.vmap(logp_gaussian_1D,in_axes=(0,None,0),out_axes=0)
logp_gaussian1d_vmap0 = jax.vmap(logp_gaussian_1D,in_axes=(0,None,None),out_axes=0)

def logp_neal_nD(x,sigma=6):
    x1 = x[1::]
    x0 = x[0]
    return logp_gaussian_1D(x0, sigma**2/4, sigma) + jnp.sum(logp_gaussian1d_vmap0(x1, 0, jnp.exp(0.5*x0)))


## FIXME: design chain distribution

def mean_f(x):
    return 2*jnp.tanh(x)

def variance_g(x):
    return jnp.exp(x/4)

def logp_chain_nD(x,sigma=6,f=mean_f,g=variance_g):
    '''
    x:(p,)
    '''
    x1 = x[1::]
    x0 = x[0:-1]
    return logp_gaussian_1D(x[0], sigma**2/4, sigma**2) + jnp.sum(jax.vmap(logp_gaussian_1D)(x1, f(x0), g(x0)))

def logp_tree_nD(x,sigma=6,f=mean_f,g=variance_g):
    x1 = x[1::]
    x0 = x[0]
    return logp_gaussian_1D(x0, sigma**2/4, sigma**2) + jnp.sum(logp_gaussian1d_vmap0(x1, f(x0), g(x0)))


def get_logp_chain(d,D,f=mean_f,g=variance_g):
    sigma = 6
    if d == 0:
        def logp(xd,x_gammad):
            '''
                xd: ()
                x_gammad: (1,)
            '''
            return logp_gaussian_1D(xd, sigma**2/4, sigma**2) + logp_gaussian_1D(x_gammad[0], f(xd), g(xd))
        return logp
    elif d == D - 1:
        def logp(xd,x_gammad):
            return logp_gaussian_1D(xd, f(x_gammad[0]), g(x_gammad[0]))
        return logp
    else:
        def logp(xd,x_gammad):
            return logp_gaussian_1D(xd, f(x_gammad[0]), g(x_gammad[0])) + logp_gaussian_1D(x_gammad[1], f(xd), g(xd))
        return logp
    

def get_logp_tree(d,D,f=mean_f,g=variance_g):   
    sigma = 6
    
    if d == 0:
        def logp(xd,x_gammad):
            return logp_gaussian_1D(xd, sigma**2/4, sigma**2) + jnp.sum(logp_gaussian1d_vmap0(x_gammad, f(xd), g(xd)))
        return logp
    else:
        def logp(xd,x_gammad):
            return logp_gaussian_1D(xd, f(x_gammad[0]), g(x_gammad[0]))
        return logp
    

    

def MP_structure(D,type='chain',f=mean_f,g=variance_g):
    sigma = 6
    idx_list = []
    logp_list = []
    if type=='chain':
        for d in range(D):
            logp_list.append(get_logp_chain(d,D))
            if d == 0:
                idx_list.append([0,1])
            elif d == D - 1:
                idx_list.append([D - 1,D - 2])
            else:
                idx_list.append([d,d-1,d+1])     
    
    elif type=='tree':
        for d in range(D):
            logp_list.append(get_logp_tree(d,D))
            if d == 0:
                idx_list.append(list(range(D)))
            else:
                idx_list.append([d,0])
    else:
        raise(ValueError)
    
    return logp_list, idx_list


def banana_sampler(size=10,D=2,seed=123,b=0.1):
    np.random.seed(seed)
    y = np.random.multivariate_normal([0,0],[[100,0],[0,1]],size=size)
    y1 = y[:,0]
    y2 = y[:,1]
    x1 = y1
    x2 = -y2 + b*y1**2- 100*b
    return np.stack([x1,x2],axis=1)

def neal_sampler(size=10,D = 2, seed=234,sigma=6):
    np.random.seed(seed)
    x_list = []
    x0 = np.random.normal(sigma**2/4, sigma, size=size)
    
    x_list.append(x0)
    for i in range(D-1):
        x_list.append(np.exp(x0/4) * np.random.normal(0,1,size=size))
    return np.stack(x_list, axis=1)

def cross_sampler(size=10,D=2,seed=345):
    np.random.seed(seed)
    m1 = np.array([0,2])*1.0
    m2 = np.array([-2,0])*1.0
    m3 = np.array([2,0])*1.0
    m4 = np.array([0,-2])*1.0

    s1 = np.array([[0.15**2,0],[0,1]])
    s2 = np.array([[1,0],[0,0.15**2]])
    N = np.random.multinomial(size,[0.25]*4)
    x1 = np.random.multivariate_normal(m1,s1,size=N[0])
    x2 = np.random.multivariate_normal(m2,s2,size=N[1])
    x3 = np.random.multivariate_normal(m3,s2,size=N[2])
    x4 = np.random.multivariate_normal(m4,s1,size=N[3])

    idx = np.random.permutation(size)

    return np.concatenate([x1,x2,x3,x4],axis=0)[idx]

def warp_sampler(size=10,D=2,seed=2323):
    np.random.seed(seed)
    y = np.random.multivariate_normal([0,0],[[1,0],[0,0.12**2]],size=size)
    y1 = y[:,0]
    y2 = y[:,1]

    norm = jnp.sqrt(y1**2 + y2**2)
    theta = jnp.arctan2(y2,y1) - 0.5*norm
    x1 = norm*jnp.cos(theta)
    x2 = norm*jnp.sin(theta)
    return np.stack([x1,x2],axis=1)

def chain_sampler(size=10,D=2,seed=2312, sigma=6, f=mean_f, g=variance_g):
    assert D > 1
    np.random.seed(seed)
    X = []
    x = np.random.normal(sigma**2/4, sigma, size=size)
    X.append(x)
    for d in range(1,D):
        x = f(x) + np.sqrt(g(x)) * np.random.normal(0,1, size=size)
        X.append(x)
    
    return np.stack(X,axis=1)

def tree_sampler(size=10,D=2,seed=2312, sigma=6, f=mean_f, g=variance_g):
    assert D > 1
    np.random.seed(seed)
    X = []
    x0 = np.random.normal(sigma**2/4, sigma, size=size)
    X.append(x0)
    for d in range(1,D):
        x = f(x0) + np.sqrt(g(x0)) * np.random.normal(0,1, size=size)
        X.append(x)
    
    return np.stack(X,axis=1)


from collections import namedtuple
rv = namedtuple('rv','logp sampler xlim ylim name')

warp = rv(logp_warp, warp_sampler, (-2,2),(-4,4),'warp')
banana = rv(logp_banana,banana_sampler, (-30,30),(-15,60),'banana')
cross = rv(logp_cross, cross_sampler,(-6,6),(-6,6),'cross')
neal = rv(logp_neal, neal_sampler,(-10,20),(-35,35),'neal')
chain = rv(logp_chain_nD, chain_sampler, None, None,'chain')
tree = rv(logp_tree_nD, tree_sampler,None,None,'tree')
