import jax
import jax.numpy as jnp
import jax.random as rd
import numpy as np
from functools import partial
from tqdm import tqdm
import optax as optim

import numpy as np
from scipy.spatial.distance import pdist, squareform
from functools import wraps

def get_median_h(X):
    '''
      median trick: A adaptive method to choose h in kernel function.
    '''
    X = jax.device_get(X)
    sq_dist = pdist(X)
    pairwise_dists = squareform(sq_dist)**2
    h = np.median(pairwise_dists)
    h = np.sqrt(0.5 * h / np.log(X.shape[0] + 1))
    return h

def inverse_permutation(idx):
    '''
     fast for numpy array.
     slow for jax array.
    '''
    idx = jax.device_get(idx)
    idy = np.zeros_like(idx,dtype=int)
    for i in range(len(idx)):
        idy[idx[i]] = i
    return jnp.array(idy)


import time
def calc_time(f):
    
    @wraps(f)
    def _f(*args, **kwargs):
        t0 = time.time()
        y = f(*args, **kwargs)
        t1 = time.time()
        print(f' {f.__name__} complete, time cost(s):{t1-t0}')
        return y
    return _f

def jax_get_median_h(X):
    '''
        这个函数比上面这个慢很多，这是什么原因？
    '''
    pairwise_dists = jnp.sum((X[None,:,:] - X[:,None,:])**2,axis=-1)
    h = jnp.median(pairwise_dists)
    h = jnp.sqrt(0.5 * h / jnp.log(X.shape[0] + 1))
    return h
  


def SVGD_sampler(logp, k, optimizer, batch_size=100, median_trick=True, message_passing=False):
    '''
      Input:
        logp: logrithm function of target distribution.  shape mapping: (p) --> (1)   
        k:    kernel function.                           shape mapping: (p),(p) --> (1)
        optimizer: an optax optimizer.
        batch_size: A interger. mini batch size.
        median_trick: A bool variable. Use adaptive kernel function scale or not.
      Output:
        A SVGD sampler.
    
    '''
    
    s = jax.grad(logp)                    # gradient of logp.     (p) --> (p)
    S = jax.vmap(s,in_axes=0, out_axes=0) # vmap version of s.  (n,p) --> (n,p)
    nabla_k = jax.grad(k, argnums=0)      # gradient of kernel function w.r.t the first arguments of k.


    def outer_op(f):
        '''
        Return a function that is a double vmap version of f.
        Input:  a function f.
          f: (p) , (p), (1) --> (1)
        return: a function F.
          F: (n,p),(n,p),(1) --> (n,n)
        '''
        K1 = jax.vmap(f, in_axes=(0,None,None), out_axes=0)  # (p) x (n,p)  --> n
        K = jax.vmap(K1, in_axes=(None,0,None), out_axes=1) # (n,p) x (n,p) --> (n,n)
        return K


    K = outer_op(k)               # K calculate all interactions between all pairs (xi,xj)
    nabla_K = outer_op(nabla_k)


    def phi(X,h,W):
        '''
        Calculate gradient respect to KL divergence between empirical(X) and target distribution.
        '''
        return  (W*K(X,X,h)) @ S(X) + jnp.sum(W[:,:,None] * nabla_K(X,X,h), axis=0)



    def phi_batch(X,h, idx, idy, W):
        '''
        Calculate gradient respect to KL divergence between empirical(X) and target distribution.
        '''
        N = X.shape[0]

        if batch_size < N:
            p = X.shape[1]
            assert N % batch_size == 0
            n = N//batch_size
            Xp = X[idx]
            Xp = Xp.reshape([n,batch_size,p])
            Yp = jax.vmap(phi, in_axes=(0,None,None), out_axes=(0))(Xp, h, W)
            Yp = Yp.reshape([N,p])
            return Yp[idy]
        else:
            return 1/N*(K(X,X,h) @ S(X) + jnp.sum(nabla_K(X,X,h), axis=0))



    @jax.jit
    def update(X, opt_state, h, idx, idy, weight):
        '''
            update function in SVGD.
        '''
        grads = -phi_batch(X,h,idx,idy,weight)
        updates, opt_state = optimizer.update(grads, opt_state, X)
        X = optim.apply_updates(X, updates)
        return X, opt_state


    @calc_time
    def SVGD(X, iter=1000, batch_rng=rd.PRNGKey(233)):
        '''
        SVGD sampler.
        Input:
          X: initial particles. A jax numpy which has shape (N,p).  N: number of particles.   p: particle dimensions.
          iter: A integer.  The number of iterations.
          batch_rng: [optional]. random seed for stochastic batch division.
        Return:
          samples: A list of samples with length iter.
          samples[i] is a jax numpy which has shape (N,p), the particles at iteration i.
        '''

        samples = []
        opt_state = optimizer.init(X)
        N = X.shape[0]
        
        weight =jnp.ones([batch_size,batch_size])*(N-1)/N/(batch_size-1) + jnp.diag(jnp.ones(batch_size)*(1/N - (N-1)/N/(batch_size-1)))
        h = 1.0
        for i in tqdm(range(iter),desc='sampling'):
            key = rd.fold_in(batch_rng, i)
            idx = rd.permutation(key, N)
            idy = inverse_permutation(idx)
            if median_trick:
                h = get_median_h(X[idx[0:int(N/10)]])
            X, opt_state = update(X, opt_state, h, idx, idy, weight)
            samples.append(jax.device_get(X))
        return samples
    

    return SVGD


def MP_SVGD_sampler(logp_list, k_list, idx_list, optimizer =  optim.adam(learning_rate=0.1), median_trick=True):
    '''
      Input:
        logp: logrithm function of target distribution.  shape mapping: (p) --> (1)   
        k:    kernel function.                           shape mapping: (p),(p) --> (1)
        optimizer: an optax optimizer.
        batch_size: A interger. mini batch size.
        median_trick: A bool variable. Use adaptive kernel function scale or not.
      Output:
        A SVGD sampler.
    
    '''
    D = len(logp_list)

    # logp: () x (p-1) --> ()
    # s: () x (p-1) --> ()
    # S: (n,) x (n,p-1) --> (n,)

    s = [jax.grad(logp_list[d], argnums=0) for d in range(D)]                    # gradient of logp.    () x (p-1) --> ()
    S = [jax.vmap(s[d],in_axes=(0,0), out_axes=0)  for d in range(D)]      # vmap version of s.  (n,) x (n,p-1) --> (n,)

    def get_nablad_K(k):
        # k: (p,) x (p,) --> ()
        def element_K(x_sd, yd, y_gammad, h):
            # (p) x () x (p-1) x () --> ()
            return k(x_sd, jnp.concatenate([yd[...,None],y_gammad],axis=-1), h)
        return jax.grad(element_K, argnums=1)   # (p) x () x (p-1) x () --> ()

    # (p) x () x (p-1) x () --> ()
    nablad_k = [get_nablad_K(k_list[d])  for d in range(D)]      # gradient of kernel function w.r.t the yd of k(x_sd, y_sd).


    def outer_op(f):
        '''
        Return a function that is a double vmap version of f.
        Input:  a function f.
          f: (p) , (p), () --> ()
        return: a function F.
          F: (n,p),(n,p),() --> (n,n)
        '''
        K1 = jax.vmap(f, in_axes=(0,None,None), out_axes=0)  # (n,p) x (p) x ()  --> (n,)
        K = jax.vmap(K1, in_axes=(None,0,None), out_axes=1) # (n,p) x (n,p) x () --> (n,n)
        return K

    def outer_k(f):
        '''
        f: (p) x () x (p-1) x () --> ()
        output: (n,p) x (n,) x (n,p-1) x () --> (n,n)
        '''
        K1 = jax.vmap(f, in_axes=(0,None,None,None), out_axes=0)  # (n,p) x () x (p-1) x ()  --> (n,)
        K = jax.vmap(K1, in_axes=(None,0,0,None), out_axes=1) # (n,p) x (n,) x (n,p-1) x () --> (n,n)
        return K

    # (n,p) x (n,p) --> (n,n)
    K = [outer_op(k_list[d]) for d in range(D)]      

    # (n,p) x (n,) x (n,p-1) x () --> (n,n)
    nablad_K = [outer_k(nablad_k[d]) for d in range(D)] 


    def construct_phi(kernel,nablad_kernel,grad_logp):
        ''' 
        kernel: (n,p),(n,p),() --> (n,n)
        nablad_kernel: (n,p) x (n,) x (n,p-1) x () --> (n,n)
        grad_logp: (n,) x (n,p-1) --> (n,)
        '''
        def phi(xd,x_gammad,h):
            '''
            Calculate gradient respect to KL divergence between empirical(X) and target distribution.
            xd: (n)
            x_gammad: (n,p-1)
            h: ()
            '''
            X = jnp.concatenate([xd[...,None],x_gammad],axis=-1)
            return  1/X.shape[0]*(kernel(X,X,h) @ grad_logp(xd,x_gammad) + jnp.sum(nablad_kernel(X,xd,x_gammad,h), axis=1))

        return phi
    

    phi = [construct_phi(K[d], nablad_K[d], S[d]) for d in range(D)]


    def construct_update(phi,optimizer):
        @jax.jit
        def update(xd,x_gammad,opt_state,h):
            grads = -phi(xd,x_gammad,h)
            updates,opt_state = optimizer.update(grads, opt_state)
            return updates, opt_state
        return update

    update = [construct_update(phi[d], optimizer) for d in range(D)]



    @calc_time
    def SVGD(X, iter=1000):
        '''
        SVGD sampler.
        Input:
          X: initial particles. A jax numpy which has shape (N,p).  N: number of particles.   p: particle dimensions.
          iter: A integer.  The number of iterations.
          batch_rng: [optional]. random seed for stochastic batch division.
        Return:
          samples: A list of samples with length iter.
          samples[i] is a jax numpy which has shape (N,p), the particles at iteration i.
        '''

        samples = []
        N = X.shape[0]
        h = 1.0

        opt_state = [optimizer.init(X[:,d]) for d in range(D)]
        for i in tqdm(range(iter),desc='sampling'):
            for d in range(D):
                Sd = idx_list[d]
                xd = X[:,d]
                x_gammad = X[:,Sd[1:]]
                if median_trick:
                    h = get_median_h(X[0:N//10, Sd])
                  
                update_d,opt_state_d = update[d](xd,x_gammad,opt_state[d],h)
                opt_state[d] = opt_state_d
                X = X.at[:,d].add(update_d)
            samples.append(jax.device_get(X))
        return samples
    

    return SVGD
