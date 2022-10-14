import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import ite
co = ite.cost.BDKL_KnnK(k=10)


def get_KSD(logp, k):
    # logp: (p) --> () 
    # s: (p) --> (p)
    s = jax.grad(logp, argnums=0) 

    # k: (p) x (p) x () --> ()
    # k0: (p) x (p) x () --> (p)
    # k1: (p) x (p) x () --> (p)
    # kd: (p) x (p) x () --> (p,p)
    k0 = jax.grad(k,argnums=0)
    k1 = jax.grad(k,argnums=1)
    kd = jax.jacobian(k0, argnums=1)

    def up(x,xp,h):
        '''
            (p) x (p) x () --> ()
        '''
        return k(x,xp)*jnp.dot(s(x),s(xp)) + jnp.dot(s(x),k1(x,xp)) + jnp.dot(k0(x,xp), s(xp)) + jnp.trace(kd(x,xp))

    
    def outer_op(f):
        '''
        Return a function that is a double vmap version of f.
        Input:  a function f.
          f: (p) , (p),() --> ()
        return: a function F.
          F: (n,p),(n,p),() --> (n,n)
        '''
        K1 = jax.vmap(f, in_axes=(0,None,None), out_axes=0)  # (p) x (n,p) x () --> (n,)
        K = jax.vmap(K1, in_axes=(None,0,None), out_axes=1) # (n,p) x (n,p) x () --> (n,n)
        return K

    
    Up = outer_op(up)

    @jax.jit
    def KSD(X,h):
        n = X.shape[0]
        U = Up(X,X,h)
        return 1/n/(n-1)*(jnp.sum(U) - jnp.trace(U))


    return KSD


def KL_path(samples, truth, exchange=True):
    N = len(samples)
    if exchange:
        s = [co.estimation(samples[i],truth) for i in range(N)]   # This is right !
    else:
        s = [co.estimation(truth, samples[i]) for i in range(N)]
    return s

def mean_variance(X):
    N = X.shape[0]
    m1 = np.mean(X,axis=0)
    m2 = 1/N*X.T@X - m1[:,None] @ m1[None,:]
    return m1,m2

def moment_path(samples, truth):
    N = len(samples)
    ms = [mean_variance(samples[i]) for i in range(N)]
    mt = mean_variance(truth)
    return ms,mt

def show_result(samples,truth,kl,ksd,xlim=(-2,2),ylim=(-4,4),x_axis=0,y_axis=1):
    n = -1
    N = samples[n].shape[0]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    if xlim != None:
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.scatter(samples[n][:,x_axis],samples[n][:,y_axis], s=3,label='samples')
    plt.legend()
    plt.subplot(1,2,2)
    if xlim != None:
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.scatter(truth[0:N,x_axis],truth[0:N,y_axis], s=3,label='truth')
    plt.legend()
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.xlabel('iteration')
    plt.ylabel('KL divergence')
    plt.plot(kl)

    plt.subplot(1,2,2)
    plt.xlabel('iteration')
    plt.ylabel('KSD')
    plt.plot(ksd)