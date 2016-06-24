import numpy as np
import scipy.linalg as la
from numpy.testing import assert_almost_equal

def euc_dist(x, y):
    return la.norm(x-y)

def d(x, Y):
    dist = list(map(lambda yi: euc_dist(x, yi), Y))
    return min(dist)

def cost(Y, C):
    return sum(d(yi, C)**2 for yi in Y)

def kmeans_plus(X, k):
    idx = np.random.choice(X.shape[0], 1)
    C = X[idx, :]
    while(C.shape[0] < k):
        cost_X = cost(X, C)
        prob = list(map(lambda xi: d(xi,C)**2/cost_X, X))
        Ct = X[np.random.choice(X.shape[0], size=1, p=prob),:]
        if(Ct.tolist() not in C.tolist()):
            C = np.r_[C, Ct]
    return C

def test_non_negativity():
    for i in range(10):
        u = np.random.normal(10,1,(100,10))
        v = np.random.normal(10,1,(100,10))
        assert euc_dist(u, v) >= 0

def test_coincidence_when_zero():
    u = np.zeros(3)
    v = np.zeros(3)
    assert euc_dist(u, v) == 0
        
def test_symmetry():
    for i in range(10):
        u = np.random.random(3)
        v = np.random.random(3)
        assert euc_dist(u, v) == euc_dist(v, u)

def test_triangle():
    u = np.random.random(3)
    v = np.random.random(3)
    w = np.random.random(3)
    assert euc_dist(u, w) <= euc_dist(u, v) + euc_dist(v, w)

def test_known():
    u = np.array([0,0])
    v = np.array([3, 4])
    assert_almost_equal(euc_dist(u, v), 5)

def test_dist_non_negativity():
    for i in range(10):
        u = np.array([10,1])
        v = np.random.normal(10,1,(100,2))
        assert d(u, v) >= 0
        
def test_dist_coincidence_when_zero():
    u = np.zeros(3)
    v = np.zeros(3)
    assert d(u, v) == 0
    
def test_cost_non_negativity():
    for i in range(10):
        u = np.random.normal(10,1,(100,2))
        v = np.random.normal(10,1,(100,2))
        assert cost(u, v) >= 0

def test_kmeansplus_centroidnumbers():
    u = np.random.normal(10,1,(100,2))
    k = 3
    assert len(kmeans_plus(u,k)) == k