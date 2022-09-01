from .src import *
from copulas.datasets import sample_trivariate_xyz

if __name__ == "__main__":
    k = sample_trivariate_xyz()
    K_set = 25

    ss_cop = LocalFPCA(K = K_set, data = k)
    ss_cov = LocalCov(K = K_set, data = k)
    
    dfSyn_cop = ss_cop.fit(sample_size=None)
    dfSyn_cov = ss_cov.fit(sample_size=None)

    ss_cop.comparePlots(['x','y','z'])
    ss_cov.comparePlots(['x','y','z'])