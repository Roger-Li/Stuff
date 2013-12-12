import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from utility import compute_grid
from pycuda.compiler import SourceModule

drv.get_version()
drv.get_driver_version()

m = SourceModule("""
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

extern "C" {
__global__ void truncnorm_kernel(float *vals, int N, float *mu, float *sigma, 
    float *lo, float *hi, int dbg)
{
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;

    // setup the RNG:
    curandState rng_state;
    curand_init(73+37*idx, 0, 0, &rng_state);

    if (idx < N)
    {
        //if (dbg)
        // {
        //   printf("thread idx: %04d\\n", idx);
        // }
        if (!isfinite(lo[idx]) && !isfinite(hi[idx]))
        {
            vals[idx] = mu[idx] + sigma[idx] * curand_normal(&rng_state);
        } // no truncation

        else if (!isfinite(hi[idx]))
        {
            float mu_neg = (lo[idx] - mu[idx]) / sigma[idx];
            float alpha = (mu_neg + sqrtf(mu_neg*mu_neg+4)) / 2;
            float z, psi, u;
            while (true)
            {
                z = mu_neg + logf(1-curand_uniform(&rng_state))/(-alpha);
                if (mu_neg < alpha)
                {
                    psi = expf(-(alpha-z)*(alpha-z)/2);
                }
                else
                {
                    psi = expf(-(mu_neg-alpha)*(mu_neg-alpha)/2-(alpha-z)*(alpha-z)/2);
                }
                u = curand_uniform(&rng_state);
                if (psi >= u)
                {
                    break;
                }
            }
            vals[idx] = z * sigma[idx] + mu[idx];
        } // lower bound

        else if (!isfinite(lo[idx]))
        {
            float mu_pos = (hi[idx] - mu[idx]) / sigma[idx];
            float alpha = (- mu_pos + sqrtf(mu_pos * mu_pos + 4)) / 2;
            float z, psi, u;
            while (true)
            {
                z = - mu_pos + logf(1 - curand_uniform(&rng_state)) / (- alpha);
                if (- mu_pos < alpha)
                {
                    psi = expf(-(alpha-z)*(alpha-z)/2);
                }
                else
                {
                    psi = expf(-(-mu_pos-alpha)*(-mu_pos-alpha)/2-(alpha-z)*(alpha-z)/2);
                }
                u = curand_uniform(&rng_state);
                if (psi >= u)
                {
                    break;
                }
            }
            vals[idx] = - z * sigma[idx] + mu[idx];
        } // truncated with higher bound

        else
        {
            float mu_neg = (lo[idx] - mu[idx]) / sigma[idx];
            float mu_pos = (hi[idx] - mu[idx]) / sigma[idx];
            float z, psi, u;
            while (true)
            {
                z = mu_neg + curand_uniform(&rng_state) * (mu_pos - mu_neg);
                if (mu_pos < 0)
                {
                    psi = expf((mu_pos * mu_pos - z * z) / 2);
                }
                else if (mu_neg > 0)
                {
                    psi = expf((mu_neg * mu_neg - z * z) / 2);
                }
                else
                {
                    psi = expf(- z * z / 2);
                }
                u = curand_uniform(&rng_state);
                if (psi >= u)
                {
                    break;
                }
            }
            vals[idx] = z * sigma[idx] + mu[idx];
        } // two-sided truncation

    }
    return;
}
}
""", include_dirs=['/usr/local/cuda/include/'], no_extern_c=1)

truncnorm = m.get_function('truncnorm_kernel')

for i in xrange(1, 9):
    n = np.int32(10**i)
    grid_dims, block_dims = compute_grid(n)

    mu = 2.0 * np.ones(n).astype(np.float32)
    sigma = 1.0 * np.ones(n).astype(np.float32)
    lo = 0.0 * np.ones(n).astype(np.float32)
    hi = 1.5 * np.ones(n).astype(np.float32)

    dest = np.zeros(n).astype(np.float32)

    start = drv.Event()
    end = drv.Event()

    start.record()
    truncnorm(drv.Out(dest), n, drv.In(mu), drv.In(sigma), drv.In(lo), drv.In(hi), np.int32(1), block=tuple(block_dims), grid=tuple(grid_dims))
    end.record()
    
    end.synchronize()
    gpu_secs = start.time_till(end)*1e-3
    print("gpu time (1e{}): {}".format(i, gpu_secs))