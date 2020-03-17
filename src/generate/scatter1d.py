import sasmodels
from sasmodels.core import load_model
from sasmodels import direct_model
from sasmodels import data
import numpy as np
import matplotlib.pyplot as plt
import torch
import ray
import psutil

N_CPUS = psutil.cpu_count(logical=False)

POINTS_PER_SAMPLE = 450

#Updates parameter dictionary with probability distributions
def __probdist__(prefix):
    pd = prefix + '_pd'
    pdn = prefix + '_pd_n'
    sigma = prefix + '_pd_nsigmas'
    pdt = prefix + '_pd_type'

    pds = np.random.uniform(low=-2, high=0, size=1)[0]
    pds = 10**pds
    pd_type = np.random.randint(0, 1)
    var = {}
    var[pd] = pds
   
    if(pd_type):
        var[pdt] = 'lognormal'
        var[pdn] = 25
        var[sigma] = 4
    else:
        var[pdt] = 'gaussian'        
        var[pdn] = 10
        var[sigma] = 3
    return var

#Generates scattering patterns for spheres
def gen_sphere(parameters):

    info = sasmodels.core.load_model_info('sphere@hardsphere')
    model = sasmodels.core.build_model(info)

    q = torch.sort(torch.pow(10, 4 * (torch.rand(POINTS_PER_SAMPLE) - 1)))[0]
    qn = q.numpy()
    data = sasmodels.data.empty_data1D(qn)

    calculator = sasmodels.direct_model.DirectModel(data, model)
    smear = sasmodels.resolution.Pinhole1D(qn,qn*0.1,qn)

    I = calculator(**parameters)
    I = smear.apply(I)
    
    I = torch.Tensor(I)

    return torch.stack((q, I), 0)

#Thread helper for generating data
@ray.remote
def __generator_thread__(nsamples):
    parameters = {'scale':1, 'radius':100, 'sld':0, 'sld_solvent':1}
    parameters.update(__probdist__('radius'))
    data = torch.empty((nsamples, 2, POINTS_PER_SAMPLE))
    target = torch.empty((data.size(0), 1))
    for i in range(data.size(0)):
        parameters['radius'] = np.power(10, np.random.uniform(1, 3))
        data[i] = gen_sphere(parameters)
        target[i,0] = parameters['radius']

    return data, target
    
#Generates samples
def gen_data(nsamples):
    samples_per_thread = int(nsamples / N_CPUS) + (nsamples % N_CPUS > 1)
    data = torch.empty((nsamples, 2, POINTS_PER_SAMPLE))
    target = torch.empty((nsamples, 1))
    ray.init(num_cpus=N_CPUS)
    results = ray.get([__generator_thread__.remote(samples_per_thread) for i in range(N_CPUS)])

    #Concatenate results into tensor
    for i in range(N_CPUS-1):
        data[i*samples_per_thread:(i+1)*samples_per_thread] = results[i][0]
        target[i*samples_per_thread:(i+1)*samples_per_thread] = results[i][1]
    data[(N_CPUS-1)*samples_per_thread:] = results[i][0][:((nsamples-1)%samples_per_thread)+1]
    target[(N_CPUS-1)*samples_per_thread:] = results[i][1][:((nsamples-1)%samples_per_thread)+1]
    return data, target
