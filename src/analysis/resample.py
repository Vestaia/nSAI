import torch

#Work in progress

def cubic(tensor, errors=None, sigma=1, resolution=128):
    result = torch.empty_like(tensor)
    return result

def linear(tensor, errors=None, sigma=1, resolution=128):
    result = torch.empty_like(tensor)
    for q in range(torch.min(tensor[:,0]), torch.max(tensor[:,0]), resolution):
        closest = torch.argmin(torch.abs(tensor - q), dim=0)
        
    return result
