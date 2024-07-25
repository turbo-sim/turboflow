
import numpy as np
from scipy.stats import qmc

def latin_hypercube_sampling(bounds, n_samples):
    """
    Generates samples using Latin Hypercube Sampling.

    Parameters:
    bounds (list of tuples): A list of (min, max) bounds for each variable.
    n_samples (int): The number of samples to generate.

    Returns:
    np.ndarray: An array of shape (n_samples, n_variables) containing the samples.
    """
    n_variables = len(bounds)
    # Create a Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=n_variables)
    
    # Generate samples in the unit hypercube
    unit_hypercube_samples = sampler.random(n=n_samples)
    
    # Scale the samples to the provided bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(unit_hypercube_samples, lower_bounds, upper_bounds)

    return scaled_samples

if __name__ == '__main__':

    # Example usage
    bounds = [(0, 10), (20, 30), (100, 200)]  # Bounds for each variable
    n_samples = 5  # Number of samples to generate

    samples = latin_hypercube_sampling(bounds, n_samples)
    print(samples)