""" Gamma distribution """


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_function

def gamma_distribution(x, shape, rate=1):
    """gamma distribution pdf"""
    if x < 0:
        raise ValueError("x must be non-negative")
    return (x ** (shape - 1)) * np.exp(-x / rate) / (gamma_function(shape) * (rate ** shape))

def exponential_distribution(x, rate=1):
    """Exponential distribution as a special case of gamma distribution"""
    return gamma_distribution(x, shape=1, rate=rate)

def plot_exponential_distribution(rate=1):
    """Plot the exponential distribution for a given rate"""
    x_values = np.linspace(0, 10, 100)
    pdf_values = [exponential_distribution(x, rate) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, pdf_values, label=f'Exponential Distribution (rate={rate})')
    plt.title('Exponential Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density Function')
    plt.legend()
    plt.grid()
    plt.show()

def plot_gamma_distribution():
    # plot gamma distribution for multiple values of shape and rate
    shape = [1, 2, 3, 5]
    rate = [1, 2, 3, 5]
    x_values = np.linspace(0, 10, 100)
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    for s, r in zip(shape, rate):
        pdf_values = [gamma_distribution(x, s, r) for x in x_values]
        plt.plot(x_values, pdf_values, label=f'shape={s}, rate={r}')
    plt.legend()
    plt.title('Gamma Distribution for Different Shape and Rate Values')
    plt.show()
    
    
    
def main():
    plot_exponential_distribution(rate=1.0)
    plot_gamma_distribution()
    
if __name__ == "__main__":
    main()