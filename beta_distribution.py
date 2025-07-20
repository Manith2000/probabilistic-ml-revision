''' Beta distribution 
a pdf which is defined on the interval [0, 1] and is parameterized by two positive shape parameters, alpha and beta. Relies on the Beta function, which relies on the Gamma function. Implementation given below'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# manual implementation of beta function, and beta distribution using numpy
# Note: for practical reasons, I use gamma function from scipy (involves some approximation probably through Lanczos approximation or Stirling's approximation)    

def beta_function(alpha, beta):
    """Beta function using the gamma function"""
    return gamma(alpha) * gamma(beta) / gamma(alpha + beta)

def beta_distribution(x, alpha, beta):
    """Beta distribution pdf for a given x, alpha, and beta"""
    if x < 0 or x > 1:
        raise ValueError("x must be in the interval [0, 1]")
    return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / beta_function(alpha, beta)


def main():
    # plot beta distribution for multiple values of alpha and beta
    alpha = [0.5, 1.0, 2.0, 5.0]
    beta = [0.5, 1.0, 2.0, 5.0]
    # note: need to avoid division by zero by avoiding x=0 and x=1 in the plot
    x_values = np.linspace(1e-5, 1 - 1e-5, 100)
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 5)
    for a, b in zip(alpha, beta):
        pdf_values = [beta_distribution(x, a, b) for x in x_values]
        plt.plot(x_values, pdf_values, label=f'alpha={a}, beta={b}')
    plt.legend()
    plt.title('Beta Distribution for Different Alpha and Beta Values')
    plt.show()
    
if __name__ == "__main__":
    main()