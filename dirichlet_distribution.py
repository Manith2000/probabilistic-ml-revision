import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gammaln

def dirichlet_pdf_vectorized(points, alpha):
    log_beta_func = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    log_prod_x = np.sum((alpha - 1) * np.log(points), axis=1)
    log_pdf = log_prod_x - log_beta_func
    return np.exp(log_pdf)

def plot_dirichlet(alpha, n_points=60):
    alpha = np.asarray(alpha)
    if alpha.size != 3:
        raise ValueError("Alpha parameter must have 3 elements for a 3D plot.")

    corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    triangle = plt.matplotlib.tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = plt.matplotlib.tri.UniformTriRefiner(triangle)
    tri_mesh = refiner.refine_triangulation(subdiv=n_points)
    
    points = np.c_[tri_mesh.x, tri_mesh.y, 1 - tri_mesh.x - tri_mesh.y]
    
    valid_points_mask = (points > 1e-10).all(axis=1)
    pdf_values = np.zeros(len(points))
    
    pdf_values[valid_points_mask] = dirichlet_pdf_vectorized(points[valid_points_mask], alpha)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(points[:, 0], points[:, 1], pdf_values, cmap='viridis', antialiased=False)

    ax.set_title(f'Dirichlet Distribution (alpha={alpha})', fontsize=16)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)

    ax.view_init(elev=30, azim=-45)
    
    plt.show()

if __name__ == "__main__":
    plot_dirichlet([10, 1, 1])
