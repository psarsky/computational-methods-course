"""Sphere transformation and SVD visualization"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def create_unit_sphere(resolution=30):
    """Creates a unit sphere using spherical coordinates."""
    s = np.linspace(0, 2*np.pi, resolution)
    t = np.linspace(0, np.pi, resolution)

    s_grid, t_grid = np.meshgrid(s, t)

    x = np.cos(s_grid) * np.sin(t_grid)
    y = np.sin(s_grid) * np.sin(t_grid)
    z = np.cos(t_grid)

    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    return points, x, y, z


def transform_sphere(points, A):
    """Transforms points using matrix A."""
    return points @ A.T


def plot_ellipsoid(sphere_coords, transformed_points, singular_values, V, title):
    """Plots the ellipsoid defined by the transformed points and the singular values."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, _, _ = sphere_coords
    points_transformed = transformed_points.reshape(x.shape[0], x.shape[1], 3)

    ax.plot_surface(points_transformed[:,:,0], points_transformed[:,:,1], points_transformed[:,:,2],
                   color='b', alpha=0.3)

    origin = np.zeros(3)

    for i in range(3):
        axis = singular_values[i] * V.T[i]
        ax.quiver(origin[0], origin[1], origin[2],
                 axis[0], axis[1], axis[2],
                 color=['r', 'g', 'y'][i], linewidth=3, arrow_length_ratio=0.1,
                 label=f'Półoś {i+1}: {singular_values[i]:.2f}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_val = np.max(np.abs(transformed_points)) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    ax.legend()

    return fig, ax


def visualize_svd_steps(sphere_points, A, sigma, VT, title):
    """Visualizes the steps of SVD."""
    fig = plt.figure(figsize=(18, 6))

    # SV^T
    ax1 = fig.add_subplot(131, projection='3d')
    sv_points = sphere_points @ VT.T
    sv_points_matrix = sv_points.reshape(30, 30, 3)
    ax1.plot_surface(sv_points_matrix[:,:,0],
                     sv_points_matrix[:,:,1],
                     sv_points_matrix[:,:,2], 
                     color='r',
                     alpha=0.3)
    ax1.set_title("SV^T")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # SΣV^T
    ax2 = fig.add_subplot(132, projection='3d')
    sigma_matrix = np.diag(sigma)
    ssigma_vt_points = sv_points @ sigma_matrix
    ssigma_vt_points_matrix = ssigma_vt_points.reshape(30, 30, 3)
    ax2.plot_surface(ssigma_vt_points_matrix[:,:,0],
                     ssigma_vt_points_matrix[:,:,1],
                     ssigma_vt_points_matrix[:,:,2],
                     color='g',
                     alpha=0.3)
    ax2.set_title("SΣV^T")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # SU^TΣV^T = SA
    ax3 = fig.add_subplot(133, projection='3d')
    transformed_points = sphere_points @ A.T
    transformed_points_matrix = transformed_points.reshape(30, 30, 3)
    ax3.plot_surface(transformed_points_matrix[:,:,0],
                     transformed_points_matrix[:,:,1],
                     transformed_points_matrix[:,:,2],
                     color='b',
                     alpha=0.3)
    ax3.set_title("SU^TΣV^T = SA")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    max_val = np.max(np.abs(transformed_points)) * 1.2
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def main():
    """Main function to run the transformations and visualizations."""
    # 1. Create unit sphere
    sphere_points, x_sphere, y_sphere, z_sphere = create_unit_sphere(resolution=30)
    sphere_coords = (x_sphere, y_sphere, z_sphere)

    # 2. Transformations
    # A1: scaling
    A1 = np.diag([2, 1, 0.5])

    # A2: rotation + scaling
    A2 = np.array([
        [1, 0.5, 0],
        [0.5, 2, 0],
        [0, 0, 1.5]
    ])

    # A3: More complex transformation
    A3 = np.array([
        [3, 1, 0.5],
        [1, 2, 0],
        [0.5, 0, 1]
    ])

    transformed_points1 = transform_sphere(sphere_points, A1)
    transformed_points2 = transform_sphere(sphere_points, A2)
    transformed_points3 = transform_sphere(sphere_points, A3)
    plt.show()

    # 3. SVD
    _, sigma1, VT1 = linalg.svd(A1)
    _, sigma2, VT2 = linalg.svd(A2)
    _, sigma3, VT3 = linalg.svd(A3)

    plot_ellipsoid(sphere_coords, transformed_points1, sigma1, VT1.T, "A1")
    plot_ellipsoid(sphere_coords, transformed_points2, sigma2, VT2.T, "A2")
    plot_ellipsoid(sphere_coords, transformed_points3, sigma3, VT3.T, "A3")
    plt.show()

    # 4. Large semi-axis ratio
    A4 = np.diag([20, 5, 0.1])

    _, sigma4, V4 = linalg.svd(A4)

    transformed_points4 = transform_sphere(sphere_points, A4)
    plot_ellipsoid(sphere_coords, transformed_points4, sigma4, V4, "Ellipsoid with large semi-axis ratio")
    plt.show()

    # 5. SVD steps
    visualize_svd_steps(sphere_points, A1, sigma1, VT1, "SVD steps for A1 matrix")
    plt.show()


if __name__ == "__main__":
    main()
