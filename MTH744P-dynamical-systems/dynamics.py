import numpy as np
import matplotlib.pyplot as plt

def plot_circular_dynamics(theta_inits, f, eps=0.2, rtol=0.02, ax=None, n_points=100):
    """
    Plot the phase portrait of a dynamical system f(θ)
    on the circle with r = 1 via the transformation
    
                x = r * cos(θ)
                y = r * sin(θ)

    Parameters
    ----------
    theta_inits: np.array(M)
        Vector of starting angles
    f: function
        funtion of the form f(θ, *args)
    eps: float
        The relative shock of the derivative
    rtol: float
        Relative tolerance to catch a fixed point
    ax: matplotlib.axis or None
        (if any) plot the diagram on the given subplot
    n_points: int
        Number of points to draw the circle
    """
    theta_circular = np.linspace(0, 2 * np.pi, n_points)
    ax = plt.subplots()[1] if ax is None else ax
    for theta in theta_inits:
        dx = -np.sin(theta) * eps * f(theta)
        dy = np.cos(theta) * eps * f(theta)

        if not np.isclose(dx, 0, rtol=rtol) or not np.isclose(dy, 0, rtol=rtol):
            arrow_y = np.sin(theta) - dy
            arrow_x = np.cos(theta) - dx
            ax.arrow(arrow_x, arrow_y, dx, dy, width=0.035)
        else:
            ax.scatter(np.cos(theta), np.sin(theta), c="black", s=50)

    ax.plot(np.sin(theta_circular), np.cos(theta_circular))
    ax.axis("equal");
    




def linear_dynamics(A, vmin=-2, vmax=2, step=0.1):
    """
    Return the evaluation of a vector field for a linear
    system of the form
                ẋ = Ax
    
    Inputs
    ------
    A: np.array(2, 2)
        A real matrix with coeffients describing the dynamics
    """
    # The mpl streamplot requires the values of x to be equal
    X = np.mgrid[vmin:vmax:step, vmin:vmax:step][::-1]
    X_dot = np.einsum("ij,jnm->inm", A, X)
    return X, X_dot