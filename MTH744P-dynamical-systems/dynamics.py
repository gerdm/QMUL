import numpy as np
import matplotlib.pyplot as plt

def plot_circular_dynamics(theta_inits, f, eps=0.2, rtol=0.02, ax=None):
    """
    Plot the phase portrait of a dynamical system f(θ)
    on the circle

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
    """
    theta_circular = np.linspace(0, 2 * np.pi, 100)
    ax = plt.subplots()[1] if ax is None else ax
    for theta in theta_inits:
        dx, dy = np.cos(theta) * eps, -np.sin(theta) * eps

        dx = np.cos(theta) * eps * f(theta)
        dy = -np.sin(theta) * eps * f(theta)

        if not np.isclose(dx, 0, rtol=rtol) or not np.isclose(dy, 0, rtol=rtol):
            ax.arrow(np.sin(theta) - dx, np.cos(theta) - dy, dx, dy, width=0.035)
        else:
            ax.scatter(np.sin(theta), np.cos(theta), c="black", s=50)

    ax.plot(np.sin(theta_circular), np.cos(theta_circular))
    ax.axis("equal");

