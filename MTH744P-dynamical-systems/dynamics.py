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
    ax.axis("equal")
    

def plot_bifurcation_diagram(f, r_values, n_arrows, xmin, xmax, h=0.1, levels=0):
    xrange = np.linspace(xmin, xmax, n_arrows)
    head_width = len(r_values) / 40
    rmin, rmax = min(r_values), max(r_values)
    XX = np.mgrid[xmin:xmax+h:h, rmin:rmax+h:h]
    Z = np.apply_along_axis(lambda xx: f(*xx), 0, XX)
    for r in r_values:
        flows = np.sign(f(x=xrange, r=r))
        for x, flow_value in zip(xrange, flows):
            color = "tab:red" if flow_value == -1 else "tab:orange"
            plt.axhline(y=r, c="tab:gray", linestyle="--", alpha=0.02)
            plt.arrow(x, r, flow_value * 0.2, 0, width=0.001, head_width=head_width,
                          color=color, head_length=0.1)
    plt.contour(*XX, Z, levels=levels)
    plt.ylabel("r", fontsize=15)
    plt.xlabel("x", fontsize=15)


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