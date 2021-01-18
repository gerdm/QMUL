import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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


def x_dot(r, θ, r_dot, θ_dot):
    """
    Compute the change in a polar dynamical system in
    terms of the x-coordinate
    """
    return r_dot(r, θ) * np.cos(θ) - θ_dot(r, θ) * r * np.sin(θ)


def y_dot(r, θ, r_dot, θ_dot):
    """
    Compute the chage in a polar dynamical system in terms
    of the y-coordinate
    """
    return r_dot(r, θ) * np.sin(θ) + θ_dot(r, θ) * r * np.cos(θ)


def f_polar(S, r_dot, θ_dot):
    """
    Evaluate the direction of a polar dynamical
    system
    """
    r, θ = S
    Δr = r_dot(r, θ)
    Δθ = θ_dot(r, θ)
    return Δr, Δθ


def f_polar_to_cartesian(S, r_dot, θ_dot):
    """
    Evaluate the direction (in the cartesian place)
    of a polar dynamical system
    """
    r, θ = S
    ẋ = x_dot(r, θ, r_dot, θ_dot)
    ẏ = y_dot(r, θ, r_dot, θ_dot)
    return ẋ, ẏ


def to_polar_space(x, y):
    """
    Transform values is R^2 to a polar plane
    """
    r = np.sqrt(x ** 2 + y ** 2)
    θ = np.arctan2(y, x)
    return r, θ


def to_cartesian_space(r, θ):
    """
    Transform values in a polar plane to R^2
    """
    x = r * np.cos(θ)
    y = r * np.sin(θ)
    return y, x


def evaluate_polar_system(r_dot, θ_dot, xmin, xmax, ymin, ymax, xstep=0.01, ystep=0.01):
    R = np.mgrid[ymin:ymax:ystep, xmin:xmax:xstep][::-1]
    S = np.stack(to_polar_space(*R))
    Rdot = np.stack(f_polar_to_cartesian(S, r_dot, θ_dot))
    return R, Rdot


def plot_polar_system(r_dot, θ_dot, xmin, xmax, ymin, ymax, xstep=0.01, ystep=0.01, alpha=1, ax=None, **kwargs):
    ax = plt.subplot() if ax is None else ax
    R, Rdot = evaluate_polar_system(r_dot, θ_dot, xmin, xmax, ymin, ymax, xstep=0.01, ystep=0.01)
    stream = ax.streamplot(*R, *Rdot, **kwargs)
    stream.lines.set_alpha(alpha)
    return stream


def integrate_polar_system(r_dot, θ_dot, initial_conditions, t_end, t_steps):
    """
    Solve an ODE of a polar system and return
    its x and y values
    """
    n_conditions, m = initial_conditions.shape
    integrations = np.zeros((n_conditions, t_steps, m))
    
    integration_time = np.linspace(0, t_end, t_steps)
    for n, initial_condition in enumerate(initial_conditions):
        polar_initial_condition = to_polar_space(*initial_condition)
        polar_solution = odeint(lambda pos, t: f_polar(pos, r_dot, θ_dot),
                                polar_initial_condition,
                                integration_time)
        cartesian_solution = np.stack(to_cartesian_space(*polar_solution.T)).T
        integrations[n, ...] = cartesian_solution
    
    return integrations


def plot_solution_polar_system(r_dot, θ_dot, initial_conditions, t_end, t_steps=100, ax=None, **kwargs):
    ax = plt.subplot() if ax is None else ax
    solutions = integrate_polar_system(r_dot, θ_dot, initial_conditions, t_end, t_steps)
    for solution in solutions:
        ax.plot(*solution.T, **kwargs)