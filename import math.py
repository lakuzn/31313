import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import sin, cos

def generate_point_seq(init_point, radius, angle, count):
    res = []
    for i in range(0, count):
        point = (init_point[0] + radius * cos(i * angle), init_point[1] + radius * sin(i * angle))
        res.append(point)
    return res
    
# defining func that calculates vector field w/ given x and y axis limits
def eq_quiver(rhs, limits, N=16):
    xlims, ylims, tlims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vfield = rhs(0.0, [x, y])
            u, v = vfield
            U[i][j] = u/(u**2+v**2)**.5
            V[i][j] = v/(u**2+v**2)**.5
    return xs, ys, U, V

def plot(rhs, ax, initial_points, limits):
    ''' rhs(t, X = [(x, y)]), initial_points = [(x_00, y_00),..], limits = [(x_min, x_max), (y_min, y_max), (t_min, t_max)]'''

    xlims, ylims, tlims = limits
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])

    xs, ys, U, V = eq_quiver(rhs, limits)
    
    ax.set_xlabel("x")
    ax.set_ylabel("x\'")
    ax.quiver(xs, ys, U, V, alpha=1, headwidth=2, scale=35)
    
    for point in initial_points:
        solution = solve_ivp(rhs, tlims, point, method='RK45', rtol=1e-12)
        x, y = solution.y
        ax.plot(x, y, 'b-')
 

# eq states: (-2, 0), (-1, 0), (0, 0) (1, 0), (2, 0)

# part for d > 0

DIS_VALUES = [1.5, 3.5, 6.0]

fig, axs = plt.subplots(3, 3)

for i in range(0, 3):
    DIS_VAL = DIS_VALUES[i]
    def rhs_dis(t, X):
        x, y = X
        return [y, -x**5 + 5*x**3 - 4*x - 2*DIS_VAL*y]

    # (-2, 0)
    plot(rhs_dis, axs[0, i], generate_point_seq((-2.0, 0.0), 0.5, 0.3, 12), limits=[(-3.0, -1.0), (-1.5, 1.5), (.0, 15.0)])

    # (0, 0)
    plot(rhs_dis, axs[2, i], generate_point_seq((0, 0), 0.75, 0.3, 12), limits=[(-0.75, 0.75), (-1.0, 1.0), (.0, 15.0)])

    # (2, 0)
    plot(rhs_dis, axs[1, i], generate_point_seq((2.0, 0.0), 0.5, 0.3, 12), limits=[(1.0, 3.0), (-1.5, 1.5), (.0, 15.0)])


plt.show()