#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:41:12 2025

@author: marvinknoller
"""

from dolfinx import fem
import numpy
from mpi4py import MPI
from ufl import (FacetNormal, SpatialCoordinate,
                 div, dx, grad, inner)
import ufl
import numpy as np
import aux_geometry
import NLaux_files
import meshio
import matplotlib.tri as tri

plot_solution = 1
eps = 1e-1
from scipy.integrate import quad

# Normalization constant C for standard mollifier
def mollifier_C():
    I, _ = quad(lambda t: np.exp(-1/(1-t**2)), -1, 1)
    return 1.0 / I

C = mollifier_C()

# Reference cumulative F(s), support [-1,1]
def F_reference_exact(s):
    if s <= -1: return 0.0
    if s >=  1: return 1.0
    val, _ = quad(lambda t: np.exp(-1/(1-t**2)), -1, s)
    return C * val

# Tabulation grid on reference interval
N = 400                       # number of intervals
s_nodes = np.linspace(-1, 1, N+1)
F_nodes = np.array([F_reference_exact(s) for s in s_nodes])


def F_reference_ufl(s):
    # endpoints
    expr = ufl.conditional(s <= -1, -1.0,
               ufl.conditional(s >=  1, 1.0, 0.0))

    # internal piecewise definition
    for i in range(N):
        sl, sr = s_nodes[i], s_nodes[i+1]
        Fl, Fr = F_nodes[i], F_nodes[i+1]

        slope = (Fr - Fl) / (sr - sl)
        piece = Fl + slope * (s - sl)

        expr = ufl.conditional(
            ufl.And(s >= sl, s < sr),
            2.0*piece-1.0,
            expr
        )

    return expr

def beta(u):
    s = u/eps

    return F_reference_ufl(s)


class domain_info_class():
    def __init__(self, R=1.0, a=0.25, n_petals=6, num_pts=500, hole_radius=0.25):
        self.R = R 
        self.a = a
        self.n_petals = n_petals
        self.num_pts = num_pts 
        self.hole_radius = hole_radius

R = 1.0
a = 0
n_petals = 0
num_pts = 500
hole_radius = 0.5

domain_info = domain_info_class(R, a, n_petals, num_pts, hole_radius)


nums = 30
errorL2 = np.zeros(nums)
errorH1 = np.zeros(nums)
h = np.zeros(nums)
approx_msh_size = np.logspace(-1, -3 ,nums)
for nn in range(0,nums):
    print(nn)
    mesh_data = aux_geometry.create_flower_geometry('flower', R = R, a = a, n_petals = n_petals, num_pts = num_pts, hole_radius = hole_radius,
                                        mesh_size=approx_msh_size[nn], element_order=1)
    
    domain = mesh_data.mesh
    cell_marker = mesh_data.cell_tags
    facet_marker = mesh_data.facet_tags
    
    ###################
    msh = meshio.read("flower.msh")

    # For quadratic triangles in Gmsh, you may have 'triangle6'
    if "triangle10" in msh.cells_dict:
        triangles = msh.cells_dict["triangle10"]
    elif "triangle6" in msh.cells_dict:
        triangles = msh.cells_dict["triangle6"]
    elif "triangle" in msh.cells_dict:
        triangles = msh.cells_dict["triangle"]
    else:
        raise RuntimeError("No triangle cells found in mesh")

    points = msh.points  # shape (num_points, 3)

    # Take only first three nodes per triangle for h_max
    tri_vertices = triangles[:, :3]  # shape (num_triangles, 3)

    # Get coordinates of vertices
    coords = points[tri_vertices]  # shape (num_triangles, 3 vertices, 3 coords)

    # Compute edge vectors
    v0 = coords[:, 1, :] - coords[:, 0, :]
    v1 = coords[:, 2, :] - coords[:, 1, :]
    v2 = coords[:, 0, :] - coords[:, 2, :]

    # Compute edge lengths
    l0 = np.linalg.norm(v0, axis=1)
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)

    # Max edge length per triangle
    h_max = np.maximum.reduce([l0, l1, l2])
    h_min = np.minimum.reduce([l0, l1, l2])
    # print("h_max for first 10 triangles:", h_max[:10])

    global_h_max = h_max.max()
    global_h_min = h_min.min()
    h[nn] = global_h_max
    ###################
    x = SpatialCoordinate(domain)
    u_exact = x[0]*x[1]*ufl.sin(5*ufl.sqrt(x[0]**2 + x[1]**2)*np.pi)

    alpha_ex =  2.0 + ufl.exp(ufl.sin(x[0]**2*x[1]))
    f = div(grad(u_exact))
    n = FacetNormal(domain)
    g = inner(n,grad(u_exact)) + alpha_ex*beta(u_exact)
    uh = NLaux_files.solve_nonlinear_pde(domain=domain,
                                    alpha = alpha_ex,  
                                    fhandle = f,
                                    ghandle = g,
                                    betahandle = beta,
                                    domain_info = domain_info,
                                    degree=1,
                                    LU = True
                                    )
    error_form = fem.form(inner(uh-u_exact, uh-u_exact) * dx)
    error_local = fem.assemble_scalar(error_form)
    errorL2[nn] = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    
    error_formH1 = (fem.form( inner(uh-u_exact, uh-u_exact) * dx + 
                    inner(grad(uh-u_exact), grad(uh-u_exact)) * dx))
    error_localH1 = fem.assemble_scalar(error_formH1)
    errorH1[nn] = numpy.sqrt(domain.comm.allreduce(error_localH1, op=MPI.SUM))

import numpy as np
import matplotlib.pyplot as plt

# Sample data

# loglog plot
plt.loglog(h, errorH1, color='black', linewidth=2, marker='o')
plt.loglog(h, h*.9e1, color='green', linewidth=2)
plt.loglog(h, errorL2, color='blue', linewidth=2, marker='d')
plt.loglog(h, h**2*.8e1, color='red', linewidth=2)
# Labels and grid
plt.xlabel("h")
plt.ylabel("error (log scale)")
plt.title("Semilogy Plot Example")
plt.grid(True, which="both", linestyle="--")

plt.show()

from tabulate import tabulate  # for nice table printing
# Compute EOC values
eocs = [None]  # First entry has no EOC
for i in range(1, len(h)):
    rate = np.log(errorL2[i]/errorL2[i-1]) / np.log(h[i]/h[i-1])
    eocs.append(rate)

# Create table
table = []
for i in range(len(h)):
    table.append([h[i], errorL2[i], eocs[i]])

# Print table
headers = ["h", "Error", "EOC"]
print(tabulate(table, headers=headers, floatfmt=".4e"))

from scipy.io import savemat    
data = {'h' : h,
        'errorL2' : errorL2,
        'errorH1' : errorH1
        }
savemat('order_finite_elementsNLnew.mat',data)


import matplotlib.ticker as ticker
if plot_solution == 1:
    fig, ax = plt.subplots(figsize=(5, 5))
    xd = domain.geometry.x
    cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
    triangulation = tri.Triangulation(xd[:, 0], xd[:, 1], cells)
    contour = plt.tricontourf(triangulation, uh.x.array,500, cmap="gnuplot2_r")
    plt.axis("equal")
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', which='major', length=8, width=1.5)
    ax.set_xticks([-1,0, 1])
    ax.set_yticks([-1,0, 1])
    # Customize minor ticks
    ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
    
    cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
    lims = contour.get_clim()
    cbar = fig.colorbar(contour, cax=cbar_ax)

    cbar.formatter = ticker.FormatStrFormatter('%.2f')
    cbar.update_ticks()
    plt.savefig('exact_solution_order.png', bbox_inches='tight')
    plt.show()
    

