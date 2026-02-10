#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 09:01:27 2025

@author: marvinknoller
"""

import numpy as np
import dolfinx
from basis_functions import create_fun, create_fun_for_plot
from dolfinx import fem, default_scalar_type
import ufl
from ufl import SpatialCoordinate
import NLaux_files
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import aux_geometry
import meshio
eps = 1e0

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
    """
    This defines the function beta. It can be used as a function handle.

    Parameters
    ----------
    u : dolfinx.fem function
        later we insert u_h^{(a)}, the numerical approximation to u^{(a)}

    Returns
    -------
    F_reference_ufl(s) : dolfinx.fem function
        this is simply -1 for u<=-eps, 1 for u>=eps and the smooth transition function in the transition zone (-eps,eps)

    """
    s = u/eps

    return F_reference_ufl(s)

def betaprime(u):
    """
    This defines the function beta'. It can be used as a function handle.
    
    Parameters
    ----------
    u : dolfinx.fem function
        later we insert u_h^{(a)}, the numerical approximation to u^{(a)}

    Returns
    -------
    beta_prime_u : dolfinx.fem function
        this is the derivative of \beta, which can be explicitly computed

    """
    betaprime_u = ufl.conditional(ufl.ge(u,eps), 0.0, 
                             ufl.conditional(ufl.le(u,-eps), 0.0, 
                                             2*ufl.exp(-1/(1-(u/eps)**2)) * C *1/eps
                                             )
                             )
    return betaprime_u

def beta2prime(u):
    """
    This defines the function beta''. It can be used as a function handle.
    
    Parameters
    ----------
    u : dolfinx.fem function
        later we insert u_h^{(a)}, the numerical approximation to u^{(a)}

    Returns
    -------
    beta2_prime_u : dolfinx.fem function
        this is \beta'', which can be explicitly computed

    """
    beta2prime_u = ufl.conditional(ufl.ge(u,eps), 0.0, 
                             ufl.conditional(ufl.le(u,-eps), 0.0, 
                                             -4*(u/eps)/((1-(u/eps)**2)**2)*ufl.exp(-1/(1-(u/eps)**2)) * C *1/eps**2
                                             )
                             )
    return beta2prime_u


""" 
-------------------------------------------------------------------------------------
We start with constructing the direct problem, which is to compute q = u_h|_{\omega}
-------------------------------------------------------------------------------------
"""

""" Want to see a plot of the solution?"""
plot_solution = 0
plot_iteration = 1

class domain_info_class():
    def __init__(self, R=1.0, a=0.25, n_petals=6, num_pts=500, hole_radius=0.25):
        self.R = R 
        self.a = a
        self.n_petals = n_petals
        self.num_pts = num_pts 
        self.hole_radius = hole_radius

R = 1.0
a = .25
n_petals = 6
num_pts = 500
hole_radius = 0.25

""" The maximal mesh size"""
num_h = 21
hh_approx = np.logspace(-2.0, -3.1, num_h)

hh_real = np.zeros(num_h)
err_vec_C0 = np.zeros(num_h)
err_vec_C1 = np.zeros(num_h)
err_vec_C2 = np.zeros(num_h)
# err_vec = np.zeros(num_h)

h_ref = 0.0006
mesh_data_reference = aux_geometry.create_flower_geometry('flower', R = R, a = a, n_petals = n_petals, num_pts = num_pts, hole_radius = hole_radius,
                                    mesh_size=h_ref, element_order=2)
domain_reference = mesh_data_reference.mesh

center = np.array([[.8, -.8, 0.8*np.cos(np.pi/3), 0.8*np.cos(2*np.pi/3), 0.8*np.cos(4*np.pi/3), 0.8*np.cos(5*np.pi/3)], 
                   [0, 0, 0.8*np.sin(np.pi/3), 0.8*np.sin(2*np.pi/3), 0.8*np.sin(4*np.pi/3), 0.8*np.sin(5*np.pi/3)]])
radius = np.array([.25,.25,.2,.2,.2,.2])

""" Determine the set \omega """

def inner_disc(x, tol=1e-13):
    ind_x = np.zeros(x[0].size)
    for cc in range(center.shape[1]):
        ind_x += (x[0]-center[0,cc])**2 + (x[1]-center[1,cc])**2 <= radius[cc]**2
    return ind_x


c_coeffs = np.array([8.0, 1., -.5, 0.5, 0.1, -.7])
s_coeffs = np.array([.8, .1, -1, 0.5, 0.2, .4])


x_ref = SpatialCoordinate(domain_reference)
cos_coeffs = fem.Constant(domain_reference, default_scalar_type(c_coeffs))
sin_coeffs = fem.Constant(domain_reference, default_scalar_type(s_coeffs))
if plot_iteration == 1:
    xx, vals = create_fun_for_plot(np.array(cos_coeffs.value), np.array(sin_coeffs.value))
alpha_exact = create_fun(cos_coeffs, sin_coeffs, x_ref)
V_reference = fem.functionspace(domain_reference, ("Lagrange", 2))

# fhandle = lambda x,y : -10 * (x) * ufl.exp(ufl.sin(4*np.pi*y))
fhandle = lambda x,y : -5 * (x*y) * ufl.exp(ufl.sin(4*np.pi*y))
ghandleref = lambda x,y : fem.Constant(domain_reference, default_scalar_type(0.0))
ghandle = lambda x,y : fem.Constant(domain, default_scalar_type(0.0))
# -------------- if ghandle is in fact a handle, we do not need to distinguish-#
# -----------------------------------------------------------------------------#
domain_info = domain_info_class(R, a, n_petals, num_pts, hole_radius)
u_exact = NLaux_files.solve_nonlinear_pde(domain=domain_reference,
                                alpha = alpha_exact,  
                                fhandle = fhandle,
                                ghandle = ghandleref,
                                betahandle = beta,
                                domain_info = domain_info,
                                degree=2,
                                LU = False)

for h_ell in range(num_h):
    print('h-Iteration no. '+str(h_ell))
    """ The (approximate) mesh size"""
    """ domain must be defined outside here, otherwise it destroys the uniqueness (of the domain)!"""
    mesh_data = aux_geometry.create_flower_geometry('flower', R = R, a = a, n_petals = n_petals, num_pts = num_pts, hole_radius = hole_radius,
                                        mesh_size=hh_approx[h_ell], element_order=1)
    
    domain = mesh_data.mesh
    
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
    hh_real[h_ell] = global_h_max
    
    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    marker = np.ones(num_cells, dtype=np.int32)
    marker[dolfinx.mesh.locate_entities(domain, tdim, inner_disc)] = 2
    
    cell_tag = dolfinx.mesh.meshtags(domain, tdim, np.arange(num_cells, dtype=np.int32), marker)
    subdx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag)
    inner_vol = dolfinx.fem.form(1 * subdx(2))
    local_volume = dolfinx.fem.assemble_scalar(inner_vol)
    # full_volume = dolfinx.fem.assemble_scalar( dolfinx.fem.form( 1* ufl.dx))
    fulldx = ufl.Measure("dx", domain=domain)
    
    full_volume = dolfinx.fem.assemble_scalar(fem.form(1.0 * fulldx))
    print("The full volume is " + str(full_volume))
    print("The volume of the subset is " + str(local_volume))
    
    print("Known: " +str(local_volume/full_volume))
    
    """ boundary value problem for z """
    from dolfinx.fem import locate_dofs_geometrical
    V = fem.functionspace(domain, ("Lagrange", 1))
    dofs = locate_dofs_geometrical(V, inner_disc)
    LU = True
    x = SpatialCoordinate(domain)
    
    exact_a = np.concat((c_coeffs,s_coeffs))
    der_exact_a = np.concat((np.arange(1,s_coeffs.size+1,1.) * s_coeffs,-np.arange(0,c_coeffs.size,1.) * c_coeffs))
    derder_exact_a = np.concat((-np.arange(0,c_coeffs.size,1.) * np.arange(0,c_coeffs.size,1.) * c_coeffs,
                                -np.arange(1,s_coeffs.size+1,1.) * np.arange(1,s_coeffs.size+1,1.) * s_coeffs
                                ))
    
    
    """ set q = uh_ex|_\omega"""
    u_interp = NLaux_files.project_between_spaces(V_reference, V, u_exact, domain_info)
    q = fem.Function(V)
    q.x.array[dofs] = u_interp.x.array[dofs]
    q.x.scatter_forward()
    
    if plot_solution == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        xd = domain.geometry.x
        cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
        triangulation = tri.Triangulation(xd[:, 0], xd[:, 1], cells)
        
        contour = plt.tricontourf(triangulation, u_interp.x.array,500, cmap="gnuplot2_r")
        plt.axis("equal")
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        ax.tick_params(axis='both', which='major', length=8, width=1.5)
        # Customize minor ticks
        ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
        
        cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
        lims = contour.get_clim()
        fig.colorbar(contour, cax=cbar_ax)
        plt.savefig('exact_solution.png', bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(5, 5))
        contour = plt.tricontourf(triangulation, q.x.array,500, cmap="gnuplot2_r")
        plt.axis("equal")
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        ax.tick_params(axis='both', which='major', length=8, width=1.5)
        # Customize minor ticks
        ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
        contour.set_clim(lims)
        cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
        fig.colorbar(contour, cax=cbar_ax)
        plt.savefig('q.png', bbox_inches='tight')
        plt.show()
        
    """ 
    -------------------------------------------------------------------------------------
    End of the direct problem.
    Now the inverse problem starts
    -------------------------------------------------------------------------------------
    """
    
    """ define an initial guess """
    num_cos = 6
    num_sin = 6
    
    if h_ell == 0:
        cos_coeffs_initial = np.array([np.pi*0.25 * 2])
        sin_coeffs_initial = np.array([0.0])
        
        cos_coeffs_ell = np.zeros(num_cos)
        sin_coeffs_ell = np.zeros(num_sin)
        cos_coeffs_ell[:cos_coeffs_initial.size] = cos_coeffs_initial
        sin_coeffs_ell[:sin_coeffs_initial.size] = sin_coeffs_initial
    
    cos_coeffs_ell_ufl = fem.Constant(domain, default_scalar_type(cos_coeffs_ell))
    sin_coeffs_ell_ufl = fem.Constant(domain, default_scalar_type(sin_coeffs_ell))
    
    x_ell = np.concatenate((cos_coeffs_ell, sin_coeffs_ell))
    
    coeff_history = x_ell[:,np.newaxis]
    
    xx, vals_ell = create_fun_for_plot(cos_coeffs_ell, sin_coeffs_ell)
    vals_history = vals_ell[:,np.newaxis]
    if plot_iteration == 1:
        fig, ax = plt.subplots()
        
        plt.plot(xx,vals, color='blue', linewidth=2)
        plt.plot(xx,vals_ell, color='red', linewidth=2)
        plt.show()
    
    x_ell0 = np.zeros(num_cos+num_sin)
    tol = 1e-6
    
    for ell in range(10000):
        if np.linalg.norm(x_ell - x_ell0)<tol:
            _, maxvals = create_fun_for_plot(c_coeffs - cos_coeffs_ell, 
                                             s_coeffs -sin_coeffs_ell,
                                             nn=1000000)
            _, dermaxvals = create_fun_for_plot(
                np.arange(1,s_coeffs.size+1,1.) * s_coeffs - np.arange(1,sin_coeffs_ell.size+1,1.) * sin_coeffs_ell, 
                -np.arange(0,c_coeffs.size,1.) * c_coeffs + np.arange(0,cos_coeffs_ell.size,1.) * cos_coeffs_ell,
                nn=1000000)
            _, derdermaxvals = create_fun_for_plot(
                -np.arange(0,c_coeffs.size,1.) * np.arange(0,c_coeffs.size,1.) * c_coeffs 
                + np.arange(0,cos_coeffs_ell.size,1.) * np.arange(0,cos_coeffs_ell.size,1.) * cos_coeffs_ell, 
                -np.arange(1,s_coeffs.size+1,1.) * np.arange(1,s_coeffs.size+1,1.) * s_coeffs 
                + np.arange(1,sin_coeffs_ell.size+1,1.) * np.arange(1,sin_coeffs_ell.size+1,1.) * sin_coeffs_ell,
                nn=1000000)
            err_vec_C0[h_ell] = np.max(np.abs(maxvals)) 
            err_vec_C1[h_ell] = np.max(np.abs(maxvals)) + np.max(np.abs(dermaxvals))
            err_vec_C2[h_ell] = np.max(np.abs(maxvals)) + np.max(np.abs(dermaxvals)) + np.max(np.abs(derdermaxvals))
            # err_vec[h_ell] = np.max(np.abs(maxvals)) + np.max(np.abs(dermaxvals)) + np.max(np.abs(derdermaxvals))
            
            ##########
            _, maxvals_ref = create_fun_for_plot(c_coeffs, 
                                             s_coeffs,
                                             nn=1000000)
            _, dermaxvals_ref = create_fun_for_plot(
                np.arange(1,s_coeffs.size+1,1.) * s_coeffs , 
                -np.arange(0,c_coeffs.size,1.) * c_coeffs ,
                nn=1000000)
            _, derdermaxvals_ref = create_fun_for_plot(
                -np.arange(0,c_coeffs.size,1.) * np.arange(0,c_coeffs.size,1.) * c_coeffs, 
                -np.arange(1,s_coeffs.size+1,1.) * np.arange(1,s_coeffs.size+1,1.) * s_coeffs,
                nn=1000000)
            # total_ref = np.max(np.abs(maxvals_ref)) + np.max(np.abs(dermaxvals_ref)) + np.max(np.abs(derdermaxvals_ref))
            total_ref_C0 = np.max(np.abs(maxvals_ref))
            total_ref_C1 = np.max(np.abs(maxvals_ref)) + np.max(np.abs(dermaxvals_ref))
            total_ref_C2 = np.max(np.abs(maxvals_ref)) + np.max(np.abs(dermaxvals_ref)) + np.max(np.abs(derdermaxvals_ref))
            ##########
            print('Stop')
            break
        else:
            print('Performing step No. ' + str(ell))
        alpha_ell = create_fun(cos_coeffs_ell_ufl, sin_coeffs_ell_ufl, x)
        F_ell, u_ha_ell, z_ha_ell = NLaux_files.evaluate_F(alpha_ell, num_cos, num_sin, domain, fhandle, ghandle, beta
                                                           , betaprime, domain_info, q, V, dofs, LU = LU)
        DF_ell = NLaux_files.evaluate_DF(alpha_ell, u_ha_ell, z_ha_ell, num_cos, num_sin, domain, beta, betaprime, beta2prime, domain_info, V, dofs, LU = LU)
        """ Newton Step"""
        update_ell = np.linalg.solve(DF_ell, -F_ell)
        # Line search
        f0 = np.linalg.norm(F_ell)
        alpha = .5
        kk = 0
        for kk in range(0,30):
            # print(kk)
            if kk == 29:
                break
            cos_coeffs_A = cos_coeffs_ell + alpha**kk*update_ell[:num_cos]
            sin_coeffs_A = sin_coeffs_ell + alpha**kk*update_ell[num_cos:num_cos+num_sin]
            cos_coeffs_ell_ufl_A = fem.Constant(domain, default_scalar_type(cos_coeffs_A))
            sin_coeffs_ell_ufl_A = fem.Constant(domain, default_scalar_type(sin_coeffs_A))
            alpha_A = create_fun(cos_coeffs_ell_ufl_A, sin_coeffs_ell_ufl_A, x)
            xx, vals_pos = create_fun_for_plot(cos_coeffs_A, sin_coeffs_A)
            if max(vals_pos<0) == True: #never compute when alpha<0 somewhere. This breaks the solver due to non-convergence.
                continue
            
            F_ell_A, _, _ = NLaux_files.evaluate_F(alpha_A, num_cos, num_sin, domain,fhandle, ghandle, beta
                                                               , betaprime, domain_info, q, V, dofs, LU = LU)
            if (np.linalg.norm(F_ell_A)< f0) and (min(vals_pos>0) == True):
                break
                
        cos_coeffs_ell += alpha**kk*update_ell[:num_cos]
        sin_coeffs_ell += alpha**kk*update_ell[num_cos:num_cos+num_sin]
        cos_coeffs_ell_ufl.value = cos_coeffs_ell
        sin_coeffs_ell_ufl.value = sin_coeffs_ell
        x_ell0 = x_ell
        x_ell = np.concatenate((cos_coeffs_ell, sin_coeffs_ell))
        coeff_history = np.concatenate((coeff_history, x_ell[:,np.newaxis]),axis=1)
        xx, vals_ell = create_fun_for_plot(cos_coeffs_ell, sin_coeffs_ell)
        vals_history = np.concatenate((vals_history, vals_ell[:,np.newaxis]),axis=1)
        if plot_iteration == 1:
            fig, ax = plt.subplots()
            
            plt.plot(xx,vals, color='blue', linewidth=2)
            plt.plot(xx,vals_ell, color='red', linewidth=2)
            plt.show()
            
            
            

import matplotlib.pyplot as plt

# Sample data

# loglog plot
hh = hh_real
fig, ax = plt.subplots()
plt.loglog(hh, err_vec_C2, color='blue', linewidth=2, marker='o', label='$\Vert a_h-\widetilde{a} \Vert_{C^1(\partial\Omega)}$')
plt.loglog(hh, hh*1e3, color='black',linestyle='--', linewidth=2)
plt.loglog(hh, hh**2*4e5, color='black',linestyle='--', linewidth=2)

plt.xlabel("h",fontsize=18)
plt.ylabel("error",fontsize=18)

plt.grid(True, which="both", linestyle="--",alpha=.7)
plt.rcParams['axes.labelsize'] = 14       # Axis label font size
plt.rcParams['xtick.labelsize'] = 14      # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 14      # Y-axis tick font size
# Customize major ticks
ax.tick_params(axis='both', which='major', length=8, width=1.5)
# Customize minor ticks
ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
plt.legend(fontsize=14, loc='lower right')
plt.savefig('order_a.eps', bbox_inches='tight')  # Save as EPS file
plt.show()

from tabulate import tabulate  # for nice table printing
# Compute EOC values
eocs = [None]  # First entry has no EOC
eoc_np = np.zeros(len(hh))
for i in range(1, len(hh)):
    rate = np.log(err_vec_C2[i]/err_vec_C2[i-1]) / np.log(hh[i]/hh[i-1])
    eoc_np[i-1] = np.log(err_vec_C2[i]/err_vec_C2[i-1]) / np.log(hh[i]/hh[i-1])
    eocs.append(rate)

# Create table
table = []
for i in range(len(hh)):
    table.append([hh[i], err_vec_C2[i], eocs[i]])

# Print table
headers = ["h", "Error", "EOC"]
print(tabulate(table, headers=headers, floatfmt=".4e"))

from scipy.io import savemat    
data = {'xx' : xx,
        'vals_history' : vals_history,
        'coeff_history' : coeff_history,
        'hh' : hh,
        'err_vec_C0' : err_vec_C0,
        'err_vec_C1' : err_vec_C1,
        'err_vec_C2' : err_vec_C2,
        'eoc' : eoc_np,
        'total_C0' : total_ref_C0,
        'total_C1' : total_ref_C1,
        'total_C2' : total_ref_C2
        }
savemat('4NLorder_a.mat',data)