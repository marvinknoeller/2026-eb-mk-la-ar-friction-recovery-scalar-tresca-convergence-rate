#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:35:22 2025

@author: marvinknoller
"""
import dolfinx
from dolfinx import default_scalar_type
from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import numpy
from mpi4py import MPI
from ufl import (SpatialCoordinate, TrialFunction, TestFunction,
                 dx, grad, inner)
import ufl
import numpy as np
from basis_functions import create_fun
from petsc4py import PETSc
import basix.ufl
import types

def project_between_spaces(V_fine, V_coarse, u_fine, domain_info):
    """
    Computes the projection of u_fine onto V_coarse. The result is u_coarse.

    Parameters
    ----------
    V_fine : fem.functionspace
        function space to project from 
    V_coarse : fem.functionspace
        function space to project onto
    u_fine : dolfinx.fem function
        function to project

    Returns
    -------
    u_coarse : dolfinx.fem function
        projected function on coarse grid.

    """
    R = domain_info.R
    petal_amp = domain_info.a
    n_petals = domain_info.n_petals
    
    degree = 4
    Qe = basix.ufl.quadrature_element(
        V_coarse.mesh.topology.cell_name(), degree=degree)
    V_quadrature = dolfinx.fem.functionspace(V_coarse.mesh, Qe)
    cells = np.arange(V_quadrature.mesh.topology.index_map(V_quadrature.mesh.topology.dim).size_local)
    nmmid = dolfinx.fem.create_interpolation_data(V_quadrature, V_fine, cells) 
    q_func = dolfinx.fem.Function(V_quadrature)
    q_func.interpolate_nonmatching(u_fine, cells, nmmid)

    # Project fine function at quadrature points to coarse grid
    u = ufl.TrialFunction(V_coarse)
    v = ufl.TestFunction(V_coarse)
    a_coarse = ufl.inner(u, v) * ufl.dx
    L_coarse = ufl.inner(q_func, v)*ufl.dx
    def flower_radius(x, y):
        theta = np.arctan2(y, x)
        return R + petal_amp * np.cos(n_petals * theta)
    
    def on_flower_boundary(x):
        # x has shape (3, N)
        xx = x[0]
        yy = x[1]
        r = np.sqrt(xx**2 + yy**2)

        r_exact = flower_radius(xx, yy)
        return np.isclose(r, r_exact, atol=1e-5)

    boundary_dofs = fem.locate_dofs_geometrical(V_coarse, on_flower_boundary)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V_coarse)
    
    problem = LinearProblem(
        a_coarse,
        L_coarse,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="membrane_",
    )
    u_coarse = problem.solve()
    return u_coarse

def solve_robin(domain, alpha, fhandle, ghandle, betaprime, domain_info, uhgiven, degree=1, LU = True):
    """
    Compute the approximation to the Robin problem on domain with right hand sides f and g.

    Parameters
    ----------
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    fhandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side f
    ghandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side g
    betaprime : types.FunctionType or ufl function
        the function (handle) that defines the derivative of beta
    domain_info : class
        class containing information about the petal geometry
    uhgiven : dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}
    degree : int, optional
        The degree of the finite element approximation. The default is 1.
    LU : bool, optional
        do you want to use the LU decomposition or an interative solver? The default is True.

    Returns
    -------
    uh: dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}

    """
    R = domain_info.R,
    petal_amp = domain_info.a,
    r0 = domain_info.hole_radius,
    n_petals = domain_info.n_petals
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = SpatialCoordinate(domain)
    if isinstance(fhandle, types.FunctionType) and fhandle.__name__ == "<lambda>": # if fhandle is a lambda function handle
        f = fhandle(x[0], x[1])
    else:
        f = fhandle
    
    if isinstance(ghandle, types.FunctionType) and ghandle.__name__ == "<lambda>": # if ghandle is a lambda function handle
        g = ghandle(x[0], x[1])
    else:
        g = ghandle
    
    u = TrialFunction(V)
    v = TestFunction(V)
    def on_inner_circle(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        return np.isclose(r, r0, atol=1e-5)

    

    tdim = domain.topology.dim
    fdim = tdim - 1

    inner_facets = locate_entities_boundary(domain, fdim, on_inner_circle)

    # Create facet tag array (default value 0 = unmarked)
    tags = np.zeros(domain.topology.index_map(fdim).size_local, dtype=np.int32)
    tags[inner_facets] = 1   # mark inner circle with id=1

    facet_tags = meshtags(domain, fdim, inner_facets, np.full(len(inner_facets), 1, dtype=np.int32))
    ds_inner = ufl.ds(subdomain_data=facet_tags, subdomain_id=1)

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(alpha*betaprime(uhgiven)*u,v) * ds_inner
    L = -ufl.dot(f, v) * ufl.dx + ufl.dot(g,v)*ds_inner
    
    
    def flower_radius(x, y):
        theta = np.arctan2(y, x)
        return R + petal_amp * np.cos(n_petals * theta)
    
    def on_flower_boundary(x):
        # x has shape (3, N)
        xx = x[0]
        yy = x[1]
        r = np.sqrt(xx**2 + yy**2)

        r_exact = flower_radius(xx, yy)
        return np.isclose(r, r_exact, atol=1e-5)

    boundary_dofs = fem.locate_dofs_geometrical(V, on_flower_boundary)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
    if LU == True:
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="linear_robin_",
        )
    else:
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",           # or "ilu", "jacobi", "gamg", etc.
                "ksp_rtol": 1e-8,             # optional: GMRES convergence tolerance
                "ksp_max_it": 2000            # optional: max iterations
            },
            petsc_options_prefix="linear_robin_",
        )
        
    uh = problem.solve()
    return uh

def solve_nonlinear_pde(domain, alpha, fhandle, ghandle, betahandle, domain_info, degree=1, LU = True):
    """
    Compute the approximation to the Robin problem on domain with right hand sides f and g.

    Parameters
    ----------
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    fhandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side f
    ghandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side g
    betahandle : types.FunctionType or ufl function
        the function (handle) that defines beta
    domain_info : class
        class containing information about the petal geometry
    degree : int, optional
        The degree of the finite element approximation. The default is 1.
    LU : bool, optional
        do you want to use the LU decomposition or an interative solver? The default is True.

    Returns
    -------
    uh: dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}

    """
    R = domain_info.R,
    petal_amp = domain_info.a,
    r0 = domain_info.hole_radius,
    n_petals = domain_info.n_petals
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = SpatialCoordinate(domain)
    if isinstance(fhandle, types.FunctionType) and fhandle.__name__ == "<lambda>": # if fhandle is a lambda function handle
        f = fhandle(x[0], x[1])
    else:
        f = fhandle
    
    if isinstance(ghandle, types.FunctionType) and ghandle.__name__ == "<lambda>": # if ghandle is a lambda function handle
        g = ghandle(x[0], x[1])
    else:
        g = ghandle
    
    uh = fem.Function(V)
    v = TestFunction(V)
    def on_inner_circle(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        return np.isclose(r, r0, atol=1e-5)
    tdim = domain.topology.dim
    fdim = tdim - 1

    inner_facets = locate_entities_boundary(domain, fdim, on_inner_circle)

    # Create facet tag array (default value 0 = unmarked)
    tags = np.zeros(domain.topology.index_map(fdim).size_local, dtype=np.int32)
    tags[inner_facets] = 1   # mark inner circle with id=1

    facet_tags = meshtags(domain, fdim, inner_facets, np.full(len(inner_facets), 1, dtype=np.int32))

    ds_inner = ufl.ds(subdomain_data=facet_tags, subdomain_id=1)
    
    F = (ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx + ufl.dot(alpha*betahandle(uh),v) * ds_inner
         + ufl.dot(f, v) * ufl.dx - ufl.dot(g,v)*ds_inner) #a - L
    
    def flower_radius(x, y):
        theta = np.arctan2(y, x)
        return R + petal_amp * np.cos(n_petals * theta)
    
    def on_flower_boundary(x):
        # x has shape (3, N)
        xx = x[0]
        yy = x[1]
        r = np.sqrt(xx**2 + yy**2)

        r_exact = flower_radius(xx, yy)
        return np.isclose(r, r_exact, atol=1e-5)

    boundary_dofs = fem.locate_dofs_geometrical(V, on_flower_boundary)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
    if LU == True:
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 200,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",  # or "superlu_dist" for distributed
            "snes_rtol": 1e-6,
            "snes_atol": 1e-6,
            }

    else:
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 200,   # <-- increase max iterations
            "snes_atol": 1e-6,
            "snes_rtol": 1e-6,
            "ksp_error_if_not_converged": True,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_cycle_type": "v",
            }
    
    problem = NonlinearProblem(
        F,
        uh,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="nonlinrobin",
        )

    problem.solve()
    converged = problem.solver.getConvergedReason()
    assert converged > 0, "Solver did not converge, got {converged}."
    
    return uh


def evaluate_F(alpha, num_cos, num_sin, domain, fhandle, ghandle, beta, betaprime, domain_info, q, V, dofs, LU=True):
    """
    evaluate the function F, whose jth component is F_{h,j}(a) = <\phi_j u_h^{(a)}, z_h^{(a)}>_{L^2(\partial \Omega)}

    Parameters
    ----------
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    num_cos : int
        number of cosine functions
    num_sin : int
        number of sine functions
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    fhandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side f
    ghandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side g
    beta : types.FunctionType or ufl function
        the function (handle) that defines beta
    betaprime : types.FunctionType or ufl function
        the function (handle) that defines the derivative of beta
    domain_info : class
        class containing information about the petal geometry
    q : dolfinx.fem function
        the given data on the known domain \omega, which is q = u^{(\tilde{a})}|_\omega
    V : fem.functionspace
        the finite element space
    dofs : np.array
        the degrees of freedom that belong to \omega within \Omega
    LU : bool
        do you want to use the LU decomposition or an interative solver?

    Returns
    -------
    F : np.array(J)
        the vector containing the entries F_{h,j}
    u_ha : dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}
    z_ha : dolfinx.fem function
        this is z_h^{(a)}, the numerical approximation to z^{(a)}

    """
    
    x = SpatialCoordinate(domain) # the spatial coordinates of the domain
    J = num_cos + num_sin # total number of basis functions
    F = np.zeros(J) # we initialize the vector F
    ''' compute u_h^{(a)} '''
    u_ha = solve_nonlinear_pde(domain = domain, 
                       alpha=alpha,
                       fhandle = fhandle,
                       ghandle = ghandle,
                       betahandle = beta,
                       domain_info = domain_info,
                       degree = 1,
                       LU = LU
                       )
    
    uh_res = fem.Function(V)
    uh_res.x.array[dofs] = u_ha.x.array[dofs]
    uh_res.x.scatter_forward()
    z_ha = solve_robin(domain = domain,
                       alpha=alpha,
                       fhandle = uh_res-q,
                       ghandle = fem.Constant(domain, default_scalar_type(0.0)),
                       betaprime = betaprime, 
                       domain_info = domain_info,
                       uhgiven = u_ha,
                       degree = 1,
                       LU = LU
                       )
    cos_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    phi_j = create_fun(cos_coeffs, sin_coeffs, x)
    
    r0 = domain_info.hole_radius
    def on_inner_circle(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        return np.isclose(r, r0, atol=1e-5)

    tdim = domain.topology.dim
    fdim = tdim - 1

    inner_facets = locate_entities_boundary(domain, fdim, on_inner_circle)

    # Create facet tag array (default value 0 = unmarked)
    tags = np.zeros(domain.topology.index_map(fdim).size_local, dtype=np.int32)
    tags[inner_facets] = 1   # mark inner circle with id=1

    facet_tags = meshtags(domain, fdim, inner_facets, np.full(len(inner_facets), 1, dtype=np.int32))

    ds_inner = ufl.ds(subdomain_data=facet_tags, subdomain_id=1)
    ####
    
    Fform = fem.form(inner(phi_j * beta(u_ha), z_ha)*ds_inner)
    for nn in range(num_cos):
        cos_coeffs.value = 0.0
        sin_coeffs.value = 0.0
        cos_coeffs.value[nn] = 1.0
        Fform_ass = fem.assemble_scalar(Fform)
        F[nn] = domain.comm.allreduce(Fform_ass, op=MPI.SUM)
    
    for nn in range(num_sin):
        cos_coeffs.value = 0.0
        sin_coeffs.value = 0.0
        sin_coeffs.value[nn] = 1.0
        Fform_ass = fem.assemble_scalar(Fform)
        F[nn+num_cos] = domain.comm.allreduce(Fform_ass, op=MPI.SUM)
        
    return F, u_ha, z_ha


def evaluate_DF(alpha, u_ha, z_ha, num_cos, num_sin, domain, beta, betaprime, beta2prime, domain_info, V, dofs, LU=True):
    """
    evaluate the derivative DF, whose jth component is 
    F'_{h,j}(a) = <\phi_j \dot{u}_h^{(a)}, z_h^{(a)}>_{L^2(\partial \Omega)} + <\phi_j u_h^{(a)}, \dot{z}_h^{(a)}>_{L^2(\partial \Omega)}

    Parameters
    ----------
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    u_ha : dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}
    z_ha : dolfinx.fem function
        this is z_h^{(a)}, the numerical approximation to z^{(a)}
    num_cos : int
        number of cosine functions
    num_sin : int
        number of sine functions
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    beta : types.FunctionType or ufl function
        the function (handle) that defines beta
    betaprime : types.FunctionType or ufl function
        the function (handle) that defines the derivative of beta
    beta2prime : types.FunctionType or ufl function
        the function (handle) that defines the 2nd derivative of beta
    domain_info : class
        class containing information about the petal geometry
    V : fem.functionspace
        the finite element space
    dofs : np.array
        the degrees of freedom that belong to \omega within \Omega
    LU : bool
        do you want to use the LU decomposition or an interative solver?

    Returns
    -------
    DF : np.array((J,J))
        the Jacobian matrix corresponding to F, from evaluate_F

    """
    ###
    R = domain_info.R,
    petal_amp = domain_info.a,
    r0 = domain_info.hole_radius,
    n_petals = domain_info.n_petals
    def on_inner_circle(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        return np.isclose(r, r0, atol=1e-5)

    tdim = domain.topology.dim
    fdim = tdim - 1

    inner_facets = locate_entities_boundary(domain, fdim, on_inner_circle)

    # Create facet tag array (default value 0 = unmarked)
    tags = np.zeros(domain.topology.index_map(fdim).size_local, dtype=np.int32)
    tags[inner_facets] = 1   # mark inner circle with id=1

    facet_tags = meshtags(domain, fdim, inner_facets, np.full(len(inner_facets), 1, dtype=np.int32))

    ds_inner = ufl.ds(subdomain_data=facet_tags, subdomain_id=1)

    
    x = SpatialCoordinate(domain) # the spatial coordinates of the domain
    J = num_cos + num_sin # total number of basis functions
    DF = np.zeros((J,J)) # we initialize the Jacobi matrix
    '''  
    the rhs on the boundary of \dot{u} and \dot{z} is -\eta u_h and -\eta z_h,
    respectively. 
    '''
    cos_coeffs_eta = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs_eta = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    eta = create_fun(cos_coeffs_eta, sin_coeffs_eta, x)
    
    ''' phi_j is a single basis function '''
    cos_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    phi_j = create_fun(cos_coeffs, sin_coeffs, x)
    
    ''' 
    initialize the place holder functions for \dot{u}_h and \dot{z}_h as well as the form
    that will determine later the entries of the matrix.
    It is exactly the derivative of F_{h,j}
    '''
    u_ha_prime_PH = fem.Function(V)
    z_ha_prime_PH = fem.Function(V)
    DFform = fem.form(inner(phi_j * betaprime(u_ha)*u_ha_prime_PH, z_ha)*ds_inner + inner(phi_j * beta(u_ha), z_ha_prime_PH)*ds_inner)
    
    '''
    this is all for solving the pde. For both problems \dot{u}_h and \dot{z}_h there is the same bilinear form a,
    however, we set up two different right hand sides, due to different f and g.
    '''
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + inner(alpha*betaprime(u_ha)*u, v) * ds_inner
    f1 = fem.Function(V) # f1 remains zero
    f2 = fem.Function(V)
    g1 = -eta*beta(u_ha)
    Luh1 = -inner(f1, v) * dx + inner(g1,v)*ds_inner
    a_compiled = dolfinx.fem.form(a)
    Luh1_compiled = dolfinx.fem.form(Luh1)
    
    def flower_radius(x, y):
        theta = np.arctan2(y, x)
        return R + petal_amp * np.cos(n_petals * theta)
    
    def on_flower_boundary(x):
        # x has shape (3, N)
        xx = x[0]
        yy = x[1]
        r = np.sqrt(xx**2 + yy**2)

        r_exact = flower_radius(xx, yy)
        return np.isclose(r, r_exact, atol=1e-5)

    boundary_dofs = fem.locate_dofs_geometrical(V, on_flower_boundary)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
    
    A = fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    # Create solution functions
    u_ha_prime = A.createVecRight()
    z_ha_prime = A.createVecRight()
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ''' 
    we need to write it like this, otherwise the preconditioner or the lu decomposition is not saved (I believe.)
    '''
    if LU == True:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
    else:
        ksp.setType('gmres')
        ksp.setTolerances(rtol=1e-14)
        ksp.getPC().setType('hypre')
        
    g2 = -eta*betaprime(u_ha)*z_ha - alpha*beta2prime(u_ha)*u_ha_prime_PH*z_ha
    Luh2 = -inner(f2, v) * dx + inner(g2,v)*ds_inner
    Luh2_compiled = dolfinx.fem.form(Luh2)
    
    for nn in range(num_cos):
        """ 
        Cosine perturbation. This syntax works since everything is a ufl function. Meaning: The eta in the g and accordingly, in the form
        gets updated automatically 
        """
        cos_coeffs_eta.value = 0.0
        cos_coeffs_eta.value[nn] = 1.0
        sin_coeffs_eta.value = 0.0
        
        b = fem.petsc.assemble_vector(Luh1_compiled)
        ksp.solve(b, u_ha_prime)
        # update the place holder
        u_ha_prime_PH.x.array[:] = u_ha_prime.array[:]
        u_ha_prime_PH.x.scatter_forward()
        
        uhprime_res = fem.Function(V)
        uhprime_res.x.array[dofs] = u_ha_prime.array[dofs]
        uhprime_res.x.scatter_forward()
        f2.x.array[:] = uhprime_res.x.array[:]
        b = fem.petsc.assemble_vector(Luh2_compiled)
        ksp.solve(b, z_ha_prime)
        # update the place holder
        z_ha_prime_PH.x.array[:] = z_ha_prime.array[:]
        z_ha_prime_PH.x.scatter_forward()
        
        for mm1 in range(num_cos):
            """ Here we fill the matrix: The cosine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            cos_coeffs.value[mm1] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm1,nn] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
        for mm2 in range(num_sin):
            """ Here we fill the matrix: The sine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            sin_coeffs.value[mm2] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm2+num_cos, nn] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
    for nn in range(num_sin):
        """ 
        Sine perturbation. This syntax works since everything is a ufl function. Meaning: The eta in the g and accordingly, in the form
        gets updated automatically 
        """
        cos_coeffs_eta.value = 0.0
        sin_coeffs_eta.value = 0.0
        sin_coeffs_eta.value[nn] = 1.0
        
        b = fem.petsc.assemble_vector(Luh1_compiled)
        ksp.solve(b, u_ha_prime)
        # update the place holder
        u_ha_prime_PH.x.array[:] = u_ha_prime.array[:]
        u_ha_prime_PH.x.scatter_forward()
        
        uhprime_res = fem.Function(V)
        uhprime_res.x.array[dofs] = u_ha_prime.array[dofs]
        uhprime_res.x.scatter_forward()
        f2.x.array[:] = uhprime_res.x.array[:]
        b = fem.petsc.assemble_vector(Luh2_compiled)
        ksp.solve(b, z_ha_prime)
        # update the place holder
        z_ha_prime_PH.x.array[:] = z_ha_prime.array[:]  # or .setArray or .interpolate as appropriate
        z_ha_prime_PH.x.scatter_forward()  # sync ghost values for parallel
        for mm1 in range(num_cos):
            """ Here we fill the matrix: The cosine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            cos_coeffs.value[mm1] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm1,nn+num_cos] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
        for mm2 in range(num_sin):
            """ Here we fill the matrix: The sine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            sin_coeffs.value[mm2] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm2+num_cos, nn+num_cos] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
    return DF