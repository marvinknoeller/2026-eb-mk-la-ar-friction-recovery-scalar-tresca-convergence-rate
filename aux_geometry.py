#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:53:09 2025

@author: marvinknoller
"""

import gmsh
import meshio
import numpy as np
from mpi4py import MPI
import dolfinx


def create_flower_geometry(name, R = 1.0, a = 0.25, n_petals = 6, num_pts = 500, hole_radius = 0.25, mesh_size = 0.02, element_order = 2):
    """

    Parameters
    ----------
    name : string
    R : float, optional
        nominal outer radius. The default is 1.0.
    a : float, optional
        petal amplitude. The default is 0.25.
    n_petals : float (should be a natural number), optional
        number of petals. The default is 6.
    num_pts : int, optional
        sampling points on petal boundary. The default is 200.
    hole_radius : float, optional
        radius of central hole (disk to cut out). The default is 0.25.
    mesh_size : float, optional
        target mesh element size. The default is 0.02.
    element_order : int, optional
        quadratic curved elements (must be 2 currently). The default is 2.

    Returns
    -------
    mesh_data : a mesh data with mesh and tags of corresponding entities by
    codimension.

    """

    gmsh.initialize()
    gmsh.model.add("flower_with_hole")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # -------------------------
    # Create petal-shaped outer boundary via spline
    # -------------------------
    point_tags = []
    theta = np.linspace(0.0, 2.0 * np.pi, num_pts, endpoint=False)
    for th in theta:
        r = R + a * np.cos(n_petals * th)
        x = r * np.cos(th)
        y = r * np.sin(th)
        pt = gmsh.model.occ.addPoint(x, y, 0.0, mesh_size)
        point_tags.append(pt)
    
    # create a closed spline through the points
    spline_tag = gmsh.model.occ.addSpline(point_tags + [point_tags[0]])  # close loop by repeating first point
    curve_loop = gmsh.model.occ.addCurveLoop([spline_tag])
    
    # plane surface bounded by the petal curve
    petal_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
    
    # -------------------------
    # Create central circular disk (hole)
    # -------------------------
    # disk can be created with addDisk which returns a surface
    disk_surface = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, hole_radius, hole_radius)
    
    # synchronize CAD kernel
    gmsh.model.occ.synchronize()
    
    # -------------------------
    # Boolean cut: petal_surface minus disk_surface
    # -------------------------
    # perform cut operation
    out = gmsh.model.occ.cut([(2, petal_surface)], [(2, disk_surface)])
    # out is a list of resulting entities; extract the resultant surface tag(s)
    # result format: (out_entities, in_entities)
    # out[0][0] is the first entity: (dim, tag)
    if not out or not out[0]:
        raise RuntimeError("Boolean cut failed or returned no entities.")
    result_surfs = [ent for ent in out[0]]
    # For simplicity pick the first resulting surface as the domain
    domain_surface = result_surfs[0][1]
    
    gmsh.model.occ.synchronize()
    
    # Optionally set physical groups (useful for boundary marking)
    # Mark the interior surface as domain (dim=2)
    gmsh.model.addPhysicalGroup(2, [domain_surface], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    
    # Mark the hole boundary (line) so we can identify Dirichlet/Neumann boundaries if needed.
    # The cut created new curves for the hole boundary; gather curves bounding domain_surface:
    bnd_curves = gmsh.model.getBoundary([(2, domain_surface)], oriented=False, recursive=False)
    # bnd_curves is a list of (dim, tag) pairs; keep only lines (dim == 1)
    line_tags = [t for (d, t) in bnd_curves if d == 1]
    if line_tags:
        gmsh.model.addPhysicalGroup(1, line_tags, tag=2)
        gmsh.model.setPhysicalName(1, 2, "Boundary")
    
    # -------------------------
    # Mesh options: high-order geometry
    # -------------------------
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)
    # Generate 2D mesh
    gmsh.model.mesh.setOrder(element_order)
    gmsh.model.mesh.generate(2)
    # Write .msh (Gmsh format)
    msh_filename = name + ".msh"
    gmsh.write(msh_filename)
    
    # Also write a VTK for quick visualization if you want
    gmsh.write(name+".vtk")
    mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    # -------------------------
    # Convert with meshio: keep only the triangle6 block (curved triangles)
    # -------------------------
    m = meshio.read(msh_filename)
    
    # inspect what's available
    print("meshio cell keys:", getattr(m, "cells_dict", None) and list(m.cells_dict.keys()))
    
    return mesh_data
