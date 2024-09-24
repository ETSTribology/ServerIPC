import numpy as np
import itertools
import meshio
import ipctk
import logging

def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = [0] + list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C

def to_surface(x: np.ndarray, mesh, cmesh: ipctk.CollisionMesh):
    X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order="F").T
    XB = cmesh.map_displacements(X)
    return XB

def find_codim_vertices(mesh, boundary_edges):
    # Initialize a set with all vertex indices
    all_vertices = set(range(len(mesh.X)))

    # Find vertices connected to boundary edges
    surface_vertices = set()
    for edge in boundary_edges:
        surface_vertices.update(edge)

    # Codimensional vertices are those not connected to any boundary edge
    codim_vertices = list(all_vertices - surface_vertices)
    if len(codim_vertices) == 0:
        logging.warning("No codimensional vertices found.")
    return codim_vertices
