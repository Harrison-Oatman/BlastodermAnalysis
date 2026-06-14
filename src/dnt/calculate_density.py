import numpy as np
import pymeshlab
import pandas as pd
from collections import defaultdict
from .timepoints import find_stationary_timepoints
from blender_tissue_cartography import interface_pymeshlab as intmsl
from blender_tissue_cartography import mesh as tcmesh
from blender_tissue_cartography.mesh import ObjMesh


def mesh_area_density(mesh: ObjMesh, points: np.ndarray) -> np.ndarray:
    """
    Calculate area density of mesh vertices, given a mesh and 3d locations of the vertices.

    :param mesh: mesh containing n vertices
    :param points: n x ndim array of vertex locations
    :return: areas_normed: n array of area densities, normalized to sum to n
    """

    faces = np.array(mesh.faces)
    areas = np.zeros(len(points))

    for shift in range(3):
        order = np.roll(faces, shift, axis=-1)

        ptx = points[order, :]

        ab = ptx[:, 1] - ptx[:, 0]
        ac = ptx[:, 2] - ptx[:, 0]

        area = np.linalg.norm(np.cross(ab, ac), axis=-1) / 2

        areas[order[:, 0]] += area / 3

    areas_normed = len(areas) * areas / (np.sum(areas))

    return areas_normed


def mesh_from_points(points) -> ObjMesh:
    point_cloud = tcmesh.ObjMesh(vertices=points, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)

    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.apply_coord_hc_laplacian_smoothing()
    ms.meshing_close_holes()

    mesh_reconstructed = intmsl.convert_from_pymeshlab(ms.current_mesh())

    return mesh_reconstructed


def smoothed_mesh_from_points(points, targetlen=1) -> ObjMesh:
    point_cloud = tcmesh.ObjMesh(vertices=points, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)

    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_screened_poisson(depth=8, fulldepth=5, )

    seed_value = 42
    import random
    random.seed(seed_value)

    # Set the random seed for the NumPy library
    np.random.seed(seed_value)

    ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=pymeshlab.PercentageValue(targetlen))

    return intmsl.convert_from_pymeshlab(ms.current_mesh())

def triangulated_mesh_from_points(points) -> ObjMesh:
    # generates mesh where each vertex is a point
    point_cloud = tcmesh.ObjMesh(vertices=points, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)
    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_ball_pivoting()

    return intmsl.convert_from_pymeshlab(ms.current_mesh())

import numpy as np
import igl

def vertex_voronoi_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the mixed Voronoi area for each vertex of a triangle mesh.

    Uses the Voronoi-hybrid (Meyer et al. 2003) mass matrix from libigl:
    - For non-obtuse triangles: uses the true Voronoi cell area (circumcenter-based).
    - For obtuse triangles: falls back to barycentric (1/3 of triangle area)
      to avoid negative contributions from circumcenters outside the triangle.

    Parameters
    ----------
    V : np.ndarray, shape (N, 3)
        Vertex positions. Each row is a 3D point (a nucleus position).
    F : np.ndarray, shape (M, 3), dtype int
        Triangle faces. Each row contains indices into V forming a triangle.

    Returns
    -------
    areas : np.ndarray, shape (N,)
        Per-vertex area. The area associated with vertex i is areas[i].
        The sum of all areas equals the total surface area of the mesh.

    Notes
    -----
    - The mesh must be a valid manifold triangle mesh with sphere topology.
    - Vertices not referenced by any face will have area 0.
    - As nuclei divide and the mesh is re-triangulated, simply call this
      function again on the updated (V, F) pair.

    Example
    -------
    >>> import numpy as np
    >>> import igl
    >>> # Icosphere as a simple sphere-topology test mesh
    >>> V, F = igl.icosahedron()
    >>> areas = vertex_voronoi_areas(V, F)
    >>> print(areas.shape)          # (12,)
    >>> print(areas.sum())          # ≈ surface area of the unit icosahedron
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    # Mass matrix M is diagonal: M[i, i] = mixed Voronoi area of vertex i.
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)

    # Extract the diagonal (the per-vertex areas).
    areas = np.zeros(V.shape[0])
    referenced = np.unique(F)
    areas[referenced] = M.diagonal()[referenced]
    return areas


def calculate_surface_area_along_axis(mesh: ObjMesh, dividers, axis=1):
    """

    Parameters
    ----------
    mesh
        mesh containing vertices and faces
    dividers
        list of positions along the axis to divide the mesh
    axis
        axis along which to divide the mesh (in point coordinates)

    Returns
    -------
        binned surface areas between dividers

    """

    surface_areas = [0., ]

    for div in dividers:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "embryo")
        ms.generate_polyline_from_planar_section(planeaxis=axis, planeoffset=div, splitsurfacewithsection=True)
        surface_areas.append(ms.get_geometric_measures()["surface_area"])

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "embryo")
    surface_areas.append(ms.get_geometric_measures()["surface_area"])

    return np.diff(surface_areas)


def calculate_all_surface_areas(spots_dfs, stems, all_mmfs, ap_vals):
    all_surface_areas = defaultdict(dict)

    for k in range(len(spots_dfs)):
        stem = stems[k]
        df = spots_dfs[k]

        min_mvmt_frames = all_mmfs[stem]

        points = df[df["frame"] == min_mvmt_frames[-1]][["x", "y", "z"]].values
        aps = df[df["frame"] == min_mvmt_frames[-1]]["AP"].values
        mesh = mesh_from_points(points)

        y_max = points[:, 1].max()
        y_min = points[:, 1].min()
        y_vals = y_min + ap_vals * (y_max - y_min)

        try:
            surface_areas = calculate_surface_area_along_axis(mesh, y_vals)
        except Exception as e:
            mesh = smoothed_mesh_from_points(points)
            surface_areas = calculate_surface_area_along_axis(mesh, y_vals)

        # flip order of surface areas if y_max corresponds to AP=0
        if aps[np.argmax(points[:, 1])] < aps[np.argmin(points[:, 1])]:
            surface_areas = surface_areas[::-1]

        ap_bounds = [(l, r) for l, r in zip([0.0, *ap_vals], [*ap_vals, 1.0])]
        centers = [(l + r) / 2 for l, r in ap_bounds]

        all_surface_areas[stem]["centers"] = centers
        all_surface_areas[stem]["areas"] = surface_areas
        all_surface_areas[stem]["bounds"] = ap_bounds

        all_surface_areas[stem] = pd.DataFrame(all_surface_areas[stem])

        spots_dfs[k]["AP_bin"] = pd.cut(
            spots_dfs[k]["AP"], bins=[0.0, *ap_vals, 1.0], labels=centers
        )

    for stem, surface_area_df in all_surface_areas.items():
        surface_area_df["area_normed"] = surface_area_df["areas"] / surface_area_df["areas"].sum()
        surface_area_df["area_normed"] = surface_area_df["area_normed"].astype(float)
        surface_area_df["center_copy"] = surface_area_df["centers"]
        surface_area_df.set_index("center_copy", inplace=True)

    return all_surface_areas

def calculate_relative_densities(spots_dfs, stems, all_mmfs, all_surface_areas, condition_map, cycles):
    cycle_relative_densities = defaultdict(list)

    for k, spots_df in enumerate(spots_dfs):
        stem = stems[k]

        """
        Density related metrics
        """
        spots_df["nucleus_weight"] = 1 / spots_df["frame"].map(spots_df.groupby("frame").size())
        spots_df["local_area"] = np.array(spots_df["AP_bin"].map(all_surface_areas[stems[k]]["area_normed"]))
        spots_df["AP_bin"] = np.array(spots_df["AP_bin"].astype(float))
        spots_df["nucleus_rel_density"] = spots_df["nucleus_weight"] / spots_df["local_area"]

        """
        Displacement related metrics
        """
        spots_df["time_since_nc10"] = np.abs(spots_df["frame"] - all_mmfs[stem][0])
        idx = spots_df.groupby("track_id")["time_since_nc10"].idxmin()
        result = spots_df.loc[idx].set_index("track_id")["AP"]
        spots_df["track_AP_init"] = spots_df["track_id"].map(result)
        spots_df["displacement_from_start"] = spots_df["AP"] - spots_df["track_AP_init"]
        last_frame = all_mmfs[stem][-1]
        spots_df["final_displacement"] = spots_df["track_id"].map(
            spots_df[spots_df["frame"] == last_frame].groupby("track_id")["AP"].mean()) - spots_df["AP"]

        """
        Get density and displacement at at min movement frames for each cycle
        """
        min_mvmt_frames = all_mmfs[stem]
        for cycle, frame in zip(cycles, min_mvmt_frames):
            cycle_df = spots_df[spots_df["frame"] == frame].copy()

            positions = cycle_df.groupby("AP_bin")["nucleus_rel_density"].mean().index.values
            densities = cycle_df.groupby("AP_bin")["nucleus_rel_density"].sum().values
            displacements = cycle_df.groupby("AP_bin")["displacement_from_start"].mean().values
            final_displacements = cycle_df.groupby("AP_bin")["final_displacement"].mean().values
            surface_areas = cycle_df.groupby("AP_bin")["local_area"].mean().values

            cycle_relative_densities["positions"].extend(positions)
            cycle_relative_densities["densities"].extend(densities)
            cycle_relative_densities["cycle"].extend([cycle for _ in range(len(positions))])
            cycle_relative_densities["condition"].extend([condition_map[stems[k][:-6]] for _ in range(len(positions))])
            cycle_relative_densities["source"].extend([k for _ in range(len(positions))])
            cycle_relative_densities["avg_displacement"].extend(displacements)
            cycle_relative_densities["avg_final_displacement"].extend(final_displacements)
            cycle_relative_densities["surface_area"].extend(surface_areas)

    cycle_relative_densities = pd.DataFrame(cycle_relative_densities)
    return cycle_relative_densities
