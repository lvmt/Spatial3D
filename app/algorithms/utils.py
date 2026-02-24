import numpy as np  
import pyvista as pv  
from pandas import DataFrame   
import matplotlib.pyplot as plt  
import matplotlib as mpl  


import open3d as o3d
from open3d import geometry 
import mcubes 
from scipy.spatial.distance import cdist
import pyacvd
import pymeshfix as mf




def _pv2o3d(pc):
    """Convert a point cloud in PyVista to Open3D."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc.points)
    if "norms" in pc.point_data.keys():
        cloud.normals = o3d.utility.Vector3dVector(pc["norms"])
    else:
        cloud.estimate_normals()
    return cloud


def _o3d2pv(trimesh):
    """Convert a triangle mesh in Open3D to PyVista."""
    v = np.asarray(trimesh.vertices)
    f = np.array(trimesh.triangles)
    f = np.c_[np.full(len(f), 3), f]

    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate().clean()
    return mesh


def scale_model(model, distance = None, scale_factor = 1, scale_center = None, inplace = False) :
 
    model_s = model.copy() if not inplace else model

    if not (distance is None):
        model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

    if not (scale_factor is None):
        model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor, scale_center=scale_center)

    model_s = model_s.triangulate()

    return model_s if not inplace else None


def _scale_model_by_distance(model, distance = 1, scale_center = None) :
    # Check the distance.
    distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
    if len(distance) != 3:
        raise ValueError(
            "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the distance.
    for i, (d, c) in enumerate(zip(distance, scale_center)):
        p2c_bool = np.asarray(model.points[:, i] - c) > 0
        model.points[:, i][p2c_bool] += d
        model.points[:, i][~p2c_bool] -= d

    return model


def _scale_model_by_scale_factor(
    model,
    scale_factor = 1,
    scale_center = None,
) :
    # Check the scaling factor.
    scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
    if len(scale_factor) != 3:
        raise ValueError(
            "`scale_factor` value is wrong."
            "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the scale center.
    for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
        model.points[:, i] = (model.points[:, i] - c) * f + c

    return model


def rigid_transform(
    coords,
    coords_refA,
    coords_refB,
) -> np.ndarray:
    # Check the spatial coordinates

    coords, coords_refA, coords_refB = (
        coords.copy(),
        coords_refA.copy(),
        coords_refB.copy(),
    )
    assert (
        coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
    ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
    coords_dim = coords.shape[1]
    if coords_dim == 2:
        coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
        coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
        coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

    # Compute optimal transformation based on the two sets of points.
    coords_refA = coords_refA.T
    coords_refB = coords_refB.T

    centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
    centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

    Am = coords_refA - centroid_A
    Bm = coords_refB - centroid_B
    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # Apply the transformation to other points
    new_coords = (R @ coords.T) + t
    new_coords = np.asarray(new_coords.T)
    return new_coords[:, :2] if coords_dim == 2 else new_coords


def merge_models(models):
    """Merge all models in the `models` list. The format of all models must be the same."""

    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)

    return merged_model


def clean_mesh(mesh):
    """Removes unused points and degenerate cells."""

    sub_meshes = mesh.split_bodies()
    n_mesh = len(sub_meshes)

    if n_mesh == 1:
        return mesh
    else:
        inside_number = []
        for i, main_mesh in enumerate(sub_meshes[:-1]):
            main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
            for j, check_mesh in enumerate(sub_meshes[i + 1 :]):
                check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                inside = pv.PolyData(inside.points, inside.cells)
                if check_mesh == inside:
                    inside_number.append(i + 1 + j)

        cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
        if len(cm_number) == 1:
            cmesh = sub_meshes[cm_number[0]]
        else:
            cmesh = merge_models([sub_meshes[i] for i in cm_number])

        return pv.PolyData(cmesh.points, cmesh.cells)


def fix_mesh(mesh):
    """Repair the mesh where it was extracted and subtle holes along complex parts of the mesh."""
    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=False)
    fixed_mesh = meshfix.mesh.triangulate().clean()

    if fixed_mesh.n_points == 0:
        raise ValueError(
            f"The surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
        )

    return fixed_mesh


def uniform_mesh(mesh, nsub = 3, nclus = 20000):
    # Check pyacvd package
    # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
    if not (nsub is None):
        mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

    # Uniformly remeshing.
    clustered = pyacvd.Clustering(mesh)
    clustered.cluster(nclus)

    new_mesh = clustered.create_mesh().triangulate().clean()
    return new_mesh


def smooth_mesh(mesh, n_iter = 100, **kwargs):
    """
    Adjust point coordinates using Laplacian smoothing.
    https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

    Args:
        mesh: A mesh model.
        n_iter: Number of iterations for Laplacian smoothing.
        **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

    Returns:
        smoothed_mesh: A smoothed mesh model.
    """

    smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

    return smoothed_mesh



# 定义颜色 
def get_continuous_cmap_colors(color_len:int):
    cmap = mpl.colormaps['rainbow'] 
    colors_rgb = cmap(np.linspace(0, 1, color_len))   # shape: (256, 4)  RGBA, 0~1
    colors_rgb = [tuple(int(255 * c) for c in color[:3]) for color in colors_rgb]  
    return colors_rgb  


def get_discrete_cmap_colors(color_len:int):  
    if color_len > 10:
        raise ValueError("tab10 颜色映射表最多支持 10 种颜色")  
    colors = plt.get_cmap('tab10').colors 
    rgb_colors = [tuple(int(255 * c) for c in color[:3]) for color in colors]
    return rgb_colors 