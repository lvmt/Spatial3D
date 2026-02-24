import numpy as np  
import pyvista as pv  
from pandas import DataFrame   
import open3d as o3d
from open3d import geometry 
import mcubes 
from scipy.spatial.distance import cdist
import pyacvd
import pymeshfix as mf


from utils import (
    rigid_transform, 
    clean_mesh, 
    fix_mesh, 
    uniform_mesh, 
    merge_models, 
    smooth_mesh, 
    _pv2o3d, 
    _o3d2pv,
    scale_model,
    get_continuous_cmap_colors,  
    get_discrete_cmap_colors 
)  


def _validate_point_cloud(pc, min_points=10):
    if pc is None:
        raise ValueError("Point cloud is None.")
    if not hasattr(pc, "points"):
        raise ValueError("Invalid point cloud object.")
    if pc.n_points < min_points:
        raise ValueError(f"Point cloud has too few points ({pc.n_points}). Need at least {min_points} points.")


def _adaptive_clean_tolerance(points: np.ndarray, ratio: float = 1e-4, min_tol: float = 1e-8) -> float:
    diag = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    if diag <= 0:
        return min_tol
    return max(diag * ratio, min_tol)


def _estimate_nn_distance(points: np.ndarray, sample_size: int = 1500, random_seed: int = 42) -> float:
    if points.shape[0] < 2:
        raise ValueError("Need at least 2 points to estimate nearest-neighbor distance.")

    rng = np.random.default_rng(random_seed)
    n_points = points.shape[0]
    if n_points > sample_size:
        idx = rng.choice(n_points, size=sample_size, replace=False)
        sample = points[idx]
    else:
        sample = points

    dist = cdist(sample, points, metric="euclidean")
    for i in range(sample.shape[0]):
        dist[i, np.argmin(dist[i])] = np.nan

    nn = np.nanmin(dist, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        raise ValueError("Failed to estimate nearest-neighbor distance from point cloud.")
    return float(np.median(nn))


def construct_pc(df, anno=None):
    # Placeholder for constructing a path condition
    bucket_xyz = df[['x', 'y', 'z']].to_numpy().astype(np.float64) 
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
        
    pc = pv.PolyData(bucket_xyz)
    if anno is not None:
        if not anno in df.columns:
            raise ValueError(f"Annotation column '{anno}' not found in DataFrame.")  
        
        labels = df[anno].to_numpy()
        unique_labels = np.unique(labels) 

        if len(unique_labels) <= 10:
            colors = get_discrete_cmap_colors(len(unique_labels)) 
        else:
            colors = get_continuous_cmap_colors(len(unique_labels)) 

        color_map = dict(zip(unique_labels, colors))
        label_to_color = [color_map[label] for label in labels]  

        # 把labels转换为RGBA格式  
        labels_rgba = np.array(label_to_color, dtype=np.uint8) 
        pc.point_data[f"{anno}_rgba"] = labels_rgba
        
        # pyvista 绘制
        p = pv.Plotter(off_screen=True)     # 导出html建议 off_screen
        p.set_background("white")
        p.show_axes() 

        p.add_mesh(pc, scalars=f"{anno}_rgba", rgba=True, point_size=5, render_points_as_spheres=True)  

        # 添加图例  
        legend_entries = []  
        for label in unique_labels:
            color = color_map[label]
            legend_entries.append([f"{anno} {label}", color])  
        p.add_legend(legend_entries, bcolor="white")  

        p.show()  
                


def mesh4pyvista(pc, alpha=2, smooth_iter=1000):  
    """
    Generate a 3D tetrahedral mesh based on pyvista (VT3D-style with post-processing).
    
    Args:
        pc: PyVista PolyData point cloud
        alpha: Alpha value for Delaunay 3D
        smooth_iter: Number of smoothing iterations
    """
    _validate_point_cloud(pc, min_points=10)
    alpha = float(alpha)
    smooth_iter = int(smooth_iter)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")

    print(f"Starting PyVista Delaunay 3D with {pc.n_points} points, alpha={alpha}")
    
    # For large point clouds, use smaller alpha or recommend sampling
    if pc.n_points > 50000:
        print(f"⚠ Large point cloud ({pc.n_points} points). Consider reducing sampling ratio or using Poisson reconstruction.")
        # Automatically adjust alpha for large datasets
        if alpha > 0:
            alpha = min(alpha, 0.5)
            print(f"Auto-adjusted alpha to {alpha} for large dataset")
    
    # Generate initial mesh with error handling
    try:
        print("Generating Delaunay 3D mesh...")
        mesh = pc.delaunay_3d(alpha=alpha)
        print(f"Delaunay 3D complete: {mesh.n_points} points, {mesh.n_cells} cells")
        
        print("Extracting surface...")
        mesh = mesh.extract_surface()
        print(f"After extract_surface: {mesh.n_points} points, {mesh.n_cells} faces")

        clean_tolerance = _adaptive_clean_tolerance(np.asarray(pc.points))
        mesh = mesh.triangulate().clean(tolerance=clean_tolerance)
        print(f"After triangulate+clean: {mesh.n_points} points, {mesh.n_cells} faces")
        
        if mesh.n_points == 0 or mesh.n_cells == 0:
            raise ValueError(
                f"\nDelaunay 3D produced empty mesh. Try: 1) Reduce sampling ratio to 25-50%, "
                f"2) Adjust alpha parameter, or 3) Use Poisson reconstruction for large datasets."
            )
    except Exception as e:
        raise ValueError(
            f"\nPyVista Delaunay 3D failed: {str(e)}\n"
            f"Suggestions:\n"
            f"  - Reduce sampling ratio to 25-50% (current: {pc.n_points} points)\n"
            f"  - Try smaller alpha value (current: {alpha})\n"
            f"  - Use Poisson reconstruction for point clouds > 50k points"
        )
    
    # VT3D-style post-processing (skip if mesh is too small)
    if mesh.n_points < 10:
        print("⚠ Mesh too small, skipping post-processing")
        return mesh
    
    # Split into separate bodies
    print("Splitting bodies...")
    sub_meshes = mesh.split_bodies()
    print(f"Found {len(sub_meshes)} body/bodies")
    
    if len(sub_meshes) > 1:
        # Merge multiple bodies
        print("Merging bodies...")
        merged_mesh = sub_meshes[0]
        for next_mesh in sub_meshes[1:]:
            merged_mesh = merged_mesh.merge(next_mesh)
        mesh = merged_mesh
        print(f"After merge: {mesh.n_points} points, {mesh.n_cells} faces")
    
    # Fix mesh using pymeshfix (VT3D approach)
    if mesh.n_points > 0:
        try:
            print("Repairing mesh with pymeshfix...")
            meshfix = mf.MeshFix(mesh)
            meshfix.repair(verbose=False)
            mesh = meshfix.mesh.triangulate().clean()
            print(f"✓ Mesh repaired: {mesh.n_points} points, {mesh.n_cells} faces")
        except Exception as e:
            print(f"Warning: pymeshfix repair failed: {e}")
    
    # Smooth the mesh (VT3D approach)
    if smooth_iter > 0 and mesh.n_points > 0:
        print(f"Smoothing mesh with {smooth_iter} iterations...")
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✓ Mesh smoothed: {mesh.n_points} points, {mesh.n_cells} faces")
    
    # Compute consistent normals to fix lighting artifacts
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    print(f"✓ Normals computed and oriented")

    return mesh


def mesh4alphashape(pc, alpha=2.0, smooth_iter=500):  
    """
    Alpha shape mesh generation with VT3D-style post-processing.
    
    Args:
        pc: PyVista PolyData point cloud
        alpha: Alpha value for alpha shape
        smooth_iter: Number of smoothing iterations
    """
    _validate_point_cloud(pc, min_points=10)
    alpha = float(alpha)
    smooth_iter = int(smooth_iter)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")

    cloud = _pv2o3d(pc)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha) 
    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"The point cloud cannot generate a surface mesh with `alpha shape` method and alpha == {alpha}."
        )

    mesh = _o3d2pv(trimesh=trimesh)
    
    # VT3D-style post-processing
    try:
        meshfix = mf.MeshFix(mesh)
        meshfix.repair(verbose=False)
        mesh = meshfix.mesh.triangulate().clean()
        print(f"✓ Alpha shape mesh repaired")
    except Exception as e:
        print(f"Warning: mesh repair failed: {e}")
    
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✓ Mesh smoothed with {smooth_iter} iterations")
    
    # Compute consistent normals to fix lighting artifacts
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    return mesh


def mesh4ballpivoting(pc, radii=[1], smooth_iter=500):
    """
    Ball pivoting mesh generation with VT3D-style post-processing.
    
    Args:
        pc: PyVista PolyData point cloud
        radii: List of radii for ball pivoting
        smooth_iter: Number of smoothing iterations
    """
    _validate_point_cloud(pc, min_points=10)
    smooth_iter = int(smooth_iter)
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")

    cloud = _pv2o3d(pc)
    if radii is None or len(radii) == 0:
        nn = _estimate_nn_distance(np.asarray(pc.points))
        radii = [nn, nn * 2.0, nn * 4.0]

    radii = sorted({float(r) for r in radii if float(r) > 0})
    if len(radii) == 0:
        raise ValueError("radii must contain at least one positive number.")

    radii_o3d = o3d.utility.DoubleVector(radii)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii_o3d)

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"The point cloud cannot generate a surface mesh with `ball pivoting` method and radii == {radii}."
        )

    mesh = _o3d2pv(trimesh=trimesh)
    
    # VT3D-style post-processing
    try:
        meshfix = mf.MeshFix(mesh)
        meshfix.repair(verbose=False)
        mesh = meshfix.mesh.triangulate().clean()
        print(f"✓ Ball pivoting mesh repaired")
    except Exception as e:
        print(f"Warning: mesh repair failed: {e}")
    
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✓ Mesh smoothed with {smooth_iter} iterations")
    
    # Compute consistent normals
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    return mesh



def mesh4poisson(pc, depth=8, width=0, scale=1.1, linear_fit=False, density_threshold=0.01, smooth_iter=500): 
    """
    Poisson surface reconstruction with VT3D-style post-processing.
    
    Args:
        pc: PyVista PolyData point cloud
        depth: Octree depth for Poisson reconstruction
        width, scale, linear_fit: Poisson parameters
        density_threshold: Density quantile threshold for vertex removal
        smooth_iter: Number of smoothing iterations
    """
    _validate_point_cloud(pc, min_points=20)
    depth = int(depth)
    width = int(width)
    scale = float(scale)
    smooth_iter = int(smooth_iter)
    if depth < 1 or depth > 14:
        raise ValueError(f"depth must be in [1, 14], got {depth}.")
    if width < 0:
        raise ValueError(f"width must be >= 0, got {width}.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")
    if density_threshold is not None:
        density_threshold = float(density_threshold)
        if density_threshold < 0 or density_threshold > 1:
            raise ValueError(f"density_threshold must be in [0, 1], got {density_threshold}.")

    cloud = _pv2o3d(pc)
    trimesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    if len(trimesh.vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `poisson` method and depth == {depth}.")

    # A low density value means that the vertex is only supported by a low number of points from the input point cloud.
    # Remove all vertices (and connected triangles) that have a lower density value than the density_threshold quantile
    # of all density values.
    if not (density_threshold is None):
        trimesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, density_threshold))

    mesh = _o3d2pv(trimesh=trimesh)
    
    # VT3D-style post-processing
    try:
        meshfix = mf.MeshFix(mesh)
        meshfix.repair(verbose=False)
        mesh = meshfix.mesh.triangulate().clean()
        print(f"✓ Poisson mesh repaired")
    except Exception as e:
        print(f"Warning: mesh repair failed: {e}")
    
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✓ Mesh smoothed with {smooth_iter} iterations")
    
    # Compute consistent normals
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    return mesh


def mesh4marchingcube(pc, levelset=0, mc_scale_factor=1, dist_sample_num=100, smooth_iter=500):
    """
    Marching cubes mesh generation with VT3D-style post-processing.
    
    Args:
        pc: PyVista PolyData point cloud
        levelset: Iso-surface level for marching cubes
        mc_scale_factor: Scale factor for marching cubes
        dist_sample_num: Number of samples for distance calculation
        smooth_iter: Number of smoothing iterations
    """
    _validate_point_cloud(pc, min_points=10)
    smooth_iter = int(smooth_iter)
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")

    mc_scale_factor = float(mc_scale_factor)
    if mc_scale_factor <= 0:
        raise ValueError(f"mc_scale_factor must be > 0, got {mc_scale_factor}.")

    if dist_sample_num is not None:
        dist_sample_num = int(dist_sample_num)
        if dist_sample_num <= 0:
            raise ValueError(f"dist_sample_num must be > 0 when provided, got {dist_sample_num}.")

    # Move the model so that the coordinate minimum is at (0, 0, 0).
    raw_points = np.asarray(pc.points).copy()
    new_points = raw_points - np.min(raw_points, axis=0)

    # Generate new models for calculatation.
    if dist_sample_num is None:
        dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
        row, col = np.diag_indices_from(dist)
        dist[row, col] = np.nan
    else:
        rng = np.random.default_rng(42)
        rand_idx = (
            rng.choice(new_points.shape[0], dist_sample_num, replace=False)
            if new_points.shape[0] >= dist_sample_num
            else np.arange(new_points.shape[0])
        )
        dist = cdist(XA=new_points[rand_idx, :], XB=new_points, metric="euclidean")
        dist[np.arange(rand_idx.shape[0]), rand_idx] = np.nan
    nearest_dist = np.nanmin(dist, axis=1)
    max_dist = float(np.nanmax(nearest_dist))
    if not np.isfinite(max_dist) or max_dist <= 0:
        raise ValueError("Invalid nearest-neighbor distance estimated for marching cube scaling.")
    mc_sf = max_dist * mc_scale_factor

    centered_pc = pv.PolyData(new_points)
    scale_pc = scale_model(model=centered_pc, scale_factor=1 / mc_sf, scale_center=(0, 0, 0))
    scale_pc_points = scale_pc.points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)

    # Generate grid for calculatation based on new model.
    volume_array = np.zeros(
        shape=[
            scale_pc_points[:, 0].max() + 3,
            scale_pc_points[:, 1].max() + 3,
            scale_pc_points[:, 2].max() + 3,
        ]
    )
    volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    # mcubes.marching_cubes_func(volume, isovalue, xmin, ymin, zmin, xmax, ymax)
    # Note: 7 positional arguments, bounds are separate integers
    shape = volume_array.shape
    vertices, triangles = mcubes.marching_cubes(volume_array, levelset)
    # vertices, triangles = mcubes.marching_cubes_func(
    #     volume_array, 
    #     (levelset), 
    #     0, 0, 0,  # xmin, ymin, zmin
    #     int(shape[0]-1), 
    #     int(shape[1]-1)
    # )

    if len(vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    # Generate mesh model.
    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    mesh.clean(inplace=True)
    mesh = scale_model(model=mesh, scale_factor=mc_sf, scale_center=(0, 0, 0))

    # Transform.
    scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf, scale_center=(0, 0, 0))
    mesh.points = rigid_transform(
        coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
    )
    
    # VT3D-style post-processing
    try:
        meshfix = mf.MeshFix(mesh)
        meshfix.repair(verbose=False)
        mesh = meshfix.mesh.triangulate().clean()
        print(f"✓ Marching cubes mesh repaired")
    except Exception as e:
        print(f"Warning: mesh repair failed: {e}")
    
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✓ Mesh smoothed with {smooth_iter} iterations")
    
    # Compute consistent normals
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    return mesh

     
def mesh4voxel(
    pc,
    grid_size: int = 60,
    sigma: float = 2.5,
    iso_percentile: float = 70,
    hole_size_factor: float = 0.6,
    smooth_iter: int = 30,
    closure_smooth_factor: float = 1.0,
):
    from scipy.ndimage import gaussian_filter
    points = np.asarray(pc.points)

    _validate_point_cloud(pc, min_points=10)

    if points.size == 0:
        raise ValueError("Empty point cloud.")

    grid_size = int(grid_size)
    if grid_size < 5:
        raise ValueError(f"grid_size must be >= 5, got {grid_size}.")

    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    iso_percentile = float(iso_percentile)
    if not (0 < iso_percentile < 100):
        raise ValueError(f"iso_percentile must be between (0, 100), got {iso_percentile}.")

    hole_size_factor = float(hole_size_factor)
    if hole_size_factor < 0:
        raise ValueError(f"hole_size_factor must be >= 0, got {hole_size_factor}.")

    smooth_iter = int(smooth_iter)
    if smooth_iter < 0:
        raise ValueError(f"smooth_iter must be >= 0, got {smooth_iter}.")

    closure_smooth_factor = float(closure_smooth_factor)
    if closure_smooth_factor <= 0:
        raise ValueError(f"closure_smooth_factor must be > 0, got {closure_smooth_factor}.")

    # 建体素网格
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    if np.any(spans == 0):
        raise ValueError("Point cloud is degenerate (one or more axes have zero span).")

    scale = (grid_size - 1) / spans

    vox = np.zeros((grid_size,)*3, dtype=np.float32)

    idx = ((points - mins) * scale).astype(int)
    idx = np.clip(idx, 0, grid_size-1)
    vox[idx[:,0], idx[:,1], idx[:,2]] += 1

    # 高斯平滑 = “组织感”
    vox = gaussian_filter(vox, sigma=sigma)
    
    spacing = (maxs - mins) / (grid_size - 1)
    grid = pv.ImageData(
        dimensions=vox.shape,
        spacing=tuple(spacing),
        origin=tuple(mins),
    )
    grid.point_data["density"] = vox.flatten(order="F")
    surface = grid.contour(isosurfaces=[np.percentile(vox, iso_percentile)])
    surface = surface.extract_surface().triangulate().clean()

    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(
            "Voxel contour produced empty mesh. Try: increase grid_size, decrease iso_percentile, or decrease sigma."
        )

    # ====== 封上开口/缺口（首尾） ======
    # 说明：fill_holes 的 size 是“要填补的孔洞的最大直径（世界坐标单位）”
    bounds = np.array(surface.bounds).reshape(3, 2)
    diag = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    hole_size = diag * hole_size_factor  # 可调：0.2~0.6，越大越容易把首尾封上
    surface = surface.fill_holes(hole_size).extract_surface().triangulate().clean()

    # 对封洞后的三角面进行轻度细分，减少“盖帽”处的大面片感
    if surface.n_points > 0 and surface.n_points < 300000:
        try:
            surface = surface.subdivide(1, subfilter="loop").clean()
        except Exception:
            pass

    # 封洞后再轻微平滑一次，让补上的“盖子”过渡更自然
    if smooth_iter > 0:
        taubin_iter = max(10, int(max(40, smooth_iter) * closure_smooth_factor))
        laplace_iter = max(5, int(max(10, smooth_iter // 2) * closure_smooth_factor))

        try:
            # 第一阶段：Taubin 平滑，抑制收缩并改善封洞处折线感
            surface = surface.smooth_taubin(
                n_iter=taubin_iter,
                pass_band=0.08
            ).clean()
        except Exception:
            pass

        # 第二阶段：轻度拉普拉斯平滑，进一步消除局部尖折
        surface = surface.smooth(
            n_iter=laplace_iter,
            relaxation_factor=0.01,
            feature_smoothing=False,
            boundary_smoothing=True,
        ).clean()

    # 重算法线，改善光照下的块面感
    surface = surface.compute_normals(auto_orient_normals=True, consistent_normals=True)
        
    return surface      

     


def construct_surface(pc, cs_method, cs_args, smooth=3000, nsub=3, nclus=20000, scale_distance=None, scale_factor=1): 
    # Surface mesh reconstruction based on 3D point cloud model.
    """
    # 3D 重构方法 
    'pyvista': Generate a 3D tetrahedral mesh based on pyvista.
    'alpha_shape': Computes a triangle mesh on the alpha shape algorithm.
    'ball_pivoting': Computes a triangle mesh based on the Ball Pivoting algorithm.
    'poisson': Computes a triangle mesh based on thee Screened Poisson Reconstruction.
    'marching_cube': Computes a triangle mesh based on the marching cube algorithm.
    'voxel': Computes a surface mesh from a voxelized density field + contour.

    # 3D 重构方法参数 
    'pyvista': {'alpha': 0}
    'alpha_shape': {'alpha': 2.0}
    'ball_pivoting': {'radii': [1]}
    'poisson': {'depth': 8, 'width'=0, 'scale'=1.1, 'linear_fit': False, 'density_threshold': 0.01}
    'marching_cube': {'levelset': 0, 'mc_scale_factor': 1, 'dist_sample_num': 100}
    'voxel': {'grid_size': 60, 'sigma': 2.5, 'iso_percentile': 70, 'hole_size_factor': 0.6, 'smooth_iter': 30}
    
    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            ``uniform_surf.cell_data[key_added]``, the ``label`` array;
            ``uniform_surf.cell_data[f'{key_added}_rgba']``, the rgba colors of the ``label`` array.
        inside_pc: A point cloud, which contains the following properties:
            ``inside_pc.point_data['obs_index']``, the obs_index of each coordinate in the original adata.
            ``inside_pc.point_data[key_added]``, the ``groupby`` information.
            ``inside_pc.point_data[f'{key_added}_rgba']``, the rgba colors of the ``groupby`` information.
        plot_cmap: Recommended colormap parameter values for plotting.
    
    """
    method_aliases = {
        "alphashape": "alpha_shape",
        "ballpivoting": "ball_pivoting",
        "marchingcube": "marching_cube",
    }
    cs_method = method_aliases.get(cs_method, cs_method)

    if cs_method == "pyvista":
        surf = mesh4pyvista(pc=pc, **cs_args)
    elif cs_method == "alpha_shape":
        surf = mesh4alphashape(pc=pc, **cs_args)
    elif cs_method == "ball_pivoting":
        surf = mesh4ballpivoting(pc=pc, **cs_args)
    elif cs_method == "poisson":
        surf = mesh4poisson(pc=pc, **cs_args)
    elif cs_method == "marching_cube":
        surf = mesh4marchingcube(pc=pc, **cs_args)
    elif cs_method == "voxel":
        surf = mesh4voxel(pc=pc, **cs_args)
        return surf 
    else:
        raise ValueError(f"`cs_method` value '{cs_method}' is not supported.")  
    
    # Removes unused points and degenerate cells.
    csurf = clean_mesh(mesh=surf) 
    uniform_surfs = []
    for sub_surf in csurf.split_bodies():
        # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
        sub_fix_surf = fix_mesh(mesh=sub_surf.extract_surface())

        # Get a uniformly meshed surface using voronoi clustering.
        sub_uniform_surf = uniform_mesh(mesh=sub_fix_surf, nsub=nsub, nclus=nclus)
        uniform_surfs.append(sub_uniform_surf)
    uniform_surf = merge_models(models=uniform_surfs)
    uniform_surf = uniform_surf.extract_surface().triangulate().clean()

    # Adjust point coordinates using Laplacian smoothing.
    if not (smooth is None):
        uniform_surf = smooth_mesh(mesh=uniform_surf, n_iter=smooth)

    # Scale the surface mesh.
    uniform_surf = scale_model(model=uniform_surf, distance=scale_distance, scale_factor=scale_factor)
    return uniform_surf  



def export_mesh_to_obj(mesh: pv.PolyData, filename: str) -> str:
    """
    Export mesh to OBJ file (VT3D compatible format)
    
    Args:
        mesh: PyVista PolyData mesh
        filename: Output filename (.obj)
        
    Returns:
        Path to exported file
    """
    try:
        mesh.save(filename)
        print(f"✓ Exported mesh to {filename}: {mesh.n_points} vertices, {mesh.n_cells} faces")
        return filename
    except Exception as e:
        print(f"Error exporting mesh: {str(e)}")
        return None


def export_mesh_to_json(mesh: pv.PolyData, filename: str) -> str:
    """
    Export mesh to JSON format (VT3D structure)
    
    Args:
        mesh: PyVista PolyData mesh
        filename: Output filename (.json)
        
    Returns:
        Path to exported file
    """
    import json
    
    try:
        # Extract vertices and faces
        vertices = mesh.points.tolist()
        
        # Get faces (PyVista format: [n_points, v0, v1, v2, ...])
        faces_raw = mesh.faces
        faces = []
        i = 0
        while i < len(faces_raw):
            n_points = faces_raw[i]
            if n_points == 3:  # Triangle
                faces.append([faces_raw[i+1], faces_raw[i+2], faces_raw[i+3]])
            i += n_points + 1
        
        # Create JSON structure
        mesh_data = {
            'vectors': vertices,
            'faces': faces,
            'n_points': mesh.n_points,
            'n_faces': len(faces)
        }
        
        with open(filename, 'w') as f:
            json.dump(mesh_data, f)
        
        print(f"✓ Exported mesh to JSON: {mesh.n_points} vertices, {len(faces)} faces")
        return filename
        
    except Exception as e:
        print(f"Error exporting mesh to JSON: {str(e)}")
        return None






