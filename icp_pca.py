import open3d as o3d
import numpy as np

def load_and_preprocess(filename, z_threshold=1.0):
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # סינון לפי Z קדמי
    mask = points[:, 2] < np.percentile(points[:, 2], z_threshold)
    filtered = pcd.select_by_index(np.where(mask)[0])
    
    return filtered

def align_roll_with_pca(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    _, _, vh = np.linalg.svd(cov)
    R_align = vh.T

    # נוודא שה-Z מצביע קדימה
    if R_align[2, 2] < 0:
        R_align[:, 2] *= -1

    pcd.rotate(R_align, center=centroid)
    return pcd

def run_color_icp(source_pcd, target_pcd, voxel_size=0.01):
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, voxel_size,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=50))
    return result_icp.transformation

def transform_and_merge(source, target, transform):
    source_copy = source.transform(transform)
    merged = source_copy + target
    return merged

# === MAIN ===
source_path = "frame_0000_1.ply"
target_path = "frame_0000.ply"

# 1. Load and filter
source_pcd = load_and_preprocess(source_path)
target_pcd = load_and_preprocess(target_path)

# 2. Align each to remove ROLL
source_pcd = align_roll_with_pca(source_pcd)
target_pcd = align_roll_with_pca(target_pcd)

# 3. Run color ICP
trans = run_color_icp(source_pcd, target_pcd)

# 4. Transform and merge
merged = transform_and_merge(source_pcd, target_pcd, trans)

# 5. Save result
o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/fused_color_icp.ply", merged)
o3d.visualization.draw_geometries([merged])
