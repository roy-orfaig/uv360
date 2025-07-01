import numpy as np
import open3d as o3d

def load_ply_points(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def filter_front(points, fraction=0.1):
    z_min = np.percentile(points[:, 2], fraction * 100)
    return points[points[:, 2] < z_min]

def compute_heading_pca(points):
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    heading_vector = vh[0]
    return heading_vector / np.linalg.norm(heading_vector)

def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.99999:
        return -np.eye(3)  # 180 degree flip
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2 + 1e-10))

def transform_point_cloud(points, R, t):
    return (R @ points.T).T + t

def main():
    file0 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply"
    file1 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply"

    points0, colors0 = load_ply_points(file0)
    points1, colors1 = load_ply_points(file1)

    front0 = filter_front(points0)
    front1 = filter_front(points1)

    heading0 = compute_heading_pca(front0)
    heading1 = compute_heading_pca(front1)

    # Project to horizontal plane (ignore Z)
    heading0[2] = 0
    heading1[2] = 0

    R = rotation_matrix_from_vectors(heading1, heading0)

    # Rotate point cloud 1 to align with point cloud 0
    rotated1 = transform_point_cloud(points1, R, np.zeros(3))

    # Translate to roughly align centroids
    t = points0.mean(axis=0) - rotated1.mean(axis=0)
    aligned1 = rotated1 + t

    # Combine clouds
    fused_points = np.vstack([points0, aligned1])
    fused_colors = np.vstack([colors0, colors1])

    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(fused_points)
    pcd_combined.colors = o3d.utility.Vector3dVector(fused_colors)

    o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/fused_heading_align.ply", pcd_combined)
    print("Saved fused point cloud to fused_heading_align.ply")

if __name__ == "__main__":
    main()
    
# import open3d as o3d
# import numpy as np
# from sklearn.decomposition import PCA

# # Load PLYs
# pcd0 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply")
# pcd1 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply")

# # Downsample
# pcd0_down = pcd0.voxel_down_sample(voxel_size=0.01)
# pcd1_down = pcd1.voxel_down_sample(voxel_size=0.01)

# # Estimate Yaw via PCA (XY only)
# def compute_yaw_pca(pcd):
#     xyz = np.asarray(pcd.points)[:, :2]
#     pca = PCA(n_components=2).fit(xyz)
#     heading = pca.components_[0]
#     yaw = np.arctan2(heading[1], heading[0])
#     return yaw

# yaw0 = compute_yaw_pca(pcd0_down)
# yaw1 = compute_yaw_pca(pcd1_down)
# delta_yaw = yaw0 - yaw1

# # Create rotation matrix around Z axis (Yaw)
# cos_yaw, sin_yaw = np.cos(delta_yaw), np.sin(delta_yaw)
# R_yaw = np.array([
#     [cos_yaw, -sin_yaw, 0],
#     [sin_yaw,  cos_yaw, 0],
#     [0,             0, 1]
# ])

# # Center → Rotate → Return
# center1 = np.mean(np.asarray(pcd1.points), axis=0)
# pcd1.translate(-center1)
# pcd1.rotate(R_yaw)
# pcd1.translate(center1)

# # Align front bumpers
# def front_centroid(pcd, percentile=5):
#     xyz = np.asarray(pcd.points)
#     z_threshold = np.percentile(xyz[:, 2], percentile)
#     front = xyz[xyz[:, 2] <= z_threshold]
#     return np.mean(front, axis=0)

# delta = front_centroid(pcd0_down) - front_centroid(pcd1_down)
# pcd1.translate(delta)

# # Merge and save
# merged = pcd0 + pcd1
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/fused_yaw_only_pca.ply", merged)
# print("✅ Saved fused_yaw_only_pca.ply")
