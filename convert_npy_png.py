import open3d as o3d
import numpy as np

def load_and_filter_headlight_region(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # הגדרת מסכה לפי צבע צהבהב-בהיר (RGB נורמלי)
    mask_color = (colors[:, 0] > 0.4) & (colors[:, 1] > 0.4) & (colors[:, 2] < 0.5)

    # הגבלת האזור לחזית (ה־Z הכי נמוך)
    z_threshold = np.percentile(points[:, 2], 5)
    mask_z = points[:, 2] < z_threshold

    # שילוב שתי המסכות
    mask = mask_color & mask_z

    print(f"[INFO] {ply_path}: Found {np.sum(mask)} headlight-like points")

    headlight_pcd = o3d.geometry.PointCloud()
    headlight_pcd.points = o3d.utility.Vector3dVector(points[mask])
    headlight_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return headlight_pcd

# נתיבים לקבצים
ply_path0 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply"
ply_path1 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply"

# מיצוי האזורים של הפנסים
headlight0 = load_and_filter_headlight_region(ply_path0)
headlight1 = load_and_filter_headlight_region(ply_path1)

# שמירת קבצים לבדיקה
o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/headlights_0.ply", headlight0)
o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/headlights_1.ply", headlight1)

# import open3d as o3d
# import numpy as np

# def load_point_cloud_with_colors(ply_path):
#     pcd = o3d.io.read_point_cloud(ply_path)
#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors) / 255.0  
#     return points, colors, pcd

# def detect_bright_yellow_regions(points, colors, brightness_thresh=0.8):
#     brightness = np.mean(colors, axis=1)
#     is_yellow = (colors[:, 0] > 0.6) & (colors[:, 1] > 0.6) & (colors[:, 2] < 0.3)
#     is_bright = brightness > brightness_thresh
#     mask = is_yellow & is_bright
#     return points[mask], colors[mask]

# # Load point clouds
# ply0 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply"
# ply1 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply"
# points0, colors0, pcd0 = load_point_cloud_with_colors(ply0)
# points1, colors1, pcd1 = load_point_cloud_with_colors(ply1)

# # Extract headlight-like regions
# headlight_pts0, headlight_colors0 = detect_bright_yellow_regions(points0, colors0)
# headlight_pts1, headlight_colors1 = detect_bright_yellow_regions(points1, colors1)

# pcd_crop0 = o3d.geometry.PointCloud()
# pcd_crop0.points = o3d.utility.Vector3dVector(headlight_pts0)
# pcd_crop0.colors = o3d.utility.Vector3dVector(headlight_colors0)

# pcd_crop1 = o3d.geometry.PointCloud()
# pcd_crop1.points = o3d.utility.Vector3dVector(headlight_pts1)
# pcd_crop1.colors = o3d.utility.Vector3dVector(headlight_colors1)

# # Save cropped clouds for alignment
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/headlights_frame_0000.ply", pcd_crop0)
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/headlights_frame_0000_1.ply", pcd_crop1)

# import open3d as o3d
# import numpy as np

# def load_ply_points(filename):
#     return o3d.io.read_point_cloud(filename)

# def filter_front(pcd, fraction=0.1):
#     points = np.asarray(pcd.points)
#     z_min = np.percentile(points[:, 2], fraction * 100)
#     indices = np.where(points[:, 2] < z_min)[0]
#     return pcd.select_by_index(indices)

# def compute_heading_pca(pcd):
#     points = np.asarray(pcd.points)
#     centered = points - points.mean(axis=0)
#     _, _, vh = np.linalg.svd(centered, full_matrices=False)
#     heading = vh[0]
#     heading[2] = 0  # רק בזווית YAW
#     return heading / np.linalg.norm(heading)

# def rotation_matrix_from_vectors(a, b):
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     if c < -0.99999:
#         return -np.eye(3)
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -v[2], v[1]],
#                      [v[2], 0, -v[0]],
#                      [-v[1], v[0], 0]])
#     return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2 + 1e-10))

# def apply_transformation(pcd, R, t):
#     return pcd.rotate(R).translate(t)

# def run_icp_colored(source, target, voxel_size=0.01):
#     source_down = source.voxel_down_sample(voxel_size)
#     target_down = target.voxel_down_sample(voxel_size)

#     source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
#     target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

#     result = o3d.pipelines.registration.registration_colored_icp(
#         source_down, target_down, voxel_size,
#         np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationForColoredICP(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
#     )
#     return result.transformation

# def main():
#     file0 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply"
#     file1 = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply"

#     pcd0 = load_ply_points(file0)
#     pcd1 = load_ply_points(file1)

#     # Heading alignment
#     heading0 = compute_heading_pca(filter_front(pcd0))
#     heading1 = compute_heading_pca(filter_front(pcd1))
#     R = rotation_matrix_from_vectors(heading1, heading0)
#     t = pcd0.get_center() - pcd1.get_center()
#     pcd1_aligned = apply_transformation(pcd1, R, t)

#     # Colored ICP
#     T_icp = run_icp_colored(pcd1_aligned, pcd0)
#     pcd1_icp = pcd1_aligned.transform(T_icp)

#     # Merge
#     fused = pcd0 + pcd1_icp
#     o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/fused_heading_icp_color.ply", fused)
#     print("✅ Saved: fused_heading_icp_color.ply")

# if __name__ == "__main__":
#     main()
