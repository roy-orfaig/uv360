

# import open3d as o3d
# import numpy as np

# # Load the point clouds
# pcd0 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply")      # צד ימין
# pcd1 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply")    # צד שמאל

# def extract_front_region(pcd, percentile=5):
#     """
#     Extracts the front part of the car based on Z (depth).
#     Smaller Z is assumed to be more frontal.
#     """
#     points = np.asarray(pcd.points)
#     z_threshold = np.percentile(points[:, 2], percentile)
#     mask = points[:, 2] <= z_threshold
#     return pcd.select_by_index(np.where(mask)[0])

# # 1. Extract front 5% of the car from both sides
# front0 = extract_front_region(pcd0, percentile=5)
# front1 = extract_front_region(pcd1, percentile=5)

# # 2. Align front centroids
# centroid0 = np.mean(np.asarray(front0.points), axis=0)
# centroid1 = np.mean(np.asarray(front1.points), axis=0)
# initial_translation = centroid0 - centroid1
# pcd1_translated = pcd1.translate(initial_translation)

# # 3. ICP refinement (only slight adjustment)
# reg = o3d.pipelines.registration.registration_icp(
#     pcd1_translated, pcd0,
#     max_correspondence_distance=0.05,
#     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
# )

# # 4. Apply refined transformation
# pcd1_aligned = pcd1_translated.transform(reg.transformation)

# # 5. Merge both clouds
# merged = pcd0 + pcd1_aligned

# # Optional: downsample for better viewing
# merged = merged.voxel_down_sample(voxel_size=0.002)

# # 6. Save result
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/final_registered_fused.ply", merged)
# print("Saved merged cloud to final_registered_fused.ply")



# import open3d as o3d
# import numpy as np
# import copy

# # Load point clouds
# pcd0 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000.ply")
# pcd1 = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/xyz_rgb/frame_0000_1.ply")

# # Downsample
# pcd0_down = pcd0.voxel_down_sample(0.01)
# pcd1_down = pcd1.voxel_down_sample(0.01)
# pcd0_down.estimate_normals()
# pcd1_down.estimate_normals()

# # PCA to get orientation
# def compute_pca(pcd):
#     pts = np.asarray(pcd.points)
#     centroid = pts.mean(axis=0)
#     centered = pts - centroid
#     cov = np.cov(centered.T)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     return eigvecs[:, np.argsort(eigvals)[::-1]], centroid

# R0, c0 = compute_pca(pcd0_down)
# R1, c1 = compute_pca(pcd1_down)

# # Initial alignment matrix
# R_align = R0 @ R1.T
# U, _, Vt = np.linalg.svd(R_align)
# R_init = U @ Vt

# # Convert to Euler angles and clip roll
# def R2eul(R):
#     sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
#     if sy < 1e-6:
#         return np.rad2deg([np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0])
#     else:
#         return np.rad2deg([np.arctan2(R[2,1], R[2,2]),
#                            np.arctan2(-R[2,0], sy),
#                            np.arctan2(R[1,0], R[0,0])])

# def eul2R(rx, ry, rz):
#     rx, ry, rz = np.deg2rad([rx, ry, rz])
#     Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
#     Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
#     Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
#     return Rz @ Ry @ Rx

# roll, pitch, yaw = R2eul(R_init)
# roll = np.clip(roll, -30, 30)
# R_limited = eul2R(roll, pitch, yaw)

# # Apply rotation and recenter
# pts1 = np.asarray(pcd1.points)
# pts1_centered = pts1 - c1
# transformed = (R_limited @ pts1_centered.T).T + c0
# pcd1_trans = copy.deepcopy(pcd1)
# pcd1_trans.points = o3d.utility.Vector3dVector(transformed)

# # ICP
# reg = o3d.pipelines.registration.registration_icp(
#     pcd1_trans, pcd0, 0.05, np.eye(4),
#     o3d.pipelines.registration.TransformationEstimationPointToPoint()
# )
# pcd1_final = copy.deepcopy(pcd1_trans).transform(reg.transformation)

# # Merge
# fused = pcd0 + pcd1_final
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/Downloads/fused_roll_limited_icp.ply", fused)
