import numpy as np
import trimesh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# === Utilities ===

def extract_front_region(xyz, ratio=0.1):
    """Return front bumper region (closest Z points)"""
    z_thresh = np.percentile(xyz[:, 2], ratio * 100)
    return xyz[xyz[:, 2] <= z_thresh]

def get_facing_direction_fixed(xyz, target_dir=np.array([0, 0, 1])):
    """Get principal direction of bumper, flipped to match target_dir"""
    xyz_centered = xyz - xyz.mean(axis=0)
    pca = PCA(n_components=3).fit(xyz_centered)
    principal = pca.components_[0]
    if np.dot(principal, target_dir) < 0:
        principal *= -1
    return principal

def rotation_matrix_from_vectors(vec1, vec2):
    """Find rotation matrix that aligns vec1 to vec2"""
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / s ** 2)

def run_icp(A, B, max_iter=20, tolerance=1e-5):
    """Basic ICP implementation"""
    src = B.copy()
    prev_error = np.inf
    for _ in range(max_iter):
        nbrs = NearestNeighbors(n_neighbors=1).fit(A)
        distances, indices = nbrs.kneighbors(src)
        tgt = A[indices[:, 0]]

        centroid_src = src.mean(axis=0)
        centroid_tgt = tgt.mean(axis=0)

        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        src = (R @ src.T).T + t
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    return src

# === Load Data ===

# Change to your paths:
pcd0 = trimesh.load("frame_0000.ply")
pcd1 = trimesh.load("frame_0000_1.ply")

xyz0 = pcd0.vertices[::40]
xyz1 = pcd1.vertices[::40]
rgb0 = pcd0.visual.vertex_colors[::40, :3]
rgb1 = pcd1.visual.vertex_colors[::40, :3]

# === Step 1: Extract Bumpers ===
front0 = extract_front_region(xyz0)
front1 = extract_front_region(xyz1)

# === Step 2: Align Facing to +Z ===
dir0 = get_facing_direction_fixed(front0)
dir1 = get_facing_direction_fixed(front1)

R0 = rotation_matrix_from_vectors(dir0, np.array([0, 0, 1]))
R1 = rotation_matrix_from_vectors(dir1, np.array([0, 0, 1]))

xyz0_rot = (R0 @ (xyz0 - xyz0.mean(axis=0)).T).T
xyz1_rot = (R1 @ (xyz1 - xyz1.mean(axis=0)).T).T

# === Step 3: ICP Alignment ===
xyz1_icp = run_icp(xyz0_rot, xyz1_rot)

# === Step 4: Merge & Save ===
xyz_fused = np.vstack([xyz0_rot, xyz1_icp])
rgb_fused = np.vstack([rgb0, rgb1])
fused = trimesh.PointCloud(vertices=xyz_fused.astype(np.float32),
                           colors=rgb_fused.astype(np.uint8))

fused.export("fused_pca_fixed_icp_frame_0000.ply")
print("✅ Saved: fused_pca_fixed_icp_frame_0000.ply")
