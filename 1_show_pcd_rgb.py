import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Step 1: Load inputs ---
depth_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_marigold/frame_0000.npy"
rgb_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colors/frame_0000.png"
intrinsics_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_output/intrinsic/frame_0000.txt"

depth_map = np.load(depth_path)              # (H, W), in meters
rgb_image = np.array(Image.open(rgb_path))   # (H, W, 3)
K = np.loadtxt(intrinsics_path)              # 3x3 intrinsic matrix

# --- Step 2: Convert depth to point cloud ---
def depth_to_point_cloud(depth_map, K):
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    X = x_norm * depth_map
    Y = y_norm * depth_map
    Z = depth_map
    points_3D = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    return points_3D

points_3D = depth_to_point_cloud(depth_map, K)

# --- Step 3: Visualize 2D projection colored by depth ---
def visualize_projection_with_depth(rgb, points_3D, K, step=10):
    H, W, _ = points_3D.shape
    pc_flat = points_3D.reshape(-1, 3).T  # 3 x N
    mask = pc_flat[2] > 0
    valid_pts = pc_flat[:, mask]
    proj = K @ valid_pts
    proj[:2] /= proj[2:]
    u, v, z_vals = proj[0], proj[1], valid_pts[2]

    plt.figure(figsize=(10, 7))
    plt.imshow(rgb)
    plt.scatter(u[::step], v[::step], c=z_vals[::step], s=1, cmap='plasma', alpha=0.7, marker='.', rasterized=True)
    plt.colorbar(label="Depth (m)")
    plt.title("Projected 3D Points Colored by Depth")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#visualize_projection_with_depth(rgb_image, points_3D, K, step=50)

# --- Step 4: Show 3D RGB Point Cloud in Plotly ---
def show_3d_rgb_point_cloud(points_3D, rgb_image, step=10):
    H, W, _ = points_3D.shape
    pc = points_3D[::step, ::step].reshape(-1, 3)      # downsampled (N, 3)
    rgb = rgb_image[::step, ::step].reshape(-1, 3)     # downsampled (N, 3)

    mask = pc[:, 2] > 0  # valid Z
    pc = pc[mask]
    rgb = rgb[mask]

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    rgb_strings = ["rgb({},{},{})".format(r, g, b) for r, g, b in rgb]

    dataf = [go.Scatter3d(
        x=x.tolist(), y=y.tolist(), z=z.tolist(),
        mode="markers",
        marker=dict(size=2, color=rgb_strings, opacity=0.8),
        name="RGB Point Cloud"
    )]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X (m)", nticks=8, range=[-10, 10]),
            yaxis=dict(title="Y (m)", nticks=8, range=[-10, 10]),
            zaxis=dict(title="Z (m)", nticks=8, range=[-10, 10]),
            bgcolor='rgb(30, 30, 30)'
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(family="Courier New, monospace", color='rgb(127, 127, 127)')
    )

    fig = go.Figure(data=dataf, layout=layout)
    fig.show()

show_3d_rgb_point_cloud(points_3D, rgb_image, step=10)

# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# # --- Step 1: Load inputs ---
# depth_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_marigold/frame_0000.npy"
# rgb_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colors/frame_0000.png"
# intrinsics_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_output/intrinsic/frame_0000.txt"

# depth_map = np.load(depth_path)              # (H, W), in meters
# rgb_image = np.array(Image.open(rgb_path))   # (H, W, 3)
# K = np.loadtxt(intrinsics_path)              # 3x3 intrinsic matrix

# # --- Step 2: Convert depth to point cloud ---
# def depth_to_point_cloud(depth_map, K):
#     h, w = depth_map.shape
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     x_norm = (x - cx) / fx
#     y_norm = (y - cy) / fy
#     X = x_norm * depth_map
#     Y = y_norm * depth_map
#     Z = depth_map
#     points_3D = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
#     return points_3D

# points_3D = depth_to_point_cloud(depth_map, K)

# # --- Step 3: Project 3D points to 2D for overlay ---
# def project_points_to_image(points_3D, K):
#     H, W, _ = points_3D.shape
#     points_flat = points_3D.reshape(-1, 3).T
#     mask = points_flat[2] > 0
#     points_valid = points_flat[:, mask]
#     projected = K @ points_valid
#     projected[:2] /= projected[2:]
#     u = projected[0]
#     v = projected[1]
#     z = points_valid[2]
#     return u, v, z

# u, v, z_vals = project_points_to_image(points_3D, K)

# # --- Step 4: Visualize 2D projection colored by depth ---
# def visualize_projection_with_depth(rgb, u, v, z_vals, step=10):
#     plt.figure(figsize=(10, 7))
#     plt.imshow(rgb)
#     u_step = u[::step]
#     v_step = v[::step]
#     z_step = z_vals[::step]
#     scatter = plt.scatter(
#         u_step, v_step, c=z_step, s=1,
#         cmap='plasma', alpha=0.8, marker='.', rasterized=True
#     )
#     plt.colorbar(scatter, label="Depth (m)")
#     plt.title("Projected 3D Points Colored by Depth")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()

# #visualize_projection_with_depth(rgb_image, u, v, z_vals, step=50)

# # --- Step 5: Show 3D scatter colored by RGB ---
# # Flatten and mask
# points_flat = points_3D.reshape(-1, 3)
# rgb_flat = rgb_image.reshape(-1, 3)
# valid_mask = points_flat[:, 2] > 0
# xyz = points_flat[valid_mask]
# rgb = rgb_flat[valid_mask]

# # Downsample
# step = 20
# xyz = xyz[::step]
# rgb = rgb[::step]

# # Convert RGB to 'rgb(r,g,b)'
# rgb_strings = ["rgb({},{},{})".format(r, g, b) for r, g, b in rgb]

# # Plotly scatter
# fig = go.Figure(data=[go.Scatter3d(
#     x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
#     mode='markers',
#     marker=dict(size=2, color=rgb_strings, opacity=0.8)
# )])

# fig.update_layout(
#     scene=dict(
#         xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
#         aspectmode='data', bgcolor='rgb(10,10,10)'
#     ),
#     margin=dict(l=0, r=0, t=0, b=0),
#     paper_bgcolor='black'
# )

# fig.show()
