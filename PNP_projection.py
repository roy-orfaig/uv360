import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
# Camera intrinsics (not used here, but can be useful for later back-projection)


selected_points = []
scale_factor = 0.5  # Display image at 50% size

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x_disp, y_disp = event.xdata, event.ydata
        x_orig = int(x_disp / scale_factor)
        y_orig = int(y_disp / scale_factor)
        selected_points.append((x_orig, y_orig))
        print(f"Point selected (original size): ({x_orig}, {y_orig})")
        plt.plot(x_disp, y_disp, 'ro')
        plt.draw()

def select_points_from_image(image_path):
    global selected_points
    selected_points = []

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return []

    # Resize for display only
    display_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(display_image_rgb)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click to select points. Close window when done.")
    plt.show()

    return selected_points

# Example usage
image_path = "/home/roy.o@uveye.local/projects/uv360/frames4_extracted/frame_0005.png"
# points = select_points_from_image(image_path)
# print("Selected points:", points)
points = np.array([
    [ 609,  476],
    [1100,  476],
    [ 642,  675],
    [1281,  634],
    [1155,  920],
    [1697,  823],
    [1260, 1178],
    [1660, 1067]
])

# Intrinsic matrix
K = np.array([
    [1577.0,    0.0, 1032.0],
    [   0.0, 1573.3,  772.0],
    [   0.0,    0.0,    1.0]
], dtype=np.float64)  # important!

# Convert points to float64 explicitly to avoid type mismatch
points_image = np.array([
    [ 608,  473],
    [1094,  468],
    [ 632,  674],
    [1288,  641],
    [1194,  919],
    [1690,  834],
    [1270, 1158],
    [1659, 1051]
], dtype=np.float64)

# points = np.array([
#     [ 610,  469],
#     [1094,  467],
#     [ 634,  673],
#     [1285,  639],
#     [1193,  916],
#     [1683,  830],
#     [1658, 1052],
#     [1272, 1156]
# ])


points_CAD = np.array([
    [-70.661,    -53.862, 179.842],
    [ 70.221611, -53.367, 180.200],
    [-80.819,   -133.810, 139.835],
    [ 77.962059, -128.730, 138.603],
    [-54.956,   -249.470, 117.820],
    [ 56.013645, -250.750, 118.647],
    [-50.914,   -257.360,  76.466743],
    [ 50.252354, -256.600,  76.737900]
], dtype=np.float64)
# Distortion coefficients (assumed zero for now)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

# Run solvePnP
success, rvec, tvec = cv2.solvePnP(points_CAD, points_image, K, dist_coeffs)

# Convert rotation vector to matrix
R, _ = cv2.Rodrigues(rvec)
extrinsic = np.hstack((R, tvec))

# print("Rotation matrix:\n", R)
# print("Translation vector:\n", tvec)
print("Extrinsic matrix:\n", extrinsic)

pcd = o3d.io.read_point_cloud("/home/roy.o@uveye.local/projects/uv360/cad_example/gmc_yukon_pointcloud2.ply")

# Convert to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

# Downsample (optional, for speed)
sampled = points[::10]
sampled_colors = colors[::10] if colors is not None else None

# Create scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=sampled[:, 0],
    y=sampled[:, 1],
    z=sampled[:, 2],
    mode='markers',
    marker=dict(
        size=1,
        color=sampled_colors if sampled_colors is not None else 'gray',
        opacity=0.8
    )
)])

# Set layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title="3D Point Cloud",
    margin=dict(l=0, r=0, b=0, t=30)
)

# Show plot
#fig.show()
fig.write_html("pointcloud_plot.html")

sampled = points.astype(np.float64)  # ensure correct type

# Project using OpenCV
projected_points, _ = cv2.projectPoints(sampled, rvec, tvec, K, dist_coeffs)
projected_points = projected_points.squeeze()  # shape (N, 2)

# Load original image
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found at path:", image_path)

# Overlay projected points
for (u, v) in projected_points:
    u, v = int(round(u)), int(round(v))
    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        cv2.circle(image, (u, v), radius=1, color=(0, 255, 255), thickness=1)  # red dot

# Save the new image
output_path = "/home/roy.o@uveye.local/projects/uv360/projected_overlay_frame_0005.png"
cv2.imwrite(output_path, image)
print(f"Saved image with projected points to:\n{output_path}")