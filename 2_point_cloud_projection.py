import numpy as np
from PIL import Image
import struct
# import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pypcd
import numpy as np
def read_points3d_binary(path):
    """Parses COLMAP points3D.bin into an (N,3) array of XYZ."""
    pts = []
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            _id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            _r, _g, _b = struct.unpack("<BBB", f.read(3))
            _err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(16 * track_len)  # skip track entries
            pts.append((x, y, z))
    return np.array(pts)

def read_points3d_bin(path):
    """
    Read a COLMAP points3D.bin file.
    Returns a dict: point3D_id -> {
      'xyz': (x,y,z),
      'rgb': (r,g,b),
      'error': float,
      'track': [(image_id, point2d_idx), ...]
    }
    """
    points = {}
    with open(path, "rb") as f:
        # number of points
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id = struct.unpack("<I", f.read(4))[0]
                pt2d_idx = struct.unpack("<I", f.read(4))[0]
                track.append((img_id, pt2d_idx))
            points[pid] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': track
            }
    return points

def save_point_cloud_to_bin(point_cloud_data, filename):
    x = point_cloud_data[:,0]
    y = point_cloud_data[:,1]
    z = point_cloud_data[:,2]
    arr = np.zeros(x.shape[0] * 3, dtype=np.float32)
    arr[::3] = x
    arr[1::3] = y
    arr[2::3] = z
    arr.astype('float32').tofile(filename)

# Example usage (assuming point_cloud_data is a pypcd.PointCloud object)
# save_point_cloud_to_bin(point_cloud_data, "my_point_cloud.bin")


# --- 1) Load inputs ---
depth = np.load("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_marigold/frame_0000.npy")           # (H, W) in meters
rgb   = np.array(Image.open("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colors/frame_0000.png"))  # (H, W, 3), uint8

# Intrinsics in a plain‐text 3×3 (whitespace‐separated)
K = np.loadtxt("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_output/intrinsic/frame_0000.txt")  
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

H, W = depth.shape

# --- 2) Reproject depth → camera coords ---
u = np.arange(W)
v = np.arange(H)
uu, vv = np.meshgrid(u, v)

Z = depth
X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy

# Flatten
local_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
# Normalize RGB to [0,1]
local_cols = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)

# --- 3) Read hloc/COLMAP binary point cloud ---

hloc_pts = read_points3d_bin("/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colomap_whole_scene/points3D.bin")
pointssfm = np.array([
    [*pt['xyz'], *pt['rgb']]
    for pt in hloc_pts.values()
], dtype=float)

xx=pointssfm[:,0].tolist()
yy=pointssfm[:,1].tolist()
zz=pointssfm[:,2].tolist()
R=pointssfm[:,3].tolist()
G=pointssfm[:,4].tolist()
B=pointssfm[:,5].tolist()



# --- 4) Plot with Plotly ---
x=local_pts[0].tolist()
y=local_pts[1].tolist()
z=local_pts[2].tolist()
x=x[0::10]
y=y[0::10]
z=z[0::10]


pc=local_pts.T
dataf = []
x=pc[0].tolist()
y=pc[1].tolist()
z=pc[2].tolist()
x=x[0::10]
y=y[0::10]
z=z[0::10]
prjocted_pts = np.stack([x, y, z], axis=-1)
hloc_pts = np.stack([xx, yy, zz], axis=-1)

if True:
    pc=local_pts.T
    dataf = []
    x=pc[0].tolist()
    y=pc[1].tolist()
    z=pc[2].tolist()
    x=x[::10]
    y=y[::10]
    z=z[::10]
    prjocted_pts = np.stack([x, y, z], axis=-1)
    hloc_pts = np.stack([xx, yy, zz], axis=-1)
    point_size = np.full(len(x), 2)
    # fig = px.scatter_3d(x=x ,y=y, z=z, size=point_size, size_max=1,opacity=1)
    # dataf = [go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=1,color=z,colorscale='Viridis'))]
    # #the global hloc reconstruction
    #     go.Scatter3d(
    #        x=hloc_pts[:, 0], y=hloc_pts[:, 1], z=hloc_pts[:, 2],
    #        mode="markers",
    #        marker=dict(size=1, color="lightgray", opacity=0.6),
    #        name="hloc points3D.bin"
    #  )
    rgb_colors = np.stack([R, G, B], axis=1)
    rgb_strings = ["rgb({},{},{})".format(int(r), int(g), int(b)) for r, g, b in rgb_colors]
    dataf = [
    # local point cloud (colored by z)
    go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=1, color=z, colorscale='Viridis'),
        name="Local point cloud"
    ),
    # global hloc point cloud (gray)
    go.Scatter3d(
        x=hloc_pts[:, 0], y=hloc_pts[:, 1], z=hloc_pts[:, 2],
        mode="markers",
        marker=dict(size=1, color="lightgray", opacity=0.6),
        name="Global HLOC"
    ),
    # colored HLOC points
    go.Scatter3d(
        x=xx, y=yy, z=zz,
        mode="markers",
        marker=dict(size=1, color=rgb_strings, opacity=0.8),
        name="HLOC Colored"
    )
]

    rgb_colors = np.stack([R, G, B], axis=1) / 255.0  # normalize to [0, 1]

    # Convert to 'rgb(r,g,b)' strings for Plotly
    rgb_strings = ["rgb({:.0f},{:.0f},{:.0f})".format(r*255, g*255, b*255) for r, g, b in rgb_colors]

    #Add HLOC point cloud
    dataf.append(go.Scatter3d(
        x=xx, y=yy, z=zz,
        mode="markers",
        marker=dict(size=1, color=rgb_strings, opacity=0.8),
        name="HLOC points3D.bin"
    ))
    mega_centroid = np.average(pc, axis=1)
    lower_bound = [-10, -10, -10]
    upper_bound =  [10, 10, 10]

    show_grid_lines=True
    # Setup layout
    grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
    layout = go.Layout(scene=dict(
            xaxis=dict(nticks=8,
                    range=[lower_bound[0], upper_bound[0]],
                    showbackground=True,
                    backgroundcolor='rgb(30, 30, 30)',
                    gridcolor=grid_lines_color,
                    zerolinecolor=grid_lines_color),
            yaxis=dict(nticks=8,
                    range=[lower_bound[1], upper_bound[1]],
                    showbackground=True,
                    backgroundcolor='rgb(30, 30, 30)',
                    gridcolor=grid_lines_color,
                    zerolinecolor=grid_lines_color),
            zaxis=dict(nticks=8,
                    range=[lower_bound[2], upper_bound[2]],
                    showbackground=True,
                    backgroundcolor='rgb(30, 30, 30)',
                    gridcolor=grid_lines_color,
                    zerolinecolor=grid_lines_color),
            xaxis_title="x (meters)",
            yaxis_title="y (meters)",
            zaxis_title="z (meters)"
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(
            family="Courier New, monospace",
            color=grid_lines_color
        ),
        legend=dict(
            font=dict(
                family="Courier New, monospace",
                color='rgb(127, 127, 127)'
            )
        )
    )

    fig = go.Figure(data=dataf,layout=layout)
    fig.show()