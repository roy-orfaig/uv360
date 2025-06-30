import os
import numpy as np
from collections import defaultdict
import struct
import matplotlib.pyplot as plt
import cv2
import datetime
from PIL import Image
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from glob import glob
import trimesh
import plotly.graph_objs as go
from scipy.spatial import cKDTree
import os
from glob import glob
from scipy.spatial.transform import Rotation as RR

def rotate_around_y(xyz, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    R_y = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad)],
        [ 0,                1, 0               ],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    return (R_y @ xyz.T).T

def extract_yaw_pitch_roll_real_world(R):
    """
    Extract yaw (Z), pitch (Y), and roll (X) in degrees from a 3x3 rotation matrix,
    in real-world frame where:
      X = forward
      Y = left/right
      Z = up

    Rotation order: yaw(Z) → pitch(Y) → roll(X)
    """
    if abs(R[2, 0]) < 1.0:  # avoid gimbal lock
        pitch = np.arcsin(-R[2, 0])
        yaw   = np.arctan2(R[1, 0], R[0, 0])
        roll  = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock
        pitch = np.pi / 2 if R[2, 0] <= -1 else -np.pi / 2
        yaw   = np.arctan2(-R[0, 1], R[1, 1])
        roll  = 0

    return np.degrees([yaw, pitch, roll])

def extract_yaw_pitch_roll_custom(R):
    """
    Extract yaw (around Y), pitch (around X), and roll (around Z)
    for world where Z=depth, Y=up, X=side.
    R is a 3x3 rotation matrix.
    Returns angles in degrees.
    """
    if abs(R[2, 1]) < 0.9999:
        pitch = np.arcsin(-R[2, 1])
        yaw   = np.arctan2(R[0, 1], R[1, 1])
        roll  = np.arctan2(R[2, 0], R[2, 2])
    else:
        # Gimbal lock
        pitch = np.pi/2 if R[2, 1] < 0 else -np.pi/2
        yaw   = 0
        roll  = np.arctan2(-R[1, 0], R[0, 0])

    return np.degrees([yaw, pitch, roll])

def rotate_inverse_yaw(xyz, yaw_degrees):
    angle_rad = np.radians(-yaw_degrees)  # Inverse = negative angle
    R_y = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad)],
        [ 0,                1, 0               ],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    return (R_y @ xyz.T).T

def rotate_inverse_yaw_pitch_roll(xyz, yaw_deg, pitch_deg, roll_deg):
    """
    Rotate the point cloud with inverse yaw, pitch, and roll.
    Rotation is around Y (yaw), X (pitch), then Z (roll).
    """
    # Convert to radians and negate for inverse rotation
    yaw = np.radians(-yaw_deg)
    pitch = np.radians(-pitch_deg)
    roll = np.radians(-roll_deg)

    # Inverse Roll: around Z-axis
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1]
    ])

    # Inverse Pitch: around X-axis
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Inverse Yaw: around Y-axis
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,           1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Combined rotation: R = Rz * Rx * Ry
    R = Rz @ Rx @ Ry

    # Apply rotation to point cloud
    return (R @ xyz.T).T

def convert_extrinsic_camera_to_real_world(extrinsic_cam_to_world):
    """
    Given a 4x4 extrinsic matrix (camera-to-world in camera axes),
    return the equivalent 4x4 extrinsic matrix in real-world convention
    (X=forward, Y=left, Z=up).
    """
    # Rotation to align axes
    R = np.array([
        [0, 0, 1],  # forward (X_real) = Z_cam
        [1, 0, 0],  # left    (Y_real) = X_cam
        [0, 1, 0]   # up      (Z_real) = Y_cam
    ])
    R_align = np.eye(4)
    R_align[:3, :3] = R

    # Final transformation
    extrinsic_real_world = R_align @ extrinsic_cam_to_world
    return extrinsic_real_world

class PreProcessing:
    def __init__(self, input_dir):
        # Store the input directory containing the *_depth.npy files
        self.path_depth_mariglod = input_dir
        
    def remove_underline(self):
        print(f"📁 Looking for files in directory: {self.path_depth_mariglod}")
        
        # Find all .npy files that include '_depth' in their names
        files = glob(os.path.join(self.path_depth_mariglod, "*_depth.npy"))
        print(f"🔍 Found {len(files)} files matching '*_depth.npy'")

        for f in files:
            dir_path = os.path.dirname(f)
            base_name = os.path.basename(f)
            
            # Remove the '_depth' part from the file name
            new_name = base_name.replace("_depth", "")
            new_path = os.path.join(dir_path, new_name)

            print(f"✏️ Renaming: {base_name} → {new_name}")
            os.rename(f, new_path)  # Actually rename the file on disk

        print("✅ Finished renaming files.")

    
class UV360:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.images_txt_path = os.path.join(input_dir, "colomap_scene/images.txt")
        self.cameras_txt_path = os.path.join(input_dir, "colomap_scene/cameras.txt")
        self.point_cloud_path = os.path.join(input_dir, "colomap_scene/points3D.bin")
        self.depth_path = os.path.join(input_dir,"depth_marigold")
        self.images_path = os.path.join(input_dir, "colors")
        self.xyz_rgb_path = os.path.join(input_dir, "xyz_rgb")
        self.camera_params = self._read_cameras_txt()
        self.data = self._read_images_txt()
        self.points3D = self.read_points3D_bin()
        self.points3D_RGB = self.read_points3D_bin_rgb()

    def align_point_cloud_svd_with_axis_swap(self,points):
        """
        Aligns point cloud to its principal axes using SVD, then applies axis remapping:
        X → Z, Y → X, Z → Y.
        
        Args:
            points: (N, 3) ndarray

        Returns:
            aligned_points: (N, 3) point cloud after SVD + axis remapping
            R_total: (3, 3) total rotation matrix
            centroid: (3,) mean of the original points
        """
        # Step 1: center point cloud
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # Step 2: PCA alignment using SVD
        _, _, Vt = np.linalg.svd(points_centered, full_matrices=False)
        R_svd = Vt.T

        # Step 3: additional fixed axis rotation (X→Z, Y→X, Z→Y)
        R_axis_swap = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # Step 4: combine both rotations
        R_total = R_svd @ R_axis_swap

        # Step 5: rotate point cloud
        aligned_points = points_centered @ R_total

        return aligned_points, R_total, centroid
        

    def align_point_cloud_svd(self,points):
        # Compute centroid (translation)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # Apply SVD on the centered point cloud
        _, _, Vt = np.linalg.svd(points_centered, full_matrices=False)
        R = Vt.T  # Rotation matrix (align to principal axes)

        # Rotate the point cloud
        aligned_points = points_centered @ R
        return aligned_points, R, centroid
    
    def align_point_cloud_to_camera(self,points, extrinsic_matrix):
        """
        Align a point cloud to the camera coordinate system using a 4x4 extrinsic matrix.

        Args:
            points: (N, 3) ndarray — 3D points in world coordinates
            extrinsic_matrix: (4, 4) ndarray — world-to-camera transform matrix [R | t; 0 0 0 1]

        Returns:
            aligned_points: (N, 3) ndarray — points in camera coordinates
        """
        # Extract rotation and translation
        R_wc = extrinsic_matrix[:3, :3]
        t_wc = extrinsic_matrix[:3, 3]

        # Compute inverse: camera-to-world
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        # Transform point cloud: world to camera coordinates
        aligned_points = (points @ R_cw.T) + t_cw
        return aligned_points ,R_cw , t_cw
    
    def transform_camera_to_world3D(xyz):
        """
        Transform a point cloud from camera convention (X=lateral, Y=up, Z=forward)
        to real-world 3D (X=forward, Y=left/right, Z=up)
        """
        R = np.array([
            [0, 0, 1],  # X_world = Z_camera
            [1, 0, 0],  # Y_world = X_camera
            [0, 1, 0]   # Z_world = Y_camera
        ])
        return (R @ xyz.T).T
    
    def plot_point_clouds(self,original, aligned, T, output_path):
        
        x=original[:, 0]
        y=original[:, 1]
        z=original[:, 2]
        
        xn=aligned[:, 0] + T[0]
        yn=aligned[:, 1] + T[1]
        zn=aligned[:, 2] + T[2]
        dataf = [
        # local point cloud (colored by z)
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=1, color='blue', colorscale='Viridis'),
            name="Local point cloud"
        ),
        go.Scatter3d(
            x=xn, y=yn, z=zn,
            mode="markers",
            marker=dict(size=1, color='red', opacity=0.8),
            name="Global HLOC"
        ),
        ]

        layout = go.Layout(
            title='Original vs Aligned Point Cloud',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )

    #     fig = go.Figure(data=[trace1, trace2], layout=layout)
    #    # fig.show()
    #     output_path_file = os.path.join(output_path, "aligned_point_cloud.html")
    #     fig.write_html(output_path_file)
        
    #     ###
        
    #     dataf = [
    #         go.Scatter3d(x=xx, y=yy, z=zz, mode="markers", marker=dict(size=1, color=rgb_strings, opacity=0.8), name="HLOC Colored"),
    #     ]


    #     dataf = self.add_camera_poses_to_figure(dataf, cam_poses_world, image_names)

        # lower_bound = [-10, -10, -10]
        # upper_bound =  [10, 10, 10]

        show_grid_lines=True
        # Setup layout
        grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
        layout = go.Layout(scene=dict(
                xaxis=dict(nticks=10, 
                        showbackground=True,
                        backgroundcolor='rgb(30, 30, 30)',
                        gridcolor=grid_lines_color,
                        zerolinecolor=grid_lines_color),
                yaxis=dict(nticks=10,
                        showbackground=True,
                        backgroundcolor='rgb(30, 30, 30)',
                        gridcolor=grid_lines_color,
                        zerolinecolor=grid_lines_color),
                zaxis=dict(nticks=10,
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
            
        fig = go.Figure(data=dataf, layout=layout)
        # html_file = os.path.join(output_folder, "align_pc.html")
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        print(f"Saving HTML to {output_path}")
        fig.write_html(output_path,auto_open=False)
    
    def align_PC_and_save(self,output_path):
        
        
        # aligned, R, T = self.align_point_cloud_svd(self.points3D)       # Align to axes
        # self.plot_point_clouds(self.points3D, aligned, T,output_path) 
        
        
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            extrinsic_matrix = np.array(image_data['extrinsic']).reshape(4, 4)   
    
            #aligned, R, T =self.align_point_cloud_to_camera(self.points3D, extrinsic_matrix)
            aligned, R, T = self.align_point_cloud_svd_with_axis_swap(self.points3D) 
            self.data[image_id]["R_SVD"]=R
            self.data[image_id]["T_SVD"]=T
            # file_name= os.path.splitext(image_data['file_name'])[0] + ".html"
            # output_folder_depth=os.path.join(output_folder,"align")
            # os.makedirs(output_folder_depth, exist_ok=True)
            # output_folder_depth_path=os.path.join(output_folder_depth, file_name)
            # self.plot_point_clouds(self.points3D, aligned, T,output_folder_depth_path) 
            # # self.save_html(output_folder_depth_path, point_cloud_3D)
            # print(f"Saved: {output_folder_depth_path}")
    
    def read_points3D_bin_rgb(self):
        """
        Read a COLMAP points3D.bin file.
        Returns a dict: point3D_id -> {
        'xyz': (x,y,z),
        'rgb': (r,g,b),
        'error': float,
        'track': [(image_id, point2d_idx), ...]
        }
        """
        path=self.point_cloud_path
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
        
    
    def _undistort_points_radial(self, xn, yn, k1):
        r2 = xn ** 2 + yn ** 2
        factor = 1 + k1 * r2
        x_undist = xn / factor
        y_undist = yn / factor
        return x_undist, y_undist

    def create_3D_point_cloud(self, image_id, depth_map):
        if image_id not in self.data:
            raise ValueError(f"Image ID {image_id} not found.")

        cam_params = self.data[image_id]['camera']
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        k1 = cam_params['k']

        h, w = depth_map.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        x_undist, y_undist = self._undistort_points_radial(x_norm, y_norm, k1)

        X = x_undist * depth_map
        Y = y_undist * depth_map
        Z = depth_map
        
        u = x.reshape(-1)
        v = y.reshape(-1)
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)

        # Final array: [u, v, X, Y, Z]
        u_v_X_Y_Z = np.stack((u, v, X, Y, Z), axis=1)

        points_3D_depth = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        return points_3D_depth,u_v_X_Y_Z
    
    def _read_cameras_txt(self):
        camera_params = {}
        with open(self.cameras_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                cam_id = int(tokens[0])
                model = tokens[1]
                width = int(tokens[2])
                height = int(tokens[3])
                params = list(map(float, tokens[4:]))

                if model == 'SIMPLE_RADIAL' and len(params) == 4:
                    fx, cx, cy, k = params
                    fy = fx  # fx == fy in SIMPLE_RADIAL
                    camera_params[cam_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                        'k': k
                    }
        return camera_params
    


    def read_points3D_bin(self):
        path=self.point_cloud_path
        points3D = {}
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<ddd", f.read(24))
                rgb = struct.unpack("<BBB", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                track_length = struct.unpack("<Q", f.read(8))[0]
                f.read(8 * track_length)
                points3D[point_id] = np.array(xyz)
        return np.array(list(points3D.values()))

    def project_points_to_image_distortion(self, image_id, P_world_hom):
        if image_id not in self.data:
            raise ValueError(f"Image ID {image_id} not found.")

        T_world_from_cam = self.data[image_id]['extrinsic']
        cam_params = self.data[image_id]['camera']

        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        k1 = cam_params['k']

        points_cam = (T_world_from_cam @ P_world_hom.T).T
        in_front = points_cam[:, 2] > 0
        points_cam_in_camera = points_cam[in_front]

        x = points_cam_in_camera[:, 0]
        y = points_cam_in_camera[:, 1]
        z = points_cam_in_camera[:, 2]

        x_n = x / z
        y_n = y / z
        r2 = x_n**2 + y_n**2

        distortion = 1 + k1 * r2
        x_d = x_n * distortion
        y_d = y_n * distortion

        u = fx * x_d + cx
        v = fy * y_d + cy
        pixels = np.vstack((u, v)).T
        return pixels, points_cam_in_camera

    def project_points_to_image(self, image_id, points3D,rgb_img, flag=False):
        
        if flag:
            sift_match_pixels,sift_match_3d= self.extarct_sift_matches(image_id)
            sift_match_3d = sift_match_3d.astype(int)

            # Optional: filter out invalid indices (e.g., > max index or < 0)
            valid_indices = sift_match_3d[(sift_match_3d >= 0) & (sift_match_3d < len(points3D))].astype(int)
            points3D = points3D[valid_indices]

            
            
        N = points3D.shape[0]
        points_h = np.hstack((points3D, np.ones((N, 1))))
        
        pixels, points_cam_in_camera = self.project_points_to_image_distortion(image_id, points_h)
        
        h, w, _ = rgb_img.shape
        x_pix = np.round(pixels[:, 0]).astype(int)
        y_pix = np.round(pixels[:, 1]).astype(int)

        valid = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
        x_pix = x_pix[valid]
        y_pix = y_pix[valid]
        
        
        
        depths = points_cam_in_camera[valid, 2]

        norm_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths) + 1e-8)
        colors = (plt.cm.viridis(norm_depths)[:, :3] * 255).astype(np.uint8)

        img_overlay = rgb_img.copy()
        for (x, y, color) in zip(x_pix, y_pix, colors):
            cv2.circle(img_overlay, (x, y), radius=3, color=tuple(int(c) for c in color), thickness=-1)
        
        
        sift_match_pixels, _= self.extarct_sift_matches(image_id)
        for (x, y) in sift_match_pixels:
            cv2.circle(img_overlay, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=1)
        
        return img_overlay
    
    def save_html(self,save_file_path,point_cloud_3D):
        hloc_pts = self.points3D_RGB
        pointssfm = np.array([
            [*pt['xyz'], *pt['rgb']]
            for pt in hloc_pts.values()
        ], dtype=float)
        if True:
            valid_index=self.extract_valid_point3D_ids(self.images_txt_path)
            filtered_valid_ids = [idx for idx in valid_index if idx < len(self.points3D_RGB)]
            X, Y, Z = pointssfm[filtered_valid_ids, 0], pointssfm[filtered_valid_ids, 1], pointssfm[filtered_valid_ids, 2]
            R, G, B = pointssfm[filtered_valid_ids, 3], pointssfm[filtered_valid_ids, 4], pointssfm[filtered_valid_ids, 5]
        else:
            X=pointssfm[:,0].tolist()
            Y=pointssfm[:,1].tolist()
            Z=pointssfm[:,2].tolist()
            R=pointssfm[:,3].tolist()
            G=pointssfm[:,4].tolist()
            B=pointssfm[:,5].tolist()
            


        pc=point_cloud_3D.T
        dataf = []
        x=pc[0].tolist()
        y=pc[1].tolist()
        z=pc[2].tolist()
        x=x[::10]
        y=y[::10]
        z=z[::10]
        prjocted_pts = np.stack([x, y, z], axis=-1)
        
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
            x=X, y=Y, z=Z,
            mode="markers",
            marker=dict(size=1, color=rgb_strings, opacity=0.8),
            name="Global HLOC"
        ),
        # # colored HLOC points
        # go.Scatter3d(
        #     x=xx, y=yy, z=zz,
        #     mode="markers",
        #     marker=dict(size=1, color=rgb_strings, opacity=0.8),
        #     name="HLOC Colored"
        # )
    ]

        rgb_colors = np.stack([R, G, B], axis=1) / 255.0  # normalize to [0, 1]

        # Convert to 'rgb(r,g,b)' strings for Plotly
        rgb_strings = ["rgb({:.0f},{:.0f},{:.0f})".format(r*255, g*255, b*255) for r, g, b in rgb_colors]

        #Add HLOC point cloud
        # dataf.append(go.Scatter3d(
        #     x=xx, y=yy, z=zz,
        #     mode="markers",
        #     marker=dict(size=1, color=rgb_strings, opacity=0.8),
        #     name="HLOC points3D.bin"
        # ))
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
        #fig.show()
        fig.write_html(save_file_path, auto_open=False)
        
    
    def extarct_sift_matches(self, image_id):
        points2d = self.data[image_id]["POINTS2D"]  # shape: (N, 3)
        valid_mask = points2d[:, 2] > 0  # third column is POINT3D_ID

        sift_match_pixels = points2d[valid_mask, :2]  # (x, y)
        sift_match_3D_ids = points2d[valid_mask, 2] 
        return sift_match_pixels, sift_match_3D_ids
    
    def run_projection_from_depth_to_all(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            rgb_path = os.path.join(self.images_path, image_data['file_name'])
            file_name= os.path.splitext(image_data['file_name'])[0] + ".npy"
            depth_image_path = os.path.join(self.depth_path, file_name)
            if not os.path.exists(rgb_path):
                print(f"Image not found: {rgb_path}")
                continue
            depth_map = np.load(depth_image_path)
            file_name_html = os.path.splitext(image_data['file_name'])[0] + ".html"
            point_cloud_3D,_ = self.create_3D_point_cloud(image_id, depth_map)
            output_folder_depth=os.path.join(output_folder,"depth")
            os.makedirs(output_folder_depth, exist_ok=True)
            output_folder_depth_path=os.path.join(output_folder_depth, file_name_html)
            self.save_html(output_folder_depth_path, point_cloud_3D)
            print(f"Saved: {output_folder_depth_path}")
            
    
    def save_depth_map(self, uv_z_mat ,output_path):
        # Linear interpolation
        
        uv_z_mat = uv_z_mat[uv_z_mat[:, 0].argsort()]
        x_vals = uv_z_mat[:, 0]
        y_vals = uv_z_mat[:, 1]
        f_interp = interp1d(x_vals, y_vals, kind='linear')
        x_interp = np.linspace(0, 1, 500)
        mask = (x_interp >= x_vals.min()) & (x_interp <= x_vals.max())
        y_interp = f_interp(x_interp[mask])

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, color='red', label='Original Points')
        plt.plot(x_interp[mask], y_interp, color='blue', label='Linear Interpolation')
        plt.xlim(0, 1)
        plt.ylim(6, 10)
        plt.xlabel('Marigold Depth')
        plt.ylabel('HLOC Depth')
        plt.title('Depth Correspondence: Marigold vs HLOC')
        plt.grid(True)
        plt.legend()

        # Save to PNG
        plt.savefig(output_path, dpi=300)
     
    
    
    def build_3D_point_clud(self,uvxyz_mari, uvxyz_hloc,depth_map_marigold):
        # Step 1: Create dictionary for fast lookup from (u, v) → index
        hloc_dict = {(int(u), int(v)): [x, y, z] for u, v, x, y, z in uvxyz_hloc}
        
        common_uv = []
        src_pts = []
        dst_pts = []
        uv_z_mat = []

        for u, v, x, y, z in uvxyz_mari:
            key = (int(u), int(v))
            if key in hloc_dict:
                common_uv.append([int(u), int(v)])
                src_pts.append([x, y, z])            # from Marigold
                dst_pts.append(hloc_dict[key])       # from HLOC

        if len(src_pts) < 3:
            raise ValueError("Not enough correspondences for affine estimation")

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        common_uv = np.array(common_uv)

        # Extract depth values for all common_uv points
        u_coords = common_uv[:, 0].astype(int)
        v_coords = common_uv[:, 1].astype(int)
        z_depth_marigold = depth_map_marigold[v_coords, u_coords]
        z_depth_hloc=dst_pts[:, 2]
        uv_z_mat = np.stack([z_depth_marigold, z_depth_hloc], axis=1)
        # Step 3: Estimate scale
        # scale_factor =1 # np.median(np.linalg.norm(dst_pts, axis=1) / (np.linalg.norm(src_pts, axis=1) + 1e-8))
        # src_pts_scaled = src_pts * scale_factor

        # affine_matrix = self.estimate_affine(src_pts_scaled, dst_pts)

        # Step 5: Apply transformation to full Marigold
        # all_src_xyz = uvxyz_mari[:, 2:5] * scale_factor
        # all_src_xyz_h = np.hstack([all_src_xyz, np.ones((all_src_xyz.shape[0], 1))])
        # transformed_xyz = (affine_matrix @ all_src_xyz_h.T).T[:, :3]
        # uvxyz_mari_aligned = np.hstack([uvxyz_mari[:, :2], transformed_xyz])
        
        # all_src_xyz = uvxyz_mari[:, 2:5] * scale_factor
        # all_src_xyz_h = np.hstack([all_src_xyz, np.ones((all_src_xyz.shape[0], 1))])
        # transformed_xyz = (affine_matrix @ all_src_xyz_h.T).T[:, :3]
        # uvxyz_mari_aligned = np.hstack([uvxyz_mari[:, :2], transformed_xyz])

        return uv_z_mat,dst_pts,common_uv
    
    def find_affine_transform_from_uv_match(self,uvxyz_mari, uvxyz_hloc):
        # Step 1: Create dictionary for fast lookup from (u, v) → index
        hloc_dict = {(int(u), int(v)): [x, y, z] for u, v, x, y, z in uvxyz_hloc}
        
        # Step 2: Find common (u,v) pairs
        common_uv = []
        src_pts = []
        dst_pts = []

        for u, v, x, y, z in uvxyz_mari:
            key = (int(u), int(v))
            if key in hloc_dict:
                common_uv.append([u, v])
                src_pts.append([x, y, z])            # from Marigold
                dst_pts.append(hloc_dict[key])       # from HLOC

        if len(src_pts) < 3:
            raise ValueError("Not enough correspondences for affine estimation")

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        common_uv = np.array(common_uv)

        # Step 3: Estimate scale
        scale_factor =1 # np.median(np.linalg.norm(dst_pts, axis=1) / (np.linalg.norm(src_pts, axis=1) + 1e-8))
        src_pts_scaled = src_pts * scale_factor

        affine_matrix = self.estimate_affine(src_pts_scaled, dst_pts)

        # Step 5: Apply transformation to full Marigold
        # all_src_xyz = uvxyz_mari[:, 2:5] * scale_factor
        # all_src_xyz_h = np.hstack([all_src_xyz, np.ones((all_src_xyz.shape[0], 1))])
        # transformed_xyz = (affine_matrix @ all_src_xyz_h.T).T[:, :3]
        # uvxyz_mari_aligned = np.hstack([uvxyz_mari[:, :2], transformed_xyz])
        
        all_src_xyz = uvxyz_mari[:, 2:5] * scale_factor
        all_src_xyz_h = np.hstack([all_src_xyz, np.ones((all_src_xyz.shape[0], 1))])
        transformed_xyz = (affine_matrix @ all_src_xyz_h.T).T[:, :3]
        uvxyz_mari_aligned = np.hstack([uvxyz_mari[:, :2], transformed_xyz])

        return affine_matrix, uvxyz_mari_aligned
        
        
    
    def estimate_affine(self,src, dst):
        N = src.shape[0]
        src_h = np.hstack([src, np.ones((N, 1))])  # (N, 4)
        A, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)  # (4, 3)
        affine = np.eye(4)
        affine[:3, :] = A.T
        return affine

    
    
    def run_registration(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            rgb_path = os.path.join(self.images_path, image_data['file_name'])
            file_name= os.path.splitext(image_data['file_name'])[0] + ".npy"
            depth_image_path = os.path.join(self.depth_path, file_name)
            if not os.path.exists(rgb_path):
                print(f"Image not found: {rgb_path}")
                continue
            depth_map = np.load(depth_image_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sift_match_pixels,sift_match_3d= self.extarct_sift_matches(image_id)
            
            sift_match_3d = sift_match_3d.astype(int)

            points3D_dict = self.points3D  # assume dict: {POINT3D_ID: [X, Y, Z]}

            # Create a mask for valid 3D matches (i.e., ID exists in the dict)
            N = len(points3D_dict)
            valid_indices = [
                idx for idx, pid in enumerate(sift_match_3d.astype(int))
                if pid < N
            ]
            valid_pixels = sift_match_pixels[valid_indices]
            valid_ids = sift_match_3d[valid_indices]

            # Lookup 3D points
            valid_points3D = points3D_dict[valid_ids.astype(int)] 

            # Concatenate u,v,X,Y,Z
            u_v_X_Y_Z_HLOC = np.hstack((valid_pixels, valid_points3D))
            
            point_cloud_3D,u_v_X_Y_Z_Marigold = self.create_3D_point_cloud(image_id, depth_map)
            
            affine, uvxyz_mari_aligned = self.find_affine_transform_from_uv_match(u_v_X_Y_Z_Marigold, u_v_X_Y_Z_HLOC)
            
            xyz_mari_aligned = uvxyz_mari_aligned[:, 2:5]
            
            file_name_html = os.path.splitext(image_data['file_name'])[0] + ".html"
            output_folder_registration=os.path.join(output_folder,"Registration")
            os.makedirs(output_folder_registration, exist_ok=True)
            file_name_html_path=os.path.join(output_folder_registration, file_name_html)
            self.save_html(file_name_html_path, xyz_mari_aligned)
            print(f"Saved: {file_name_html_path}")
    
    def run_projection_for_all(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            rgb_path = os.path.join(self.images_path, image_data['file_name'])
            if not os.path.exists(rgb_path):
                print(f"Image not found: {rgb_path}")
                continue
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            overlay = self.project_points_to_image(image_id, self.points3D, rgb)
            output_folder_projection=os.path.join(output_folder,"projection")
            os.makedirs(output_folder_projection, exist_ok=True)
            out_path = os.path.join(output_folder_projection, image_data['file_name'])
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved: {out_path}")

    def _read_images_txt(self):
        data = {}
        with open(self.images_txt_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                i += 1
                continue

            tokens = line.split()
            if len(tokens) < 10:
                i += 1
                continue

            image_id = int(tokens[0])
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            camera_id = int(tokens[8])
            image_name = tokens[9]

            R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([[tx], [ty], [tz]])
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t.flatten()

            i += 1
            if i >= len(lines):
                break

            pts_line = lines[i].strip().split()
            pts = []
            for j in range(0, len(pts_line), 3):
                x = float(pts_line[j])
                y = float(pts_line[j+1])
                pid = int(pts_line[j+2])
                pts.append([x, y, pid])

            data[image_id] = {
                'index': image_id,
                'file_name': image_name,
                'extrinsic': extrinsic,
                'POINTS2D': np.array(pts),
                'camera': self.camera_params.get(camera_id, {})
            }
            i += 1

        return data
    def qvec2rotmat(self,qvec):
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
        ])
    
    def read_camera_poses(self,path):
        image_ids, cam_poses, image_names = [], [], []
        with open(path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            image_name = elems[9]
            R = self.qvec2rotmat(qvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec
            cam_poses.append(np.linalg.inv(T))  # to camera-to-world
            image_ids.append(image_id)
            image_names.append(image_name)
            i += 2  # skip next line
        return image_ids, cam_poses, image_names

    def add_camera_poses_to_figure(self,dataf, cam_poses_world, image_names):
        for i, pose in enumerate(cam_poses_world):
            cam_center = pose[:3, 3]
            cam_forward = -pose[:3, 2]
            end = cam_center + 0.5 * cam_forward
            dataf.append(go.Scatter3d(
                x=[cam_center[0]],
                y=[cam_center[1]],
                z=[cam_center[2]],
                mode="markers+text",
                marker=dict(size=4, color="red"),
                text=[image_names[i]],
                textposition="top center",
                name="Camera center" if i == 0 else None,
                showlegend=(i == 0)
            ))
            dataf.append(go.Scatter3d(
                x=[cam_center[0], end[0]],
                y=[cam_center[1], end[1]],
                z=[cam_center[2], end[2]],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False
            ))
        return dataf
    
    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def extract_valid_point3D_ids(self,images_txt_path):
        point3D_ids = set()

        with open(images_txt_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or len(line) == 0:
                i += 1
                continue

            # Skip first line of image block
            i += 1
            if i >= len(lines):
                break

            # Second line: 2D points with POINT3D_IDs
            tokens = lines[i].strip().split()
            for j in range(2, len(tokens), 3):
                try:
                    pid = int(tokens[j])
                    if pid > -1:
                        point3D_ids.add(pid)
                except ValueError:
                    continue  # In case of corrupted line

            i += 1

        return sorted(point3D_ids)

    
    
    def show_hloc_and_camera_poses(self,output_folder):
        hloc_data = self.points3D_RGB
        pointssfm = np.array([[*pt['xyz'], *pt['rgb']] for pt in hloc_data.values()])
        if True:
            valid_index=self.extract_valid_point3D_ids(self.images_txt_path)
            filtered_valid_ids = [idx for idx in valid_index if idx < len(hloc_data)]
            xx, yy, zz = pointssfm[filtered_valid_ids, 0], pointssfm[filtered_valid_ids, 1], pointssfm[filtered_valid_ids, 2]
            R, G, B = pointssfm[filtered_valid_ids, 3], pointssfm[filtered_valid_ids, 4], pointssfm[filtered_valid_ids, 5]
        else:
            xx, yy, zz = pointssfm[:, 0], pointssfm[:, 1], pointssfm[:, 2]
            R, G, B = pointssfm[:, 3], pointssfm[:, 4], pointssfm[:, 5]  
        rgb_strings = ["rgb({},{},{})".format(int(r), int(g), int(b)) for r, g, b in np.stack([R, G, B], axis=1)]

    
        _,cam_poses_world, image_names = self.read_camera_poses(self.images_txt_path)

        # # ====== Plot Scene ======
        
        # # T_wc: 4x4 camera-to-world matrix for some reference camera
        # T_cw = np.linalg.inv(T_wc)  # Now world → this camera

        # # Homogeneous points: Nx4
        # points_h = np.hstack([points3D, np.ones((points3D.shape[0], 1))])  # shape (N, 4)

        # # Transform all points to camera frame
        # points_camera_frame = (T_cw @ points_h.T).T[:, :3]


        dataf = [
            go.Scatter3d(x=xx, y=yy, z=zz, mode="markers", marker=dict(size=1, color=rgb_strings, opacity=0.8), name="HLOC Colored"),
        ]


        dataf = self.add_camera_poses_to_figure(dataf, cam_poses_world, image_names)

        lower_bound = [-10, -10, -10]
        upper_bound =  [10, 10, 10]

        show_grid_lines=True
        # Setup layout
        grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
        layout = go.Layout(scene=dict(
                xaxis=dict(nticks=10,
                        range=[lower_bound[0], upper_bound[0]],   
                        showbackground=True,
                        backgroundcolor='rgb(30, 30, 30)',
                        gridcolor=grid_lines_color,
                        zerolinecolor=grid_lines_color),
                yaxis=dict(nticks=10,
                        range=[lower_bound[0], upper_bound[0]],
                        showbackground=True,
                        backgroundcolor='rgb(30, 30, 30)',
                        gridcolor=grid_lines_color,
                        zerolinecolor=grid_lines_color),
                zaxis=dict(nticks=10,
                        range=[lower_bound[0], upper_bound[0]],
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
            
        fig = go.Figure(data=dataf, layout=layout)
        html_file = os.path.join(output_folder, "hloc_camera_poses.html")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Saving HTML to {html_file}")
        fig.write_html(html_file,auto_open=False)

    def backproject_pixel_to_3d(self,u, v,K, dist_coeffs,depth_map):
    # Step 1: Undistort the pixel
        distorted_pts = np.array([[[u, v]]], dtype=np.float32)
        undistorted_pts = cv2.undistortPoints(distorted_pts, K, dist_coeffs, P=K)
        u_undist, v_undist = undistorted_pts[0, 0]

        # Step 2: Back-project to 3D (in camera coordinates)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        Z=depth_map(v,u)
        x = (u_undist - cx) * Z / fx
        y = (v_undist - cy) * Z / fy
        return np.array([x, y, Z])
    
    # def backproject_pixels_to_3d_batch(self, u_coords, v_coords, K, dist_coeffs, depth_map, extrinsic=None):
    #     """
    #     Returns:
    #         np.ndarray: shape [N, 3] of 3D points (in camera or world frame depending on extrinsic)
    #     """
    #     # Validate matrices
    #     K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    #     dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    #     # Undistort pixel coordinates
    #     distorted_pts = np.stack((u_coords, v_coords), axis=-1).astype(np.float32).reshape(-1, 1, 2)
    #     undistorted_pts = cv2.undistortPoints(distorted_pts, K, dist_coeffs, P=K).reshape(-1, 2)
    #     u_undist = undistorted_pts[:, 0]
    #     v_undist = undistorted_pts[:, 1]

    #     # Sample depth
    #     u_int = np.clip(np.round(u_coords).astype(int), 0, depth_map.shape[1] - 1)
    #     v_int = np.clip(np.round(v_coords).astype(int), 0, depth_map.shape[0] - 1)
    #     Z = depth_map[v_int, u_int]

    #     # Backproject to camera frame
    #     fx, fy = K[0, 0], K[1, 1]
    #     cx, cy = K[0, 2], K[1, 2]
    #     x = (u_undist - cx) * Z / fx
    #     y = (v_undist - cy) * Z / fy
    #     points_cam = np.stack((x, y, Z), axis=-1)

    #     if extrinsic is not None:
    #         # Convert to homogeneous coordinates
    #         points_hom = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
    #         # Convert to world coordinates using 4x4 matrix
    #         extrinsic = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
    #         points_world = (extrinsic @ points_hom.T).T[:, :3]
    #         return points_world

    #     return points_cam
    
    
    def backproject_pixels_to_3d_batch(self,u_coords, v_coords, K, dist_coeffs, depth_map):
        """
        Vectorized back-projection of multiple pixels (u, v) to 3D camera coordinates.
        
        Parameters:
            u_coords (np.ndarray): array of u (x) coordinates, shape [N]
            v_coords (np.ndarray): array of v (y) coordinates, shape [N]
            K (np.ndarray): camera intrinsic matrix (3x3)
            dist_coeffs (np.ndarray): distortion coefficients
            depth_map (np.ndarray): 2D array of depth values (same size as image)
        
        Returns:
            np.ndarray: array of 3D points [x, y, z], shape [N, 3]
        """
        # Shape (N, 1, 2) for OpenCV undistortPoints
        distorted_pts = np.stack((u_coords, v_coords), axis=-1).astype(np.float32).reshape(-1, 1, 2)
        dist_coeffs = np.array([dist_coeffs[0], 0, 0, 0, 0], dtype=np.float32) 
        # Undistort points
        undistorted_pts = cv2.undistortPoints(distorted_pts, K, dist_coeffs, P=K).reshape(-1, 2)
        u_undist = undistorted_pts[:, 0]
        v_undist = undistorted_pts[:, 1]

        # Sample depth at integer (u, v)
        u_int = np.clip(np.round(u_coords).astype(int), 0, depth_map.shape[1] - 1)
        v_int = np.clip(np.round(v_coords).astype(int), 0, depth_map.shape[0] - 1)
        Z = depth_map[v_int, u_int]

        # Backproject
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u_undist - cx) * Z / fx
        y = (v_undist - cy) * Z / fy

        return np.stack((x, y, Z), axis=-1)

    def save_as_ply_with_color(self,filename, xyz_rgb):
        
        """
        Save Nx6 numpy array to PLY file (ASCII) with RGB color for use in MeshLab.
        Each row in xyz_rgb should be [x, y, z, r, g, b] with RGB in 0–255.
        """
        N = xyz_rgb.shape[0]
        header = f"""ply
        format ascii 1.0
        element vertex {N}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

        with open(filename, 'w') as f:
            f.write(header)
            for row in xyz_rgb:
                x, y, z, r, g, b = row
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    def reconstruct_from_2D_to_3D(self,output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            rgb_path = os.path.join(self.images_path, image_data['file_name'])
            file_name= os.path.splitext(image_data['file_name'])[0] + ".npy"
            depth_image_path = os.path.join(self.depth_path, file_name)
            if not os.path.exists(rgb_path):
                print(f"Image not found: {rgb_path}")
                continue
            depth_map = np.load(depth_image_path)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sift_match_pixels,sift_match_3d= self.extarct_sift_matches(image_id)
            
            sift_match_3d = sift_match_3d.astype(int)
            points3D_dict = self.points3D  # assume dict: {POINT3D_ID: [X, Y, Z]}
            ref_rgb = [236, 237, 239]
            lower_bound = np.clip(ref_rgb , 0, 255)
            upper_bound = np.clip(ref_rgb , 0, 255)
            mask_raw = np.all((rgb >= lower_bound) & (rgb <= upper_bound), axis=2)
            
# Assume mask_raw is a boolean mask
            mask_uint8 = (mask_raw.astype(np.uint8)) * 255

            # Morphological closing
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_closed = cv2.morphologyEx(~mask_uint8, cv2.MORPH_CLOSE, kernel_close)

            # Erosion
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_eroded = cv2.erode(mask_closed, kernel_erode, iterations=1)
            mask_background = (~mask_eroded > 0)          
            
            
            
            # filtered_rgb_pixels = rgb[mask]
            # mask_background = np.all((rgb >= lower_bound) & (rgb <= upper_bound), axis=2)
            
            # Create a mask for valid 3D matches (i.e., ID exists in the dict)
            N = len(points3D_dict)
            valid_indices = [
                idx for idx, pid in enumerate(sift_match_3d.astype(int))
                if pid < N
            ]
            valid_pixels = sift_match_pixels[valid_indices]
            valid_ids = sift_match_3d[valid_indices]

            # Lookup 3D points
            valid_points3D = points3D_dict[valid_ids.astype(int)] 

            # Concatenate u,v,X,Y,Z
            u_v_X_Y_Z_HLOC = np.hstack((valid_pixels, valid_points3D))
            
            point_cloud_3D,u_v_X_Y_Z_Marigold = self.create_3D_point_cloud(image_id, depth_map)
            
            uv_z_mat,XYZ ,common_uv= self.build_3D_point_clud(u_v_X_Y_Z_Marigold, u_v_X_Y_Z_HLOC,depth_map)
            
            extrinzic = image_data['extrinsic']
            K = image_data['camera']['k'] 
            dist_coeffs = np.array([K])
            
            K_dict = image_data['camera']
            K = np.array([[K_dict['fx'], 0, K_dict['cx']],
                        [0, K_dict['fy'], K_dict['cy']],
                        [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([K_dict['k']], dtype=np.float32)  # only k1 for SIMPLE_RADIAL

            
            img_z=np.zeros_like(depth_map, dtype=np.float32)

            minval=np.min(uv_z_mat[:, 1])
            maxval=np.max(uv_z_mat[:, 1])
            img_z[valid_pixels[:, 1].astype(int), valid_pixels[:, 0].astype(int)] = uv_z_mat[:, 1] -np.min(uv_z_mat[:, 1])
            dilated_depth_map = cv2.dilate(img_z, np.ones((3, 3), np.uint8), iterations=1)
            
            R_SVD=self.data[image_id]["R_SVD"]
            T_SVD=self.data[image_id]["T_SVD"]
            
            matched_points=XYZ.copy()
            matched_points_align= (matched_points -T_SVD) @ R_SVD
            
            img_z=depth_map*np.min(matched_points_align[:,2]) +np.min(matched_points_align[:,2]) 
            
            # xyz=[]
            # for uinput in range(50,650):
            #     for vinput in range (40,800):
            #         input_3D=self.backproject_pixel_to_3d(uinput, vinput,K, dist_coeffs,img_z)
            #         xyz.append(input_3D)    
            
            u_range = np.arange(0, img_z.shape[1])
            v_range = np.arange(0, img_z.shape[0])
            uu, vv = np.meshgrid(u_range, v_range)
            uu_filtered = uu[~mask_background]
            vv_filtered = vv[~mask_background]
            rgb_color=rgb[~mask_background]
            u_flat = uu_filtered.ravel()
            v_flat = vv_filtered.ravel()

            # Run vectorized backprojection
            xyz_array = self.backproject_pixels_to_3d_batch(u_flat, v_flat, K, dist_coeffs, img_z)
            
            sift_pixels=self.extarct_sift_matches(image_id)
            sift_pixels_XYZ=self.backproject_pixels_to_3d_batch(sift_pixels[0][:, 0], sift_pixels[0][:, 1], K, dist_coeffs, img_z)
            self.data[image_id]["sift_XYZ"]=sift_pixels_XYZ
            
            rgb_color=rgb[~mask_background] 
            
            if False:
                xyz_rgb=np.concatenate([xyz_array,rgb_color], axis=1)
                file_name_ply = os.path.splitext(image_data['file_name'])[0] + ".ply"
                output_folder_ply=os.path.join(output_folder,"xyz_rgb")
                os.makedirs(output_folder_ply, exist_ok=True)
                file_name_ply_path=os.path.join(output_folder_ply, file_name_ply)
                if not os.path.exists(output_folder_ply):
                    os.makedirs(output_folder)
                self.save_as_ply_with_color(file_name_ply_path, xyz_rgb)
            
            
            if False:
                rgb_normalized = rgb_color / 255.0
                rgb_hex = [mcolors.to_hex(rgb) for rgb in rgb_normalized]
                dataf = [
            # local point cloud (colored by z)
            
                go.Scatter3d(
                    x=xyz_array[::20,0], y=xyz_array[::20,1], z=xyz_array[::20,2],
                    mode="markers",
                    marker=dict(size=1, color=rgb_hex[::20], colorscale='Viridis'),
                    name="Local point cloud"
                ),
                ]

                layout = go.Layout(
                    title='Original vs Aligned Point Cloud',
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
                )


                show_grid_lines=True
                # Setup layout
                grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
                layout = go.Layout(scene=dict(
                        xaxis=dict(nticks=10, 
                                showbackground=True,
                                backgroundcolor='rgb(30, 30, 30)',
                                gridcolor=grid_lines_color,
                                zerolinecolor=grid_lines_color),
                        yaxis=dict(nticks=10,
                                showbackground=True,
                                backgroundcolor='rgb(30, 30, 30)',
                                gridcolor=grid_lines_color,
                                zerolinecolor=grid_lines_color),
                        zaxis=dict(nticks=10,
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
                    
                file_name_html = os.path.splitext(image_data['file_name'])[0] + ".html"
                output_folder_registration=os.path.join(output_folder,"2D_to_3D")
                os.makedirs(output_folder_registration, exist_ok=True)
                file_name_html_path=os.path.join(output_folder_registration, file_name_html)
                if not os.path.exists(output_folder_registration):
                    os.makedirs(output_folder)
                print(f"Saving HTML to {file_name_html_path}")
                fig = go.Figure(data=dataf,layout=layout)
                #fig.show()
                fig.write_html(file_name_html_path,auto_open=False)
            
    def align_pointcloud_svd(self,points):
        # Subtract mean (centering)
        mean = np.mean(points, axis=0)
        centered = points - mean
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        R = Vt.T
        # Rotate the point cloud
        aligned = centered @ R
        return aligned
    
    
        #   centroid = np.mean(points, axis=0)
        # points_centered = points - centroid

        # # Step 2: PCA alignment using SVD
        # _, _, Vt = np.linalg.svd(points_centered, full_matrices=False)
        # R_svd = Vt.T

        # # Step 3: additional fixed axis rotation (X→Z, Y→X, Z→Y)
        # R_axis_swap = np.array([
        #     [0, 0, 1],
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ])

        # # Step 4: combine both rotations
        # R_total = R_svd @ R_axis_swap

        # # Step 5: rotate point cloud
        # aligned_points = points_centered @ R_total
    
    def run_icp_numpy(self, src, tgt, max_iterations=30, tolerance=1e-5):
        # Subsample
        N=1
        src_sampled = src[::N]
        tgt_sampled = tgt[::N]
        
        src_transformed = src_sampled.copy()
        prev_error = float("inf")

        for i in range(max_iterations):
            # Match: find nearest neighbors from src to tgt
            tree = cKDTree(tgt_sampled)
            distances, indices = tree.query(src_transformed)

            matched_tgt = tgt_sampled[indices]

            # Compute centroids
            centroid_src = np.mean(src_transformed, axis=0)
            centroid_tgt = np.mean(matched_tgt, axis=0)

            # Center
            src_centered = src_transformed - centroid_src
            tgt_centered = matched_tgt - centroid_tgt

            # SVD
            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Reflection check
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = centroid_tgt - R @ centroid_src

            # Apply transformation to full resolution (for final return)
            src_transformed = (R @ src_transformed.T).T + t

            # Check convergence
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # Apply final R,t to the **entire original src**, not just the sampled version
        full_aligned = (R @ src.T).T + t

        return full_aligned, R, t
    
    def transform_pc1_to_pc0(self,pc1, extrinsic0, extrinsic1):
        """
        Transforms a point cloud from camera1's frame to camera0's frame using their extrinsics.
        
        Args:
            pc1 (np.ndarray): (N, 3) point cloud in camera 1 coordinates.
            extrinsic0 (np.ndarray): (4, 4) camera 0 extrinsic matrix.
            extrinsic1 (np.ndarray): (4, 4) camera 1 extrinsic matrix.
        
        Returns:
            pc0 (np.ndarray): (N, 3) point cloud in camera 0 coordinates.
            R (np.ndarray): (3, 3) rotation matrix from cam1 to cam0.
            t (np.ndarray): (3,) translation vector from cam1 to cam0.
        """
        # Compute T_1→0 = T0 * inverse(T1)
        T_1_to_0 = extrinsic0 @ np.linalg.inv(extrinsic1)

        # Decompose into R and t
        R = T_1_to_0[:3, :3]
        t = T_1_to_0[:3, 3]

        # Convert pc1 to homogeneous
        pc1_hom = np.hstack([pc1, np.ones((pc1.shape[0], 1))])  # (N, 4)

        # Apply transformation
        pc0_hom = (T_1_to_0 @ pc1_hom.T).T  # (N, 4)
        pc0 = pc0_hom[:, :3]

        return pc0, R, t    
    
    def rotate_inverse_yaw_pitch_roll_real_world(self,xyz, yaw_deg, pitch_deg, roll_deg):
        """
        Apply inverse yaw, pitch, roll rotation to a point cloud in real-world coordinates:
        X = forward, Y = left/right, Z = up
        Rotation order: yaw(Z) → pitch(Y) → roll(X)
        Inverse applied: roll⁻¹(X) → pitch⁻¹(Y) → yaw⁻¹(Z) = Rz @ Ry @ Rx
        """
        # Convert degrees to radians and negate to invert
        yaw   = np.radians(-yaw_deg)   # around Z
        pitch = np.radians(-pitch_deg) # around Y
        roll  = np.radians(-roll_deg)  # around X

        # Inverse Roll: X-axis
        Rx = np.array([
            [1, 0,            0           ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        # Inverse Pitch: Y-axis
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [ 0,             1, 0           ],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Inverse Yaw: Z-axis
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        # Combined inverse rotation: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx

        # Apply to point cloud
        return (R @ xyz.T).T
    
    def projects_pair_3D_svd(self, output_folder):
        
        ply_files = sorted(glob(os.path.join(self.xyz_rgb_path, "*.ply")))

        index_pairs = {}
        for idx, meta in self.data.items():
            file = meta["file_name"]
            base_name = file.replace("_1.png", "").replace(".png", "")
            index_pairs.setdefault(base_name, []).append(idx)

        # Only keep complete pairs
        paired_indexes = [tuple(sorted(v)) for v in index_pairs.values() if len(v) == 2]

        for idx, (i0, i1) in enumerate(paired_indexes):
            
            data0 = self.data[i0]
            data1 = self.data[i1]

            R_world = np.array([
                [0, 0, 1],  # X_real = Z_cam
                [1, 0, 0],  # Y_real = X_cam
                [0, 1, 0]   # Z_real = Y_cam
            ])
        
            # Load .ply files
            name0 = os.path.join(self.xyz_rgb_path, os.path.splitext(data0['file_name'])[0] + ".ply")
            mesh0 = trimesh.load(name0)
            points0 = mesh0.vertices
            R_SVD0=data0["R_SVD"]
            T_SVD0=data0["T_SVD"]
            ex0=data0["extrinsic"]
            ex0_world=convert_extrinsic_camera_to_real_world(ex0)
            xyz0=data0["sift_XYZ"]
            xyz0_world=(R_world @ xyz0.T).T
            
            sift_match_pixels, sift_match_3D_ids=self.extarct_sift_matches(i0)
            #yaw0, pitch0, roll0 = extract_yaw_pitch_roll_real_world(ex0_world)
            yaw0, pitch0, roll0  = extract_yaw_pitch_roll_custom(ex0)
            print("EX0:")
            print("Pitch:", pitch0)
            print("Yaw:", yaw0)
            print("Roll:", roll0)

            name1 = os.path.join(self.xyz_rgb_path, os.path.splitext(data1['file_name'])[0] + ".ply")
            mesh1 = trimesh.load(name1)
            points1 = mesh1.vertices
            R_SVD1=data1["R_SVD"]
            T_SVD1=data1["T_SVD"]
            xyz1=data1["sift_XYZ"]
            xyz1_world=(R_world @ xyz1.T).T
            ex1=data1["extrinsic"]
            # ex1_world=convert_extrinsic_camera_to_real_world(ex1)
            # yaw1, pitch1, roll1 = extract_yaw_pitch_roll_real_world(ex1_world)
            yaw1, pitch1, roll1 = extract_yaw_pitch_roll_custom(ex1)
            print("EX1:")
            print("Pitch:", pitch1)
            print("Yaw:", yaw1)
            print("Roll:", roll1)
             
            aligned0 = xyz0 - xyz0.mean(axis=0)
            aligned1 = xyz1 - xyz1.mean(axis=0)
            
            aligned0_world = xyz0_world - xyz0_world.mean(axis=0)
            aligned1_world = xyz1_world - xyz1_world.mean(axis=0)
            
            
            aligned0_icp, R, t = self.run_icp_numpy(aligned0, aligned1)
            
            points1_align,R,T= self.transform_pc1_to_pc0(aligned1, ex0, ex1)
            r = RR.from_matrix(R)
            pitch, yaw, roll = r.as_euler('xyz', degrees=True)  # or use degrees=False for radians

            yaw_rad = np.deg2rad(-90+yaw)  # convert degrees to radians
            cos_y = np.cos(yaw_rad)
            sin_y = np.sin(yaw_rad)
            
            # Rotation matrix for yaw (rotation about Y axis)
            R_yaw = np.array([
                [ cos_y, 0, sin_y],
                [     0, 1,     0],
                [-sin_y, 0, cos_y]
    ])
            xyz1_cam0=self.transform_pc_from_cam1_to_cam0(xyz1,ex0, ex1)
            print("Pitch:", pitch)
            print("Yaw:", yaw)
            print("Roll:", roll)
            xyz1_flipped = xyz1.copy()
            xyz1_flipped[:, 0] *= -1
            xyz1_flipped_icp, R, t = self.run_icp_numpy(xyz1_flipped, xyz0)
           # aligned0_rot = self.rotate_inverse_yaw_pitch_roll_real_world(aligned0_world, -yaw0,0,0)  
           # aligned1_rot = self.rotate_inverse_yaw_pitch_roll_real_world(aligned1_world, -yaw1,0,0)  
            
            aligned0_rot = rotate_inverse_yaw(aligned0, -yaw0)  
            aligned1_rot = rotate_inverse_yaw(aligned1, -yaw1)  
            # points1_align = (R_yaw @ (aligned1.T)).T
            
            # center = aligned1.mean(axis=0)        # shape (3,)
            # aligned1_centered = aligned1 - center

            # # 2. Apply yaw rotation
            # points1_rotated = (R_yaw @ aligned1_centered.T).T

            # # 3. Move it back to the original location
            # points1_align = points1_rotated + center
                    
            # Align using SVD
            # aligned0 = self.align_pointcloud_svd(points0)
            # aligned1 = self.align_pointcloud_svd(points1)

            N = 1 # Subsample for visualization

            dataf = [
                go.Scatter3d(
                    x=aligned0_rot[::N, 0], y=aligned0_rot[::N, 1], z=aligned0_rot[::N, 2],
                    mode="markers",
                    marker=dict(size=1, color='blue'),
                    name="Aligned PC1"
                ),
                go.Scatter3d(
                    x=aligned1_rot[::N, 0], y=aligned1_rot[::N, 1], z=aligned1_rot[::N, 2],
                    mode="markers",
                    marker=dict(size=1, color='red', opacity=0.8),
                    name="Aligned PC0"
                )
            ]

            layout = go.Layout(
                title='Aligned Point Clouds (SVD)',
                scene=dict(
                    xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')
                ),
                margin=dict(r=10, l=10, b=10, t=10),
                paper_bgcolor='rgb(30, 30, 30)',
                font=dict(family="Courier New, monospace", color='rgb(200, 200, 200)'),
                legend=dict(font=dict(family="Courier New, monospace", color='rgb(200, 200, 200)'))
            )

            fig = go.Figure(data=dataf, layout=layout)

            file_name_html = os.path.splitext(data0['file_name'])[0] + ".html"
            output_folder_registration = os.path.join(output_folder, "2D_to_3D_align_svd")
            os.makedirs(output_folder_registration, exist_ok=True)
            file_name_html_path = os.path.join(output_folder_registration, file_name_html)
            print(f"Saving HTML to {file_name_html_path}")
            fig.write_html(file_name_html_path, auto_open=False)

    # def projects_pair_3D_svd(self,output_folder):
        
    #     ply_files = sorted(glob(os.path.join(self.xyz_rgb_path, "*.ply")))
        
    #     index_pairs = {}
    #     for idx, meta in self.data.items():
    #         file = meta["file_name"]
    #         base_name = file.replace("_1.png", "").replace(".png", "")
    #         index_pairs.setdefault(base_name, []).append(idx)
    
    #    # Only keep complete pairs
    #     paired_indexes = [tuple(sorted(v)) for v in index_pairs.values() if len(v) == 2]
        
    #     for idx, (i0, i1) in enumerate(paired_indexes):
    #         data0 = self.data[i0]
    #         data1 = self.data[i1]
            
    #         name0= os.path.join(self.xyz_rgb_path,os.path.splitext(data0['file_name'])[0] + ".ply")
    #         mesh = trimesh.load(name0) 
    #         points0 = mesh.vertices 
    
            
    #         name1= os.path.join(self.xyz_rgb_path,os.path.splitext(data1['file_name'])[0] + ".ply")
    #         mesh= trimesh.load(name1) 
    #         points1 = mesh.vertices 
    #         N=40
            
    #         dataf = [
    #         # local point cloud (colored by z)
    #         go.Scatter3d(
    #             x=points0[::N,0], y=points0[::N,1], z=points0[::N,2],
    #             mode="markers",
    #             marker=dict(size=1, color='blue', colorscale='Viridis'),
    #             name="Local point cloud"
    #         ),
    #         go.Scatter3d(
    #             x=points1[::N,0], y=points1[::N,1], z=points1[::N,2],
    #             mode="markers",
    #             marker=dict(size=1, color='red', opacity=0.8),
    #             name="Global HLOC"
    #         ),
    #         ]

    #         layout = go.Layout(
    #             title='Original vs Aligned Point Cloud',
    #             scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    #         )

    

    #         show_grid_lines=True
    #         # Setup layout
    #         grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
    #         layout = go.Layout(scene=dict(
    #                 xaxis=dict(nticks=10, 
    #                         showbackground=True,
    #                         backgroundcolor='rgb(30, 30, 30)',
    #                         gridcolor=grid_lines_color,
    #                         zerolinecolor=grid_lines_color),
    #                 yaxis=dict(nticks=10,
    #                         showbackground=True,
    #                         backgroundcolor='rgb(30, 30, 30)',
    #                         gridcolor=grid_lines_color,
    #                         zerolinecolor=grid_lines_color),
    #                 zaxis=dict(nticks=10,
    #                         showbackground=True,
    #                         backgroundcolor='rgb(30, 30, 30)',
    #                         gridcolor=grid_lines_color,
    #                         zerolinecolor=grid_lines_color),
    #                 xaxis_title="x (meters)",
    #                 yaxis_title="y (meters)",
    #                 zaxis_title="z (meters)"
    #             ),
    #             margin=dict(r=10, l=10, b=10, t=10),
    #             paper_bgcolor='rgb(30, 30, 30)',
    #             font=dict(
    #                 family="Courier New, monospace",
    #                 color=grid_lines_color
    #             ),
    #             legend=dict(
    #                 font=dict(
    #                     family="Courier New, monospace",
    #                     color='rgb(127, 127, 127)'
    #                 )
    #             )
    #         )
                
    #         fig = go.Figure(data=dataf, layout=layout)
            
    #         file_name_html = os.path.splitext(data0['file_name'])[0] + ".html"
    #         output_folder_registration=os.path.join(output_folder,"2D_to_3D_align_svd")
    #         os.makedirs(output_folder_registration, exist_ok=True)
    #         file_name_html_path=os.path.join(output_folder_registration, file_name_html)
    #         if not os.path.exists(output_folder_registration):
    #             os.makedirs(output_folder)
    #         print(f"Saving HTML to {file_name_html_path}")
    #         fig = go.Figure(data=dataf,layout=layout)
    #         #fig.show()
    #         fig.write_html(file_name_html_path,auto_open=False)
    
    def transform_pc_from_cam1_to_cam0(self,xyz1, extrinsic0, extrinsic1):
    # Convert to homogeneous
        xyz1_h = np.hstack([xyz1, np.ones((xyz1.shape[0], 1))])  # (N, 4)
        
        # Compute world to cam0 and cam1
        T_world_to_cam0 = extrinsic0
        T_world_to_cam1 = extrinsic1

        # Invert extrinsic to get cam to world
        T_cam0_to_world = np.linalg.inv(T_world_to_cam0)
        T_cam1_to_world = np.linalg.inv(T_world_to_cam1)

        # Compute cam1 to cam0
        T_cam1_to_cam0 = T_world_to_cam0 @ T_cam1_to_world

        # Apply transformation
        xyz1_in_cam0 = (T_cam1_to_cam0 @ xyz1_h.T).T[:, :3]
        return xyz1_in_cam0

    def projects_pair_3D(self,output_folder):
        ply_files = sorted(glob(os.path.join(self.xyz_rgb_path, "*.ply")))
        
        index_pairs = {}
        for idx, meta in self.data.items():
            file = meta["file_name"]
            base_name = file.replace("_1.png", "").replace(".png", "")
            index_pairs.setdefault(base_name, []).append(idx)
    
       # Only keep complete pairs
        paired_indexes = [tuple(sorted(v)) for v in index_pairs.values() if len(v) == 2]
        
        for idx, (i0, i1) in enumerate(paired_indexes):
            data0 = self.data[i0]
            data1 = self.data[i1]
            
            name0= os.path.join(self.xyz_rgb_path,os.path.splitext(data0['file_name'])[0] + ".ply")
            mesh = trimesh.load(name0) 
            points0 = mesh.vertices 
            #colors0 = mesh.visual.vertex_colors[:, :3] 
            extrinsic=data0["extrinsic"]
            extrinsic_cam_to_world = np.linalg.inv(extrinsic)
            points_h = np.hstack([points0, np.ones((points0.shape[0], 1))])  # (N, 4)
            points_world0 = (extrinsic @ points_h.T).T[:, :3] 
            
            name1= os.path.join(self.xyz_rgb_path,os.path.splitext(data1['file_name'])[0] + ".ply")
            #extrinsic=data1["extrinsic"]
            extrinsic_cam_to_world = np.linalg.inv(extrinsic)
            mesh= trimesh.load(name1) 
            points1 = mesh.vertices 
           # colors1 = mesh.visual.vertex_colors[:, :3] 
            points_h = np.hstack([points1, np.ones((points1.shape[0], 1))])  # (N, 4)
            points_world1 = (extrinsic @ points_h.T).T[:, :3] 
            
           
            N=40
            
            dataf = [
            # local point cloud (colored by z)
            go.Scatter3d(
                x=points0[::N,0], y=points0[::N,1], z=points0[::N,2],
                mode="markers",
                marker=dict(size=1, color='blue', colorscale='Viridis'),
                name="Local point cloud"
            ),
            go.Scatter3d(
                x=points1[::N,0], y=points1[::N,1], z=points1[::N,2],
                mode="markers",
                marker=dict(size=1, color='red', opacity=0.8),
                name="Global HLOC"
            ),
            ]
            
            # dataf = [
            # # local point cloud (colored by z)
            # go.Scatter3d(
            #     x=points_world0[::N,0], y=points_world0[::N,1], z=points_world0[::N,2],
            #     mode="markers",
            #     marker=dict(size=1, color='blue', colorscale='Viridis'),
            #     name="Local point cloud"
            # ),
            # go.Scatter3d(
            #     x=points_world1[::N,0], y=points_world1[::N,1], z=points_world1[::N,2],
            #     mode="markers",
            #     marker=dict(size=1, color='red', opacity=0.8),
            #     name="Global HLOC"
            # ),
            # ]

            layout = go.Layout(
                title='Original vs Aligned Point Cloud',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
            )

    

            show_grid_lines=True
            # Setup layout
            grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
            layout = go.Layout(scene=dict(
                    xaxis=dict(nticks=10, 
                            showbackground=True,
                            backgroundcolor='rgb(30, 30, 30)',
                            gridcolor=grid_lines_color,
                            zerolinecolor=grid_lines_color),
                    yaxis=dict(nticks=10,
                            showbackground=True,
                            backgroundcolor='rgb(30, 30, 30)',
                            gridcolor=grid_lines_color,
                            zerolinecolor=grid_lines_color),
                    zaxis=dict(nticks=10,
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
                
            fig = go.Figure(data=dataf, layout=layout)
            
            file_name_html = os.path.splitext(data0['file_name'])[0] + ".html"
            output_folder_registration=os.path.join(output_folder,"2D_to_3D_align")
            os.makedirs(output_folder_registration, exist_ok=True)
            file_name_html_path=os.path.join(output_folder_registration, file_name_html)
            if not os.path.exists(output_folder_registration):
                os.makedirs(output_folder)
            print(f"Saving HTML to {file_name_html_path}")
            fig = go.Figure(data=dataf,layout=layout)
            #fig.show()
            fig.write_html(file_name_html_path,auto_open=False)
            
    
            
            
        # for pair_id, files in pairs.items():
        #     if len(files) != 2:
        #         continue  # skip incomplete pairs
            
        #     pcs_world = []
            #for ply_file in files:
                # Load point cloud
                # pcd = o3d.io.read_point_cloud(ply_file)
                # pts = np.asarray(pcd.points)
                
                # # Determine extrinsic key
                # name = os.path.basename(ply_file).split(".")[0]
                # T = extrinsics.get(name)
                # if T is None:
                #     continue
                
                # # Homogeneous transform
                # pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4
                # pts_w = (T @ pts_h.T).T[:, :3]  # Nx3
                
                # # Save transformed cloud
                # pcd.points = o3d.utility.Vector3dVector(pts_w)
                # pcs_world.append(pcd)
            
            # Merge and save or visualize
            # merged = pcs_world[0] + pcs_world[1]
            # o3d.visualization.draw_geometries([merged])
            
        #os.makedirs(output_folder, exist_ok=True)
        # for i, (image_id, image_data) in enumerate(self.data.items()):
        #     rgb_path = os.path.join(self.images_path, image_data['file_name'])
        #     file_name= os.path.splitext(image_data['file_name'])[0] + ".npy"
        #     depth_image_path = os.path.join(self.depth_path, file_name)
        #     if not os.path.exists(rgb_path):
        #         print(f"Image not found: {rgb_path}")
        #         continue
        #     depth_map = np.load(depth_image_path)
        #     rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        #     sift_match_pixels,sift_match_3d= self.extarct_sift_matches(image_id)       
        # def projects_pair_3D_svd(self,output_folder):
        
        # ply_files = sorted(glob(os.path.join(self.xyz_rgb_path, "*.ply")))
    def align_images(self, output_folder):    
        index_pairs = {}
        for idx, meta in self.data.items():
            file = meta["file_name"]
            base_name = file.replace("_1.png", "").replace(".png", "")
            index_pairs.setdefault(base_name, []).append(idx)

        # Only keep complete pairs
        paired_indexes = [tuple(sorted(v)) for v in index_pairs.values() if len(v) == 2]

        output_folder_projection = os.path.join(output_folder, "sift_align")
        os.makedirs(output_folder_projection, exist_ok=True)

        sift = cv2.SIFT_create()

        for idx, (i0, i1) in enumerate(paired_indexes):
            data0 = self.data[i0]
            data1 = self.data[i1]

            rgb_path_0 = os.path.join(self.images_path, data0['file_name'])
            rgb_path_1 = os.path.join(self.images_path, data1['file_name'])

            if not os.path.exists(rgb_path_0) or not os.path.exists(rgb_path_1):
                print(f"Image not found: {rgb_path_0} or {rgb_path_1}")
                continue

            rgb0 = cv2.imread(rgb_path_0)
            rgb1 = cv2.imread(rgb_path_1)

            # Resize by 0.5
            rgb0_small = cv2.resize(rgb0, (0, 0), fx=0.5, fy=0.5)
            rgb1_small = cv2.resize(rgb1, (0, 0), fx=0.5, fy=0.5)

            # Convert to grayscale for SIFT
            gray0 = cv2.cvtColor(rgb0_small, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(rgb1_small, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors
            kp0, des0 = sift.detectAndCompute(gray0, None)
            kp1, des1 = sift.detectAndCompute(gray1, None)

            # Match features
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des0, des1, k=2)

            # Ratio test as per Lowe's paper
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # Draw matches
            matched_img = cv2.drawMatches(rgb0_small, kp0, rgb1_small, kp1, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            out_path = os.path.join(output_folder_projection, f"match_{i0}_{i1}.png")
            cv2.imwrite(out_path, matched_img)
            print(f"Saved: {out_path}")
                
            
if __name__ == "__main__":
    
    run_preprocessing = False
    run_uv360 = True           
    
    ### pre-processing
    if run_preprocessing:
        npy_path = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front/depth_npy"
        pre_uv360 = PreProcessing(npy_path)
        pre_uv360.remove_underline()
        
    
    ### run
    if run_uv360:
        hloc_folder = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/2D_3D_front"
        # image_folder = os.path.join(hloc_folder, "../colors")
        today_str = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(hloc_folder, "projected_class", today_str)

        uv360 = UV360(hloc_folder)
        print(f"Loaded {len(uv360.data)} images")
        uv360.align_PC_and_save(output_folder)
        # uv360.show_hloc_and_camera_poses(output_folder)
        # uv360.run_projection_for_all(output_folder)
        # uv360.run_projection_from_depth_to_all(output_folder)
        # uv360.run_registration(output_folder)
        uv360.reconstruct_from_2D_to_3D(output_folder)
        #uv360.calc_depth_plot(output_folder)
        uv360.projects_pair_3D_svd(output_folder)
        #uv360.align_images(output_folder)

    # def calc_depth_plot(self,output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    #     for i, (image_id, image_data) in enumerate(self.data.items()):
    #         rgb_path = os.path.join(self.images_path, image_data['file_name'])
    #         file_name= os.path.splitext(image_data['file_name'])[0] + ".npy"
    #         depth_image_path = os.path.join(self.depth_path, file_name)
    #         if not os.path.exists(rgb_path):
    #             print(f"Image not found: {rgb_path}")
    #             continue
    #         depth_map = np.load(depth_image_path)
    #         rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    #         sift_match_pixels,sift_match_3d= self.extarct_sift_matches(image_id)
            
    #         sift_match_3d = sift_match_3d.astype(int)

    #         points3D_dict = self.points3D  # assume dict: {POINT3D_ID: [X, Y, Z]}

    #         # Create a mask for valid 3D matches (i.e., ID exists in the dict)
    #         N = len(points3D_dict)
    #         valid_indices = [
    #             idx for idx, pid in enumerate(sift_match_3d.astype(int))
    #             if pid < N
    #         ]
    #         valid_pixels = sift_match_pixels[valid_indices]
    #         valid_ids = sift_match_3d[valid_indices]

    #         # Lookup 3D points
    #         valid_points3D = points3D_dict[valid_ids.astype(int)] 

    #         # Concatenate u,v,X,Y,Z
    #         u_v_X_Y_Z_HLOC = np.hstack((valid_pixels, valid_points3D))
            
    #         point_cloud_3D,u_v_X_Y_Z_Marigold = self.create_3D_point_cloud(image_id, depth_map)
            
    #         uv_z_mat,XYZ ,common_uv= self.build_3D_point_clud(u_v_X_Y_Z_Marigold, u_v_X_Y_Z_HLOC,depth_map)
            
    #         img_z=np.zeros_like(depth_map, dtype=np.float32)
    #         img_x=np.zeros_like(depth_map, dtype=np.float32)
            
    #         minval=np.min(uv_z_mat[:, 1])
    #         maxval=np.max(uv_z_mat[:, 1])
    #         img_z[valid_pixels[:, 1].astype(int), valid_pixels[:, 0].astype(int)] = uv_z_mat[:, 1] -np.min(uv_z_mat[:, 1])
    #         dilated_depth_map = cv2.dilate(img_z, np.ones((3, 3), np.uint8), iterations=1)

    #         img_x[common_uv[:, 1].astype(int), common_uv[:, 0].astype(int)]=XYZ[:,1]
            
            
    #         vec_x = np.zeros((1, depth_map.shape[1]), dtype=np.float32)
    #         # xyz_mari_aligned = uvxyz_mari_aligned[:, 2:5]
            
    #         H, W = depth_map.shape

    #         # Initialize vec_x (1D) to store one value per column (x-axis)
    #         vec_x = np.zeros(W, dtype=np.float32)

    #         # Accumulator to count how many values go into each x
    #         vec_count = np.zeros(W, dtype=np.int32)

    #         # Fill img_x from your input
    #         img_x = np.zeros_like(depth_map, dtype=np.float32)
    #         x_vals = XYZ[:, 0]
    #         mask = (x_vals >= -1.2) & (x_vals <= 1.2)
    #         mask2 = (common_uv[:,1] >=620) & (common_uv[:,1] <= 690) & (common_uv[:,0] >= 560) & (common_uv[:,0] <= 570)
    #         mask3 = (common_uv[:,1] >=420) & (common_uv[:,1] <= 835) & (common_uv[:,0] >= 90) & (common_uv[:,0] <= 620)
    #         mask4 = (u_v_X_Y_Z_HLOC[:,1] >=420) & (u_v_X_Y_Z_HLOC[:,1] <= 835) & (u_v_X_Y_Z_HLOC[:,0] >= 90) & (u_v_X_Y_Z_HLOC[:,0] <= 620)
    #         depth_map_region=np.zeros_like(depth_map, dtype=np.float32)
    #         depth_map_region[640:690, 560:570]=depth_map[640:690, 560:570]
    #         img_x[valid_pixels[mask, 1].astype(int), valid_pixels[mask, 0].astype(int)] = XYZ[mask, 1]
    #         rgb[valid_pixels[mask, 1].astype(int), valid_pixels[mask, 0].astype(int)] = 255
    #         points= np.stack((XYZ[mask2, 0], XYZ[mask2, 1], XYZ[mask2, 2]), axis=-1)
    #         points2= np.stack((u_v_X_Y_Z_HLOC[mask4, 2], u_v_X_Y_Z_HLOC[mask4, 3], u_v_X_Y_Z_HLOC[mask4, 4]), axis=-1)
    #         R_SVD=self.data[image_id]["R_SVD"]
    #         T_SVD=self.data[image_id]["T_SVD"]
           
            
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.plot(filter_uvxyz[:,0], filter_uvxyz[:,2], c='b', s=5)
    #         # ax.set_xlabel('X')
    #         # ax.set_ylabel('Y')
    #         # ax.set_zlabel('Z')
    #         # ax.set_title('3D Point Cloud from TXT')
            
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)  # 2D plot
    #         ax.scatter(filter_uvxyz[:, 0], filter_uvxyz[:, 3], c='g', s=5)  # blue dots
    #         ax.axis('equal')  # equal scaling for x and y axes
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Z')
    #         ax.set_title('Filtered UVXYZ Projection')
    #         plt.grid(True)
    #         plt.show()
            
    #         ########
             
    #         grill_points=XYZ[mask3, :]
    #         grill_points_align= (grill_points -T_SVD) @ R_SVD
            
            
    #         # normalize depth map
    #         Zgrill_points_image=depth_map*np.min(grill_points_align[:,2]) +np.min(grill_points_align[:,2])   
            
    #         # Z align to axes
    #         Zhloc=grill_points_align[:,2]
            
    #         #   normalize depth map --> normalize point for the selected region
    #         Z_map_marigold=Zgrill_points_image[common_uv[mask3,1],common_uv[mask3,0]]    
            
    #         compare_depth=np.stack((Z_map_marigold, Zhloc), axis=-1) 
    #         diff_abs=compare_depth[:, 0] - compare_depth[:, 1]
    #         # best match
    #         filterz=abs(diff_abs)<0.4
            
    #         common_uv_filter_region=common_uv[mask3,:]
    #         filter_uvxyz = np.concatenate(
    #             [common_uv_filter_region[filterz, :], grill_points_align[filterz, :]],
    #             axis=1)   
    #         newxyz=grill_points_align[filterz, :]
    #         #########
    #         # Create sparse image

    #         # Interpolate missing values
    #         h, w = img_x.shape
    #         grid_y, grid_x = np.mgrid[0:h, 0:w]
    #         valid_mask = img_x != 0

    #         # Get valid points and their X values
    #         points = np.stack((grid_y[valid_mask], grid_x[valid_mask]), axis=-1)
    #         values = img_x[valid_mask]

    #         # Linear interpolation over the full image
    #         img_x_filled = griddata(points, values, (grid_y, grid_x), method='linear')

    #         # Fill any remaining NaNs (outside convex hull) with nearest-neighbor
    #         nan_mask = np.isnan(img_x_filled)
    #         if np.any(nan_mask):
    #             img_x_filled[nan_mask] = griddata(
    #                 points, values, (grid_y[nan_mask], grid_x[nan_mask]), method='linear'
    #             )

    #         # Normalize for 8-bit image saving
    #         img_x_normalized = np.nan_to_num(img_x_normalized, nan=0.0)
    #         img_x_normalized = cv2.normalize(img_x_normalized, None, 0, 255, cv2.NORM_MINMAX)
    #         img_x_uint8 = img_x_normalized.astype(np.uint8)

    #         # Save to PNG
    #         cv2.imwrite('img_x_filled.png', img_x_uint8)
    #         print("Saved: img_x_filled.png")
                        
            
    #         ########

    #         vec_x=img_x_filled[500,:]
            
            
    #         for y in range(H):
    #             for x in range(W):
    #                 val = img_x[y, x]
    #                 if val != 0:
    #                     vec_x[x] += val
    #                     vec_count[x] += 1

    #         # Avoid division by zero
    #         nonzero_mask = vec_count > 0
    #         vec_x[nonzero_mask] /= vec_count[nonzero_mask]
            
    #         from sklearn.linear_model import LinearRegression

    #         # Use only non-zero entries
    #         x_known = np.arange(W)[nonzero_mask].reshape(-1, 1)
    #         y_known = vec_x[nonzero_mask]

    #         # Fit linear regression
    #         model = LinearRegression().fit(x_known, y_known)

    #         # Predict for all indices
    #         x_all = np.arange(W).reshape(-1, 1)
    #         vec_x_filled = model.predict(x_all)

    #         # Fill missing values
    #         vec_x[~nonzero_mask] = vec_x_filled[~nonzero_mask]

    #         plt.figure(figsize=(10, 4))
    #         plt.plot(vec_x, marker='o', linestyle='-', label='vec_x')
    #         plt.xlabel('Pixel Column Index (x)')
    #         plt.ylabel('Value')
    #         plt.title('1D Projection Vector vec_x')
    #         plt.grid(True)
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
            
    #         depth_map_rescaled = depth_map * (maxval - minval) + minval
            
    #         file_name_png = os.path.splitext(image_data['file_name'])[0] + ".png"
    #         output_folder_uv_z_mat=os.path.join(output_folder,"3D_depth")
    #         os.makedirs(output_folder_uv_z_mat, exist_ok=True)
    #         file_name_png_path=os.path.join(output_folder_uv_z_mat, file_name_png)
    #         #self.save_depth_map(uv_z_mat,file_name_png_path)
    #         #Save dilated_depth_map as PNG 
    #         dilated_depth_map = (img_x * 255).astype(np.uint8)
    #         cv2.imwrite(file_name_png_path, rgb)
            
    #         print(f"Saved: {file_name_png_path}")
            
    #         # Step 1: Rescale from physical range to [0, 255] for saving
    #         depth_map_norm = cv2.normalize(depth_map_rescaled, None, 0, 255, cv2.NORM_MINMAX)

    #         # Step 2: Convert to uint8
    #         depth_map_uint8 = depth_map_norm.astype(np.uint8)
    #         output_folder_depth_mat=os.path.join(output_folder,"depth_rescaled")
    #         os.makedirs(output_folder_depth_mat, exist_ok=True)
    #         file_name_png_path=os.path.join(output_folder_depth_mat, file_name_png)
    #         # Step 3: Save as PNG
    #         cv2.imwrite(file_name_png_path, depth_map_uint8)
    