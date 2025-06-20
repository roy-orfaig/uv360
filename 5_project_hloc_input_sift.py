import os
import numpy as np
from collections import defaultdict
import struct
import matplotlib.pyplot as plt
import cv2
import datetime
from PIL import Image
import plotly.graph_objects as go

class UV360:
    def __init__(self, hloc_folder):
        self.hloc_folder = hloc_folder
        self.images_txt_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/images.txt")
        self.cameras_txt_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/cameras.txt")
        self.point_cloud_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/points3D.bin")
        self.depth_path = os.path.join(hloc_folder,"depth_marigold")
        self.images_path = os.path.join(hloc_folder, "colors")
        self.camera_params = self._read_cameras_txt()
        self.data = self._read_images_txt()
        self.points3D = self.read_points3D_bin()
        self.points3D_RGB = self.read_points3D_bin_rgb()

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

    def project_points_to_image(self, image_id, points3D,rgb_img, flag=True):
        
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
        fig.show()
        fig.write_html(save_file_path, auto_open=True)
        
    
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
            self.save_html(os.path.join(output_folder, file_name_html), point_cloud_3D)
            print(f"Saved: {file_name_html}")
            
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
            
            file_name_html = os.path.splitext(image_data['file_name'])[0] + ".html"
           
            self.save_html(os.path.join(output_folder, file_name_html), point_cloud_3D)
            print(f"Saved: {file_name_html}")
    
    def run_projection_for_all(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, (image_id, image_data) in enumerate(self.data.items()):
            rgb_path = os.path.join(self.images_path, image_data['file_name'])
            if not os.path.exists(rgb_path):
                print(f"Image not found: {rgb_path}")
                continue
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            overlay = self.project_points_to_image(image_id, self.points3D, rgb)
            out_path = os.path.join(output_folder, image_data['file_name'])
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

    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

if __name__ == "__main__":
    hloc_folder = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front"
    # image_folder = os.path.join(hloc_folder, "../colors")
    today_str = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(hloc_folder, "projected_class", today_str)

    uv360 = UV360(hloc_folder)
    print(f"Loaded {len(uv360.data)} images")
    #uv360.run_projection_for_all(output_folder)
    #uv360.run_projection_from_depth_to_all(output_folder)
    uv360.run_registration(output_folder)

