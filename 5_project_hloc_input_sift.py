import os
import numpy as np
from collections import defaultdict
import struct
import matplotlib.pyplot as plt
import cv2
import datetime

class UV360:
    def __init__(self, hloc_folder):
        self.hloc_folder = hloc_folder
        self.images_txt_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/images.txt")
        self.cameras_txt_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/cameras.txt")
        self.point_cloud_path = os.path.join(hloc_folder, "colomap_whole_scene_txt/points3D.bin")
        self.images_path = os.path.join(hloc_folder, "colors")
        self.camera_params = self._read_cameras_txt()
        self.data = self._read_images_txt()
        self.points3D = self.read_points3D_bin()

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
    
    def extarct_sift_matches(self, image_id):
        points2d = self.data[image_id]["POINTS2D"]  # shape: (N, 3)
        valid_mask = points2d[:, 2] > 0  # third column is POINT3D_ID

        sift_match_pixels = points2d[valid_mask, :2]  # (x, y)
        sift_match_3D_ids = points2d[valid_mask, 2] 
        return sift_match_pixels, sift_match_3D_ids
    
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
    uv360.run_projection_for_all(output_folder)

