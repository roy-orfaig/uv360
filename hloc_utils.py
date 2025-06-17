import os
import numpy as np
from collections import defaultdict
import struct
class UV360:
    def __init__(self, hloc_folder):
        self.hloc_folder = hloc_folder
        self.images_txt_path = os.path.join(hloc_folder, "images.txt")
        self.cameras_txt_path = os.path.join(hloc_folder, "cameras.txt")
        self.point_cloud_path = os.path.join(hloc_folder, "points3D.bin")
        self.camera_params = self._read_cameras_txt()
        self.data = self._read_images_txt()

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
        """Read COLMAP binary points3D.bin file."""
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
    hloc_folder = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colomap_whole_scene_txt"
    uv360 = UV360(hloc_folder)

    print(f"Loaded {len(uv360.data)} images")
    for image_id, image_data in uv360.data.items():
        print(f"{image_id} =\n{image_data}\n")

# import os
# import numpy as np
# from collections import defaultdict

# class UV360:
#     def __init__(self, hloc_folder):
#         self.hloc_folder = hloc_folder
#         self.images_txt_path = os.path.join(hloc_folder, "images.txt")
#         self.images_txt_path = os.path.join(hloc_folder, "cameras.txt")
#         self.data = self._read_images_txt()

#     def _read_images_txt(self):
#         data = {}
#         with open(self.images_txt_path, 'r') as f:
#             lines = f.readlines()

#         i = 0
#         while i < len(lines):
#             line = lines[i].strip()
#             if line.startswith('#') or line == '':
#                 i += 1
#                 continue

#             # First line of image block
#             tokens = line.split()
#             if len(tokens) < 10:
#                 i += 1
#                 continue

#             image_id = int(tokens[0])
#             qw, qx, qy, qz = map(float, tokens[1:5])
#             tx, ty, tz = map(float, tokens[5:8])
#             camera_id = int(tokens[8])
#             image_name = tokens[9]

#             # Convert quaternion + translation to 4x4 extrinsic matrix
#             R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
#             t = np.array([[tx], [ty], [tz]])
#             extrinsic = np.eye(4)
#             extrinsic[:3, :3] = R
#             extrinsic[:3, 3] = t.flatten()

#             # Second line contains 2D keypoints and 3D point IDs
#             i += 1
#             if i >= len(lines):
#                 break

#             pts_line = lines[i].strip().split()
#             pts = []
#             for j in range(0, len(pts_line), 3):
#                 x = float(pts_line[j])
#                 y = float(pts_line[j+1])
#                 pid = int(pts_line[j+2])
#                 pts.append([x, y, pid])

#             data[image_id] = {
#                 'index': image_id,
#                 'file_name': image_name,
#                 'extrinsic': extrinsic,
#                 'POINTS2D': np.array(pts)
#             }
#             i += 1

#         return data

#     def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
#         # Normalize quaternion
#         norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
#         qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

#         R = np.array([
#             [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
#             [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
#             [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
#         ])
#         return R

# if __name__ == "__main__":
#     # Example usage
#     hloc_folder = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colomap_whole_scene_txt"
#     uv360 = UV360(hloc_folder)

#     print(f"Loaded {len(uv360.data)} images")
#     for image_id, image_data in uv360.data.items():
#         print(f"{image_id} =\n{image_data}\n")