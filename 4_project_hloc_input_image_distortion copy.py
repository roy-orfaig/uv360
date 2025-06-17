import  numpy as np
# The original code that gave expected result
import struct
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


K = np.array([
    [3148.0903, 0.0, 512.0],
    [0.0, 3148.0903, 395.9407],
    [0.0, 0.0, 1.0]
])

T_world_from_cam = np.array([
    [ 0.9443328, -0.002619,  -0.3289811,  0.8779570], #0.95
    [ 0.0239247,  0.9978674,  0.0607312, -0.3114000], #-0.1
    [ 0.3281204, -0.0652213,  0.9423816,  2.2621500],
    [ 0.0,        0.0,        0.0,        1.0]
])




def create_pose_matrix(pitch_deg, yaw_deg, roll_deg, translation=np.array([0.8779570, -0.3114000, 2.2621500])):
    """
    Create a 4x4 camera-to-world transformation matrix using pitch, yaw, roll (in degrees),
    and an optional translation vector.

    Args:
        pitch_deg (float): Rotation around the Y-axis (degrees)
        yaw_deg   (float): Rotation around the Z-axis (degrees)
        roll_deg  (float): Rotation around the X-axis (degrees)
        translation (np.ndarray): 3-element array representing translation [x, y, z]

    Returns:
        np.ndarray: 4x4 transformation matrix (camera-to-world)
    """
    # Create rotation matrix from Euler angles (XYZ = roll, pitch, yaw)
    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
    R_matrix = r.as_matrix()

    # Build the full 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = translation
    return T
# T_world_from_cam = create_pose_matrix(pitch_deg=-18.15, yaw_deg=1.45, roll_deg=-5.96)

# T_world_from_cam=inverse(T_world_from_cam)  # Invert to get world-to-camera transformation
# Roll (X): ≈ -3.96°

# Pitch (Y): ≈ -19.15°

# Yaw (Z): ≈ +1.45°

def read_points3D_bin(path):
    """Read COLMAP binary points3D.bin file."""
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

# Intrinsics and distortion from camera #1


def project_points_to_image_distortion(P_world_hom):
    """
    Projects a 3D homogeneous world point to image coordinates using
    a SIMPLE_RADIAL camera model.

    Args:
        P_world_hom (np.ndarray): 4D homogeneous world point [X, Y, Z, 1]
        T_world_from_cam (np.ndarray): 4x4 transformation matrix

    Returns:
        (u, v): pixel coordinates after projection and distortion
    """
    # Transform point from world to camera coordinates
   # P_cam = T_world_from_cam @ P_world_hom  # shape (4,)
    fx = 3526.23
    fy = 3526.23
    cx = 573.5
    cy = 443.5
    k1 = -0.505595
    points_cam = (T_world_from_cam @ P_world_hom.T).T
    in_front = points_cam[:, 2] > 0
    points_cam_in_camera = points_cam[in_front]
    
    # x, y, z = points_cam_in_camera[:,:3]

    x = points_cam_in_camera[:, 0]
    y = points_cam_in_camera[:, 1]
    z = points_cam_in_camera[:, 2]

    # if z <= 0:
    #     return None  # point behind the camera

    # Normalize
    
    x_n = x / z
    y_n = y / z
    r2 = x_n**2 + y_n**2

    # Apply radial distortion
    distortion = 1 + k1 * r2
    x_d = x_n * distortion
    y_d = y_n * distortion

    # Map to pixel coordinates
    u = fx * x_d + cx
    v = fy * y_d + cy
    pixels = np.vstack((u, v)).T
    return pixels,points_cam_in_camera

def project_point(P_world_hom):
    P_cam = T_world_from_cam @ P_world_hom
    P_img = K @ P_cam[:3]
    u = P_img[0] / P_img[2]
    v = P_img[1] / P_img[2]
    return u, v

def project_points_to_image(points3D, K, world_to_cam, rgb_img):
    N = points3D.shape[0]
    points_h = np.hstack((points3D, np.ones((N, 1))))
    
    # points_cam = (world_to_cam @ points_h.T).T
    # in_front = points_cam[:, 2] > 0
    # points_cam_in_camera = points_cam[in_front]
    
    # pixels_h = (K @ points_cam_in_camera[:, :3].T).T
    # pixels = pixels_h[:, :2] / pixels_h[:, 2:3] 
    pixels,points_cam_in_camera =project_points_to_image_distortion(points_h)
    
    # Assume rgb_img is (H, W, 3) and pixels is (N_visible, 2)
    h, w, _ = rgb_img.shape

    # Round and convert to integer pixel coordinates
    x_pix = np.round(pixels[:, 0]).astype(int)
    y_pix = np.round(pixels[:, 1]).astype(int)

    # Keep only valid pixels within image bounds
    valid = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
    x_pix = x_pix[valid]
    y_pix = y_pix[valid]

    # Also filter the depth values accordingly
    depths = points_cam_in_camera[valid, 2]

    # Normalize depths for colormap visualization
    norm_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths) + 1e-8)
    colors = (plt.cm.viridis(norm_depths)[:, :3] * 255).astype(np.uint8)

    # Overlay the colored points onto the image
    img_overlay = rgb_img.copy()
    for (x, y, color) in zip(x_pix, y_pix, colors):
        cv2.circle(img_overlay, (x, y), radius=3, color=tuple(int(c) for c in color), thickness=-1)

    return img_overlay


point1 = np.array([0.09, 0.0424, 3.93, 1.0])
point2 = np.array([2.14, -0.26, 4.59, 1.0])

u1, v1 = project_point(point1)
u2, v2 = project_point(point2)



print(f"Point 1: x = {u1:.1f}, y = {v1:.1f}")
print(f"Point 2: x = {u2:.1f}, y = {v2:.1f}")



scene_dir = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colomap_whole_scene"
points3D_path = os.path.join(scene_dir, "points3D.bin")
points3D = read_points3D_bin(points3D_path)
output_folder = os.path.join(scene_dir, "projected_hloc_output")
os.makedirs(output_folder, exist_ok=True)
img_path="/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/colors"
fname_path = os.path.join(img_path, "frame_0000.png")
rgb = np.array(Image.open(fname_path))

overlay = project_points_to_image(points3D, K, T_world_from_cam, rgb)
i=0
out_path = os.path.join(output_folder, f"projected_{i:03d}.png")
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")