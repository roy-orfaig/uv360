import os
import numpy as np
import cv2

input_dir = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_marigold"
output_dir = "/home/roy.o@uveye.local/projects/uv360/uveye_input/1b84c86e-4698-42d2-8974-59700df741d2/front/depth_marigold_png"

os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if not file_name.endswith(".npy"):
        continue

    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace(".npy", ".png"))

    try:
        depth = np.load(input_path)

        if depth.size == 0:
            print(f"Skipped (empty): {file_name}")
            continue

        # Ensure it's float32
        depth = depth.astype(np.float32)

        # Normalize to 0-255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.uint8(depth_norm)

        # Save as PNG
        cv2.imwrite(output_path, depth_uint8)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("Done.")
