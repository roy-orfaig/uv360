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
image_path = "/home/roy.o@uveye.local/projects/uv360/cad_example/0418a0f0-954f-4830-b685-f3b97591027d/frames_front_01/frame_0005.png"
points = select_points_from_image(image_path)
print("Selected points:", points)