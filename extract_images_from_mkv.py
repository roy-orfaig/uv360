import subprocess
import os

# === Config ===
video_path = "/home/roy.o@uveye.local/Downloads/frames4.mkv"
output_dir = "/home/roy.o@uveye.local/Downloads/frames4_extracted"

# video_path = "/home/roy.o@uveye.local/projects/uv360/cad_example/0418a0f0-954f-4830-b685-f3b97591027d/frames_front_00.mkv"
# output_dir = "/home/roy.o@uveye.local/projects/uv360/cad_example/0418a0f0-954f-4830-b685-f3b97591027d/frames_front_00"


video_path = "/home/roy.o@uveye.local/projects/uv360/cad_example/0418a0f0-954f-4830-b685-f3b97591027d/frames_front_01.mkv"
output_dir = "/home/roy.o@uveye.local/projects/uv360/cad_example/0418a0f0-954f-4830-b685-f3b97591027d/frames_front_01"

frame_rate = 5  # frames per second

# === Create output folder ===
os.makedirs(output_dir, exist_ok=True)

# === FFmpeg command ===
output_pattern = os.path.join(output_dir, "frame_%04d.png")

command = [
    "ffmpeg",
    "-i", video_path,
    "-vf", f"fps={frame_rate}",
    output_pattern
]

# === Run FFmpeg ===
subprocess.run(command)

print(f"✅ Frames extracted to: {output_dir}")
