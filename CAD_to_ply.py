import open3d as o3d

# === Load the OBJ mesh ===
input_path = "/home/roy.o@uveye.local/projects/uv360/cad_example/GMC_Yukon_(Mk5)_Denali_2021.obj"
output_path = "/home/roy.o@uveye.local/projects/uv360/cad_example/gmc_yukon_pointcloud.ply"

mesh = o3d.io.read_triangle_mesh(input_path)

if not mesh.has_vertices():
    raise ValueError(f"❌ Failed to load mesh: {input_path}")

print(f"✅ Mesh loaded. Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")

# === Compute normals (optional but useful) ===
mesh.compute_vertex_normals()

# === Sample dense point cloud from surface ===
point_cloud = mesh.sample_points_uniformly(number_of_points=100000)

# === Optional: Use vertex colors if available ===
if mesh.has_vertex_colors():
    point_cloud.colors = mesh.vertex_colors
    print("🎨 Vertex colors transferred to point cloud.")
else:
    print("⚠️ No vertex colors found in mesh.")

# === Save to PLY ===
success = o3d.io.write_point_cloud(output_path, point_cloud)

if success:
    print(f"✅ Point cloud saved to {output_path}")
else:
    print("❌ Failed to save point cloud.")


# import open3d as o3d

# # Load the OBJ file (mesh)
# mesh = o3d.io.read_triangle_mesh("/home/roy.o@uveye.local/projects/uv360/cad_example/GMC_Yukon_(Mk5)_Denali_2021.obj")

# # Check if it loaded successfully
# if not mesh.has_vertices():
#     raise ValueError("Failed to load mesh. Check path or file.")

# # Optionally, compute vertex normals (not required for point cloud)
# mesh.compute_vertex_normals()

# # Convert mesh to point cloud using vertices only
# pcd = o3d.geometry.PointCloud()
# pcd.points = mesh.vertices

# # Optional: use vertex colors if available
# if mesh.has_vertex_colors():
#     pcd.colors = mesh.vertex_colors

# # Save point cloud as PLY
# o3d.io.write_point_cloud("/home/roy.o@uveye.local/projects/uv360/cad_example/gmc_yukon_pointcloud.ply", pcd)

# print("Saved point cloud to gmc_yukon_pointcloud.ply")
