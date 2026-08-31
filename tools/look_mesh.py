import open3d as o3d

mesh = o3d.io.read_triangle_mesh("/media/user1/Elements/possion_mesh/tsdf_fusion_post.ply")
# 简化到原三角面的20%，按需调0.1‑0.4
mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles)*0.2))
o3d.io.write_triangle_mesh("simplified_mesh.ply", mesh_smp)
