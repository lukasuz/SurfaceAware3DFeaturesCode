import sys
import os
import trimesh
import open3d as o3d
from tqdm import tqdm
import shutil
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))

from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer


shrec_path = paths.source_folder_shrec19
tgt_dir = paths.target_folder_shrec19
gt_maps = paths.gt_maps_shrec19
device = 'cuda'
seed = 0

os.makedirs(tgt_dir, exist_ok=True)
seed_everything(seed)
files = os.listdir(shrec_path)

maps = {str(i+1): [] for i in range(len(files))}
for m in os.listdir(gt_maps):
    src, tgt = m.split('_')
    maps[src].append(tgt)

all_thetas = []
with torch.no_grad():
    # Generating SMPL meshes and videos and jacobians
    for file in files:  
        i = file.split('.')[0]
        src_path = os.path.join(shrec_path, file)
        mesh = trimesh.load_mesh(src_path, process=False)

        os.makedirs(os.path.join(tgt_dir, str(i)), exist_ok=True)
        mesh_path = os.path.join(tgt_dir, str(i), "0.obj")

        o3d_faces = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_verts = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
        o3d.io.write_triangle_mesh(mesh_path, mesh)

        for tm in maps[i]:
            shutil.copy(os.path.join(gt_maps, f"{i}_{tm}"), os.path.join(tgt_dir, str(i), tm))

    pipe = init_pipe(device)
    dino_model = init_dino(device)
    # Generating Diff3D features
    for file in tqdm(files):
        i = file.split('.')[0]
        mesh_path = os.path.join(tgt_dir, str(i), "0.obj")

        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.export(mesh_path)

        source_mesh = MeshContainer().load_from_file(mesh_path)
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt='naked human')
        feat_path = os.path.join(tgt_dir, str(i), f"features_0.pt")
        torch.save(vertex_features, feat_path)
