import sys
import os
import open3d as o3d
from tqdm import tqdm
import trimesh
import torch
import paths
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Polyhaven Dataset")
    parser.add_argument('--animals', action='store_true', help='Generate animal dataset, otherwise chairs')
    args = parser.parse_args()

    max_verts = 5000
    animals = args.animals
    if animals:
        source_path = paths.source_folder_polyhaven_animals
        save_dir = paths.target_folder_polyhaven_animals
    else: # Chairs
        source_path = paths.source_folder_polyhaven_chairs
        save_dir = paths.target_folder_polyhaven_chairs

    os.makedirs(save_dir, exist_ok=True)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


    for item in tqdm(os.listdir(source_path)):
        mesh_path = os.path.join(source_path, item)
        mesh = trimesh.load_mesh(mesh_path, process=False, skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

        folder_name = item.split('.')[0]
        dist_folder = os.path.join(save_dir, folder_name)
        os.makedirs(dist_folder, exist_ok=True)
        
        remeshed = refine_mesh(mesh, max_verts, size=256)
        remeshed.export(os.path.join(dist_folder, '0.obj'))

    with torch.no_grad():
        device = 'cuda'
        pipe = init_pipe(device)
        dino_model = init_dino(device)
        for folder in tqdm(os.listdir(save_dir)):
            tqdm.write(f"Working on {folder}")
            category = folder.split('_')[0]
            mesh_dist_path = os.path.join(save_dir, folder, '0.obj')
            source_mesh = MeshContainer().load_from_file(mesh_dist_path)
            vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=category)
            feat_path = os.path.join(save_dir, folder, "features_0.pt")
            torch.save(vertex_features, feat_path)