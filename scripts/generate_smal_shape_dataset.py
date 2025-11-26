import sys
import os
import trimesh
from pathlib import Path
import paths

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

source_dir = paths.source_folder_smal
save_dir = paths.target_folder_smal
device = 'cuda'
seed = 0

parameter_path = os.path.join(source_dir, 'betas.pt')
os.makedirs(source_dir, exist_ok=True)

seed_everything(seed)

with torch.no_grad():
    paths = [str(path) for path in Path(source_dir).rglob("*.ply") if not path.name.startswith('.')]
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    for path in tqdm(paths):
        animal_name = path.split('/')[-2][:-1].replace('_', '')
        identity = path.split('/')[-1].replace('.ply', '')

        os.makedirs(os.path.join(save_dir, identity), exist_ok=True)
        mesh_path = os.path.join(save_dir, identity, f"0.obj")

        mesh = trimesh.load_mesh(path, process=False)
        verts = np.asarray(mesh.vertices) 

        # print('span',identity, np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        verts[:,1] = -1 * verts[:,1] # flip y so visualisation is consistent
        mesh.vertices = verts
        mesh.export(mesh_path)

        # Feature extraction
        feat_path = os.path.join(save_dir, identity, f"features_0.pt")
        if not os.path.exists(feat_path):
            source_mesh = MeshContainer().load_from_file(mesh_path)
            vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=animal_name)
            torch.save(vertex_features, feat_path)
