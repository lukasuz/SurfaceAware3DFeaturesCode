import sys
import os
import paths
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *
from dataset import SMALMesh

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

tgt_dir = paths.target_folder_smal_ours
device = 'cuda'
dummy_mode = True # Used for demonstration purposes, set to False otherwise

### SMAL Params
num_betas = 41
var = 0.5
seed = 0
if dummy_mode:
    num_sample_per_animal = 1
    use_rnd_poses = False
else:
    num_sample_per_animal = 10
    use_rnd_poses = False

parameter_path = os.path.join(tgt_dir, 'betas.pt')
os.makedirs(tgt_dir, exist_ok=True)

seed_everything(seed)

betas = torch.randn(5, num_sample_per_animal, num_betas).to('cpu') * var
torch.save(betas, parameter_path)
SM = SMALMesh()
with torch.no_grad():
    for animal in range(len(SM.num_to_animal.keys())):
        for i in range(num_sample_per_animal):
            thetas = 1.5e-1 * torch.randn(1, 35 * 3)
            vertices, faces, weights = SM.get_mesh(betas=betas[None, animal, i], animal_num=animal, thetas=thetas if use_rnd_poses else None)

            os.makedirs(os.path.join(tgt_dir, str(animal)), exist_ok=True)
            mesh_path = os.path.join(tgt_dir, str(animal), f"{i}.obj")

            o3d_faces = o3d.utility.Vector3iVector(faces.cpu().numpy())
            o3d_verts = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
            o3d.io.write_triangle_mesh(mesh_path, mesh)

    pipe = init_pipe(device)
    dino_model = init_dino(device)
    # Generating Diff3D features
    for animal in range(len(SM.num_to_animal.keys())):
        animal_name = SM.num_to_animal[animal]
        for i in range(num_sample_per_animal):
            mesh_path = os.path.join(tgt_dir, str(animal), f"{i}.obj")
            source_mesh = MeshContainer().load_from_file(mesh_path)
            vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=animal_name)
            feat_path = os.path.join(tgt_dir, str(animal), f"features_{i}.pt")
            torch.save(vertex_features, feat_path)
