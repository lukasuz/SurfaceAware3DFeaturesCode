import sys
import os
import open3d as o3d
from tqdm import tqdm
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from poser import Poser
from dataset import AMASS
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

tgt_dir = paths.target_folder_smpl
amass_path = paths.amass_path
device = 'cuda'
### SMPL Params
num_betas = 10
var = 1.5
num_samples = 50
seed = 0

betas_path = os.path.join(tgt_dir, 'betas.pt')
thetas_path = os.path.join(tgt_dir, 'thetas.pt')
amass_indices_path = os.path.join(tgt_dir, 'amass_indices.npy')

os.makedirs(tgt_dir, exist_ok=True)
seed_everything(seed)

# thetas, labels, buffers = self.amass[s]
amass = AMASS(amass_path, seq_len=1)
poser = Poser(batch_size=1, device=device, num_betas=num_betas)
betas = torch.randn(num_samples, num_betas).to(device) * var
amass_indices = np.random.choice(len(amass), num_samples, replace=False)
np.save(amass_indices_path, amass_indices)
torch.save(betas, betas_path)
all_thetas = []
with torch.no_grad():
    # Generating SMPL meshes and videos and jacobians
    for i in range(num_samples):  
        thetas, _, _ = amass[amass_indices[i]]
        all_thetas.append(thetas)
        vertices = poser.pose(thetas[None], betas[i])
        faces = poser.body_model.faces[None]

        os.makedirs(os.path.join(tgt_dir, str(i)), exist_ok=True)
        mesh_path = os.path.join(tgt_dir, str(i), "0.obj")

        o3d_faces = o3d.utility.Vector3iVector(faces[0].cpu().numpy())
        o3d_verts = o3d.utility.Vector3dVector(vertices[0,0].cpu().numpy())
        mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
        o3d.io.write_triangle_mesh(mesh_path, mesh)

    pipe = init_pipe(device)
    dino_model = init_dino(device)
    # Generating Diff3D features
    for i in tqdm(range(num_samples)):
        mesh_path = os.path.join(tgt_dir, str(i), "0.obj")
        source_mesh = MeshContainer().load_from_file(mesh_path)
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt='human')
        feat_path = os.path.join(tgt_dir, str(i), f"features_0.pt")
        torch.save(vertex_features, feat_path)

all_thetas = torch.concat(all_thetas, 0)
torch.save(all_thetas, thetas_path)