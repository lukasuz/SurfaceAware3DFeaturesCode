import torch
from tqdm import tqdm
import numpy as np
import potpourri3d as pp3d
import random
import os
import re 
import sys
import mesh2sdf
import pymeshlab
import torch
import numpy as np
import trimesh
from networks import *
from tqdm import tqdm
from ema_pytorch import EMA
from torchvision.utils import save_image
from Diff3D.utils import cosine_similarity
from human_body_prior.body_model.body_model import BodyModel
from visualisation import render, save_video, get_correspondence_colors
from SMALify.smal_model.smal_torch import SMAL, batch_rodrigues

sys.path.append(os.path.join(os.path.dirname(__file__), 'Diff3D'))
from Diff3D.pyFM.mesh import TriMesh
from Diff3D.pyFM.functional import FunctionalMapping # They have a custom implementation
from Diff3D.diff3f import get_features_per_vertex
from Diff3D.utils import convert_mesh_container_to_torch_mesh

sys.path.append(os.path.join('NeuralJacobianFields', 'source_njf'))
from MeshProcessor import MeshProcessor

cosine_sim = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1)

def refine_mesh(mesh: trimesh.Trimesh, target_num=10000, size=128, close=True) -> trimesh.Trimesh:
    if close:
        level = 2 / size
        vertices = (mesh.vertices - np.min(mesh.vertices)) / ((np.max(mesh.vertices) - np.min(mesh.vertices)))
        vertices = (vertices - 0.5) * 1.9 # Bit of a buffer for mesh2sdf

        _, mesh = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    else:
        vertices, faces = mesh.vertices, mesh.faces

    m = pymeshlab.Mesh(vertices, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)
    ms.meshing_isotropic_explicit_remeshing()

    # Decimate mesh, if has more than target_num vertices
    if (ms.current_mesh().vertex_number() > target_num):
        print("Decimating mesh")

        numFaces = 100 + 2 * target_num
        while (ms.current_mesh().vertex_number() > target_num):
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - target_num)

    return trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix())

def get_geometric_desc(vertices, faces):
    source_dir = os.path.join('./jacobians', str(hash(str(vertices) + str(faces))))
    mesh = MeshProcessor(vertices, faces, torch.float32, source_dir=source_dir)
    cn = torch.tensor(mesh.get_centroids()['points_and_normals']).float()
    mesh.computeWKS()
    wks = torch.tensor(mesh.faces_wks).float()

    return torch.concat([cn, wks], dim=-1)

def get_data_jac(model, idx, surface_net=None, feat_type='ours'):
    out = model[idx]
    np_vertices, np_faces = out[0].cpu().numpy(), out[1].cpu().numpy()
    mesh_has = hash(model.obj_paths[idx])
    print(mesh_has, model.obj_paths[idx])
    jacobian_dir = os.path.join('./jacobians', str(mesh_has))
    os.makedirs(jacobian_dir, exist_ok=True)
    JAC = SourceMesh.SourceMesh(0, jacobian_dir, {}, 1, ttype=torch.float)
    JAC.load(source_v=np_vertices, source_f=np_faces)
    JAC.cuda()

    vertices, faces, features, betas = out[0], out[1], out[2], out[5]
    jacobians = JAC.jacobians_from_vertices(vertices[None])
    _vertices = JAC.vertices_from_jacobians(jacobians)
    offset = (vertices - _vertices[0]).mean(dim=[0], keepdims=True)

    if feat_type == 'geo':
        face_features = get_geometric_desc(np_vertices, np_faces).cuda()
    elif feat_type == 'diff3f':
        face_features = features[faces].mean(axis=1)
    elif feat_type == 'ours':
        face_features = surface_net.encode(features[faces].mean(axis=1))[:,0]
    else:
        raise Exception("Wrong feat type")
    
    return vertices, faces, face_features, JAC, jacobians, betas, offset

def save_video_and_frames(frames, folder_path, loop=False, fps=30, step=8, del_frames=True, no_video=False, no_frames=True):
    os.makedirs(folder_path, exist_ok=True)
    if type(frames) is list:
        frames = torch.cat(frames, dim=-1)
    if not no_frames:
        for i, frame in enumerate(frames[::step]):
            save_image(frame[None], os.path.join(folder_path, f'frame_{i}.png'))
    if not no_video:
        save_video(frames, os.path.join(folder_path, f'video.mp4'), fps=fps, loop=loop)
    if del_frames:
        del frames
        torch.cuda.empty_cache()

def compute_features(device, pipe, dino_model, m, prompt, num_views= 100, H=512, W=512, num_images_per_prompt=1, tolerance=0.004, use_normal_map=True, is_tosca=False):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=is_tosca)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )
    return features

def anime_read( filename):
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))

    # flip z and y
    vert_data = np.stack([vert_data[:,0], vert_data[:,2], vert_data[:,1]], axis=-1)
    # same for offsets
    offset_data = np.stack([offset_data[:,:,0], offset_data[:,:,2], offset_data[:,:,1]], axis=-1)

    return nf, nv, nt, vert_data, face_data, offset_data

class SMALMesh():
    def __init__(self) -> None:
        self.num_to_animal = {
            0: 'big cat',
            1: 'dog',
            2: 'horse',
            3: 'cow',
            4: 'hippo'
        }
        self.models = []
        with torch.no_grad():
            for k in self.num_to_animal:
                self.models.append(SMAL('cpu', shape_family_id=k))

        self.rot = torch.tensor([
            [0, 0, 1], 
            [1, 0, 0], 
            [0, 1, 0]], 
            dtype=torch.float32)[None]
    
    @torch.no_grad()
    def get_mesh(self, betas, animal_num, thetas=None):
        assert betas.shape == torch.Size([1, 41])
        model = self.models[animal_num]
        if thetas is None:
            thetas = torch.zeros(1, 35 * 3)
        else:
            assert thetas.shape == torch.Size([1, 35 * 3])
        vertices = model(betas, thetas)[0]
        return torch.bmm(vertices, self.rot)[0], model.faces, model.weights

@torch.no_grad()
def get_point_correspondences(features_source, features_target, inv=False):
    s = cosine_similarity(features_source, features_target)
    return torch.argmax(s, dim=0 if not inv else 1).cpu().numpy()

def fit_to_pcds(args, tgt_pcd, mapping1, mapping2, v_tgt_render, f_tgt_render, betas=None, thetas=None, iters=1000, init_iters=500, 
                user_chamfer=False, arap_w=1, model_type="SMPL", shape_family_id=-1, skip_target=False, cam_pos=None, opt_frames=30,
                skip_render=False, lr=1e-3, fit_betas=False, verbose=False, inv=False, data_src=None):
    cmax = v_tgt_render.max(axis=1, keepdims=True)
    cmin = v_tgt_render.min(axis=1, keepdims=True)

    center = (cmax + cmin) / 2
    v_tgt_render = v_tgt_render - center
    abs_max = np.abs(v_tgt_render).max(axis=1, keepdims=True).max(axis=-1, keepdims=True) 
    v_tgt_render = v_tgt_render / abs_max
    tgt_pcd = (tgt_pcd  - center) / abs_max

    num = len(tgt_pcd)
    if model_type == "SMPL": 
        model = BodyModel(args.smplh_path, 'smplh', num_betas=10, batch_size=num).to('cuda')
        faces = model.faces.cpu().numpy()
        if thetas is None:
            thetas = model.pose_body.clone().cuda()
        else:
            thetas = thetas.clone().cuda()
            thetas = thetas.repeat(num, 1)
        root = model.root_orient.cuda()
        trans = model.trans.cuda()
        v_template = model.v_template[0].cpu().numpy()
        if betas is None:
            betas = model.betas.clone()
        betas = torch.nn.Parameter(betas)
    elif model_type == "SMAL":
        model = SMAL('cuda', shape_family_id=shape_family_id)
        thetas = torch.zeros(num, 35 * 3).cuda()
        root = torch.zeros(num, 3).cuda() # DUMMY
        trans = torch.zeros(num, 3).cuda()
        v_template = model.v_template.cpu().numpy()
        faces = model.faces.cpu().numpy()

        if betas is None:
            betas = torch.zeros(1, 41).cuda()
        betas = torch.nn.Parameter(betas)
    else:
        root = torch.zeros(num, 3).cuda()
        trans = torch.zeros(num, 3).cuda()
        v_template, faces = data_src
        v_template = v_template.cpu().numpy()
        v_template_t = torch.tensor(v_template).cuda()
        faces = faces.cpu().numpy()
        thetas = torch.zeros([0]).cuda()
        betas = torch.zeros([0]).cuda()
        model = None

    tgt_pcd = torch.tensor(tgt_pcd).cuda()
    root = torch.nn.Parameter(root)
    trans = torch.nn.Parameter(trans)
    scale = torch.nn.Parameter(torch.tensor([[1.]]).cuda())
    optimizer = torch.optim.AdamW([root, trans, scale], lr=lr)

    def _pose(root, thetas, betas, trans, scale):
        if model_type == "SMPL":
            betas = betas.expand(num, -1)
            src_pcd = model(root_orient=root, pose_body=thetas, betas=betas, trans=trans).vertices
        elif model_type == "SMAL":
            betas = betas.expand(num, -1)
            src_pcd = model(betas, thetas, trans)[0]
            src_pcd = torch.bmm(src_pcd, batch_rodrigues(root))
        else:
            src_pcd = torch.bmm(torch.tensor(v_template_t[None].expand(num, -1, -1)), batch_rodrigues(root)) + trans[:,None]
        
        src_pcd = torch.clamp(scale, 0.1) * (src_pcd - trans[:,None]) + trans[:,None]
        return src_pcd, center, abs_max

    prev_pcd = 0
    params_over_time = []
    scheduler = None
    for i in tqdm(range(iters + 1)):
        optimizer.zero_grad()
        if i == init_iters:
            thetas = torch.nn.Parameter(thetas)
            params = [root, trans, thetas, scale]
            if fit_betas:
                params.append(betas)
            optimizer = torch.optim.AdamW(params, lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        src_pcd, center, abs_max = _pose(root, thetas, betas, trans, scale)
        # Init ARAP
        if i == 0:
            with torch.no_grad():
                deltas_og = (src_pcd[0,faces[:,None]] - src_pcd[0,faces[:,:,None]]).pow(2).add(1e-8).sqrt().sum(-1)[None]
        
        if user_chamfer:
            D = (tgt_pcd[:,None] - src_pcd[None]).abs().sum(dim=-1)
            nn1 = D.argmin(dim=1)
            nn2 = D.argmin(dim=0)
            nn_distance1 = ((tgt_pcd - src_pcd[nn1])**2).sum(-1)
            nn_distance2 = ((src_pcd - tgt_pcd[nn2])**2).sum(-1)
            point_loss = nn_distance1.mean() + nn_distance2.mean()
        else:
            if inv:
                point_loss = (tgt_pcd[:,mapping2] - src_pcd).abs().mean()
            else:
                point_loss = (tgt_pcd - src_pcd[:,mapping1]).abs().mean()

        # ARAP
        deltas_i = (src_pcd[:,faces[:,None]] - src_pcd[:,faces[:,:,None]]).pow(2).add(1e-8).sqrt().sum(-1)
        arap_loss = arap_w * ((deltas_i - scale * deltas_og).abs().sum() / (deltas_i.shape[1] * 3 * 2 * num))
        if num > 1:
            tv_loss = (src_pcd[1:] - src_pcd[:-1]).pow(2).mean()
        else:
            tv_loss = torch.tensor([0.], device=arap_loss.device)
        loss = point_loss + arap_loss + tv_loss
        
        diff = (src_pcd - prev_pcd).abs().mean().item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scale.clamp_min_(0.01)
            if (num == 1) and (diff > 1e-3):
                prev_pcd = src_pcd.clone().detach()
                params_over_time.append([
                    root.clone().detach(), 
                    trans.clone().detach(), 
                    thetas.clone().detach(), 
                    scale.clone().detach()])
            if i % 100  == 0 and verbose:
                print(i, point_loss.item(), arap_loss.item(), tv_loss.item())
                if scheduler is not None:
                    scheduler.step()

    if inv:
        colors_tgt = get_correspondence_colors(v_tgt_render[0])
        colors_src = colors_tgt[mapping2]
    else:
        colors_src = get_correspondence_colors(v_template)
        colors_tgt = colors_src[mapping1]

    if skip_render:
        return (thetas, betas, root, trans, scale, center, abs_max, model)
    imgs = []
    if num == 1:
        imgs = []
        # Render optimization video
        sampling_indices = np.linspace(0, len(params_over_time) - 1, opt_frames).astype(int)
        with torch.no_grad():
            for i in sampling_indices:
                root, trans, thetas, scale = params_over_time[i]
                src_pcd, _, _ = _pose(root, thetas, betas, trans, scale)
                
                img = render(src_pcd[0].cpu().numpy(), faces, colors_src, cam_pos, num_frames=1)
                imgs.append(img)

        # Add a little break at convergence point
        num_break_frames = int(opt_frames / 10)
        num_rot_frames = 2 * opt_frames 
        imgs = imgs + [imgs[-1]] * num_break_frames

        # Render 360 video after convergence
        imgs_360_src = [img[None] for img in render(src_pcd[0].cpu().numpy(), faces, colors_src, cam_pos, num_frames=num_rot_frames)]

        # Render target image and merge
        img_tgt = render(v_tgt_render[0], f_tgt_render, colors_tgt, cam_pos, num_frames=1, pcd=tgt_pcd[0].cpu().numpy() if not inv else None) # tgt_pcd[0].cpu().numpy()
        imgs_360_tgt = render(v_tgt_render[0], f_tgt_render, colors_tgt, cam_pos, num_frames=num_rot_frames, pcd=tgt_pcd[0].cpu().numpy() if not inv else None) # tgt_pcd[0].cpu().numpy()
        imgs_tgt = ([img_tgt] * len(imgs)) + [img[None] for img in imgs_360_tgt]

        imgs = imgs + imgs_360_src
        imgs = torch.concatenate(imgs)

        fitting_video = torch.concat([
            imgs,
            torch.concatenate(imgs_tgt)
        ], dim=-1)

        return (thetas, betas, root, trans, scale, center, abs_max, model), fitting_video
    else:
        # Render optimization video
        with torch.no_grad():
            src_pcd, _, _ = _pose(root, thetas, betas, trans, scale)

        imgs_src = []
        imgs_tgt = []
        for n in range(num):
            img_src = render(src_pcd[n].cpu().numpy(), faces, colors_src, cam_pos, num_frames=1)
            imgs_src.append(img_src)
            if not skip_target:
                img_tgt = render(v_tgt_render[n], f_tgt_render, colors_tgt, cam_pos, num_frames=1, pcd=tgt_pcd[n].cpu().numpy())
                imgs_tgt.append(img_tgt)

        imgs_src = torch.concatenate(imgs_src)
        if not skip_target:
            imgs_tgt = torch.concatenate(imgs_tgt)
            imgs = torch.concatenate([imgs_src, imgs_tgt], dim=-1)
        else:
            imgs = imgs_src

        return (thetas, betas, root, trans, scale, center, abs_max, model), imgs

def compute_surface_map(v_src, f_src, v_tgt, f_tgt, c1, c2, source_index=None, target_index=None, use_wks=False, device=torch.device("cuda:0")):
    mesh1 = TriMesh(v_src, f_src)
    mesh2 = TriMesh(v_tgt, f_tgt)

    if not use_wks:
        process_params = {
            'n_ev': (50,50),  # Number of eigenvalues on source and Target
            'n_descr': 2048,
            'landmarks': None,
            'descr1': c1,
            'descr2': c2,
            'subsample_step': 0
        }
    else:
        process_params = {
            'n_ev': (50,50),  # Number of eigenvalues on source and Target
            'n_descr': 2048,
            'landmarks': None,
            'subsample_step': 1,  # In order not to use too many descriptors
            'descr_type': 'WKS',  # WKS or HKS
            'subsample_step': 0
            }
    model = FunctionalMapping(mesh1, mesh2)
    model.preprocess(**process_params,verbose=False)
    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }
    model.fit(**fit_params, verbose=False)
    p = model.get_p2p(n_jobs=1)
    if source_index is not None:
        p = p[source_index]
    p = torch.from_numpy(mesh1.vertices[p]).to(device)
    if target_index is not None:
        vertices = torch.from_numpy(mesh1.vertices[target_index]).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=2)[0]
    else:
        vertices = torch.from_numpy(mesh1.vertices).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=1)
    return p

def train_offset_network_jac(JAC_src, J_src, feat, offset_src, vertices_tgt, epochs=100, lr=1e-2, hidden_dim=256, n_hidden=1):
    feat_dim = feat.shape[-1]
    model = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, hidden_dim, bias=True),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.SiLU(),
        *[torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU()) for _ in range(n_hidden)] 
        ,
        torch.nn.Linear(hidden_dim, 9, bias=False),
    ).to(J_src.device)

    t = torch.nn.Parameter(
        torch.randn(1, 3, device=J_src.device) * 1e-6, requires_grad=True
    )
    parameters = list(model.parameters()) + [t]
    optimizer = torch.optim.AdamW(parameters, lr=lr)

    for e in tqdm(range(epochs)):
        optimizer.zero_grad()

        J_pred = model(feat).reshape(feat.shape[0], 3, 3)
        tgt_hat = JAC_src.vertices_from_jacobians(J_src + J_pred) + offset_src + t
        loss = torch.nn.functional.l1_loss(vertices_tgt, tgt_hat)
        
        if e % 250 == 0:
            tqdm.write(f"Epoch {e}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    return model, t

# https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def _init_model_and_optim(model, ema, path, lr, copy_ema=False, verbose=False):
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    if os.path.exists(path):
        ema, optimizer, start_i = load_checkpoint(ema, optimizer, path, copy_ema)
    else:
        if verbose:
            print(f'No checkpoint found at {path}, starting from scratch.')
        start_i = 0
    return model, ema, optimizer, start_i

def get_feature_network(args, path, verbose=True, update_after_step=100, update_every = 10, update_model_with_ema_every = None, copy_ema=True):
    if copy_ema:
        print("Warning: Copying the EMA weights into the online model.")
    update_after_step  = max(1000, update_after_step)
    model = FeatureNetwork(**vars(args)).to(args.device)
    ema = EMA(
        model,
        beta = 0.9999,             
        update_after_step = update_after_step,    
        update_every = update_every,
        update_model_with_ema_every = update_model_with_ema_every,
        # update_model_with_ema_beta=0.9999,
        power=3/4
    )

    if args.FN_path != "":
        return _init_model_and_optim(model, ema, args.FN_path, args.FN_lr, verbose=verbose, copy_ema=copy_ema)
    return _init_model_and_optim(model, ema, path, args.FN_lr, verbose=verbose, copy_ema=copy_ema)

def save_checkpoint(model, optimizer, iter, path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')

def load_checkpoint(model, optimizer, path='checkpoint.pth', copy_ema=False):
    checkpoint = torch.load(path)

    model_dict = checkpoint['model_state_dict']

    if copy_ema:
        om = [e for e in list(model_dict.keys()) if 'online_model' in e]
        ema = [e for e in list(model_dict.keys()) if 'ema' in e]
        assert len(om) == len(ema), "unequal number of online model and ema parameters"

        for key_ema, key_om in zip(ema, om):
            assert '.'.join(key_om.split('.')[1:]) == '.'.join(key_ema.split('.')[1:])
            model_dict[key_om] = model_dict[key_ema] 

    _names = set([e.split('.')[0] for e in list(model_dict.keys())])
    is_legacy = not ('online_model' in _names and not 'ema' in _names)
    if is_legacy:
        print('Warning: Legacy detected, duplicating weights.')
        from collections import OrderedDict
        new_model_dict = OrderedDict()
        for key, value in model_dict.items():
            new_model_dict[f'online_model.{key}'] = value.clone()
            new_model_dict[f'ema.{key}'] = value.clone()

        model_dict = new_model_dict
    
    model.load_state_dict(model_dict, strict=not is_legacy)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter = checkpoint['iter']
    print(f'Checkpoint loaded from {path} (iter {iter})')
    
    return model, optimizer, iter

def get_experiment_folder(args):
    from datetime import datetime
    path = args.exp_path 
    exp_name = args.exp_name 
    if exp_name == '':
        now = datetime.now()
        exp_name = now.strftime("%d_%m_%Y_%H:%M")
    folder_path = os.path.join(path, exp_name)
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def FPS_potpourri(vertices, faces, p, solver=None, rnd=True):
    N, _ = vertices.shape
    num_samples = p
    sampled_indices = np.zeros(num_samples, dtype=int)
    
    if rnd:
        farthest = np.random.randint(0, N, (1,))
    else:
        farthest = np.array([0])
    sampled_indices[0] = farthest

    if solver is None:
        solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    
    min_distances = np.full((N,), np.inf)
    distances = np.zeros((len(vertices), num_samples), dtype=np.float32)

    for i in range(0, num_samples):
        dist = solver.compute_distance(farthest)
        distances[:,i] = dist
        if i < num_samples - 1:
            min_distances = np.minimum(min_distances, dist)
            farthest = np.argmax(min_distances)
            sampled_indices[i+1] = farthest

    return sampled_indices, distances, solver

@torch.no_grad()
def get_data(model, data, shape_idx):
    vertices, faces, F, _, _, betas, thetas = data[shape_idx]
    if thetas is not None:
        thetas = thetas[:,3:] # Drop root rotation for simplicity # TODO: only for SMPL
    f, _ = model(F)

    faces = faces.cpu().numpy()
    vertices = vertices.cpu().numpy()

    return vertices, faces, f, F, betas, thetas

def do_video_fitting(fitting_data, args, model_type, mask=None, fm=False, ours=True, iters=3000, init_iters=1000, skip_target=False, cam_pos=None, fit_betas=False, w_arap=1, inv=False, data_src=None, lr=1e-3):
    v_src, f_src, features_source, diff3d_src, \
    v_tgt, f_tgt, features_target, diff3d_tgt, \
    betas_src, thetas_src, shape_id = fitting_data
    if ours:
        feat_src = features_source
        feat_tgt = features_target
    else:
        feat_src = diff3d_src
        feat_tgt = diff3d_tgt

    if not fm:
        mapping1 = get_point_correspondences(feat_src, feat_tgt[mask], inv=False)
        mapping2 = get_point_correspondences(feat_src, feat_tgt[mask], inv=True)
    else:
        assert mask is None, 'PCDs not supported with functional maps'
        mapping = compute_surface_map(v_src, f_src, v_tgt[0], f_tgt, feat_src.cpu().numpy(), feat_tgt.cpu().numpy())
        mapping = mapping.cpu().numpy()
    
    _, fitting_video = fit_to_pcds(
        args, v_tgt[:,mask], mapping1, mapping2, v_tgt, f_tgt, iters=iters, init_iters=init_iters, user_chamfer=False, 
        betas=betas_src, thetas=thetas_src, model_type=model_type, arap_w=w_arap, skip_target=skip_target, cam_pos=cam_pos, 
        shape_family_id=shape_id, fit_betas=fit_betas, inv=inv, data_src=data_src, lr=lr)
    return fitting_video

def do_video_comparison(args, model, data_src, data_tgt, save_path='./experiments/RENDERS', idx_src=0, idx_tgt=-1, p=0.1, init_iters=500, iters=1500, num_err_imgs=10, cam_pos = np.array([2, 1., -2]), model_type=False, fit_betas=False, w_arap=1, skip_sequence=False, skip_pose=False):
    v_src, f_src, features_source, diff3d_src, betas_src, thetas_src = get_data(model, data_src, idx_src)
    v_tgt, f_tgt, features_target, diff3d_tgt, _, _ = get_data(model, data_tgt, idx_tgt)
    
    if model_type == "SMAL":
        shape_id = data_src.get_smal_shape_family_id(idx_src)
    else:
        shape_id = -1

    mask = np.random.permutation(len(v_tgt))[:int(len(v_tgt) * p)]

    # Single Pose Fitting
    fitting_data = (
        v_src, f_src, features_source, diff3d_src,
        v_tgt[None], f_tgt, features_target, diff3d_tgt,
        betas_src, thetas_src, shape_id
    )
    if not skip_pose:
        pose_video_ours = do_video_fitting(fitting_data, args, model_type, mask, ours=True, iters=iters, init_iters=init_iters, cam_pos=cam_pos, fit_betas=fit_betas, w_arap=w_arap, data_src=data_src)
        pose_video_diff3f = do_video_fitting(fitting_data, args, model_type, mask, ours=False, iters=iters, init_iters=init_iters, cam_pos=cam_pos, fit_betas=fit_betas, w_arap=w_arap, data_src=data_src)
        pose_video = torch.concat([
            pose_video_ours,
            pose_video_diff3f
            
        ], dim=-1)
        
        # shape_family_id
        for i, frame in enumerate(pose_video[-60::4]):
            save_image(frame[None], os.path.join(save_path, f'pose_fitting_{i}_{p}.png'))
        save_video(pose_video, os.path.join(save_path, f'pose_fitting_{p}.mp4'), fps=30, loop=True)

    if model_type != "SMPL" and not skip_sequence:
        tgt_pcd, seq_name = data_tgt.get_animation(idx_tgt)

        mask = np.random.permutation(len(v_tgt))[:int(len(v_tgt) * p)]
        # Render Video sequence
        fitting_data = (
            v_src, f_src, features_source, diff3d_src,
            tgt_pcd, f_tgt, features_target, diff3d_tgt,
            betas_src, thetas_src, shape_id
        )
        sequence_video_ours = do_video_fitting(fitting_data, args, model_type, mask, fm=False, ours=True, iters=iters, init_iters=init_iters, cam_pos=cam_pos, w_arap=w_arap)
        sequence_video_diff3f = do_video_fitting(fitting_data, args, model_type, mask, fm=False, ours=False, iters=iters, init_iters=init_iters, cam_pos=cam_pos, w_arap=w_arap)
        pose_video = torch.concat([
            sequence_video_ours,
            sequence_video_diff3f
        ], dim=-1)

        per_frame_diff = (sequence_video_ours - sequence_video_diff3f).pow(2).mean([1,2,3])
        for i in range(num_err_imgs):
            max_idx = per_frame_diff.argmax()
            per_frame_diff[max_idx] = 0
            save_image(pose_video[max_idx][None], os.path.join(save_path, f'err_img_{i}_{p}.png'))
        save_video(pose_video, os.path.join(save_path, f'sequence_fitting_{p}.mp4'), fps=30, loop=True)

def train_skinning_weights_network(data, weights, encoder, epochs=100, ours=True, gamma=0.8, lr=1e-1, skinning=True):
    fps = data.fps_p
    data.fps_p = 0
    with torch.no_grad():
        _, _, feat, _, _, _, _ = data[0]
    if ours:
        with torch.no_grad():
            feat = encoder.encode(feat)
            feat_dim = feat.shape[-1]
    else:
        feat_dim = 2048

    model = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, weights.shape[-1], bias=False),
        torch.nn.Softmax(dim=-1) if skinning else torch.nn.Sigmoid()
    ).to(weights.device)

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    with torch.no_grad():
        _, _, feat, _, _, _, _ = data[0]
        if ours:
            feat = encoder.encode(feat)[:,0]

    for e in tqdm(range(epochs)):
        optimizer.zero_grad()
        weights_hat = model(feat)
        loss = torch.nn.functional.mse_loss(weights_hat, weights)
        loss.backward()
        optimizer.step()

        scheduler.step()
    data.fps_p = fps
    return model, loss