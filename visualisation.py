import numpy as np
import torch
from seaborn import color_palette
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import imageio
import open3d as o3d

@torch.no_grad()
def get_data(model, data, shape_idx):
    vertices, faces, F, _, _, betas, thetas = data[shape_idx]
    if thetas is not None:
        thetas = thetas[:,3:] # Drop root rotation for simplicity
    f, _ = model(F)

    faces = faces.cpu().numpy()
    vertices = vertices.cpu().numpy()

    return vertices, faces, f, F, betas, thetas


def rotate_vector(pos, N):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    rotated_pos = []
    for angle in angles:
        rotation_matrix = np.array([
            [np.cos(angle),  0, np.sin(angle)],
            [0,              1, 0             ],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        rotated_pos.append(pos @ rotation_matrix.T)
    
    return np.array(rotated_pos)

def init_ps(h=1024, w=1024, ssaa=4, up='y_up'):
    import polyscope as ps
    ps.set_SSAA_factor(ssaa)
    ps.set_window_size(h, w)
    ps.init()
    ps.set_up_dir(up)
    ps.set_ground_plane_mode("shadow_only")

    return ps

def polyscope_render(ps, R, vertices, faces, colors, pcd=None, radius=0.006):

    if pcd is None:
        if faces is not None:
            mesh = ps.register_surface_mesh("Mesh", vertices, faces, smooth_shade=True, material='clay')
            mesh.add_color_quantity("Segmentation", colors, enabled=True)
        else:
            pcd = ps.register_point_cloud("Points", vertices, radius=radius)
            pcd.add_color_quantity("Segmentation", colors, enabled=True)
    else:
        mesh = ps.register_surface_mesh("Mesh", vertices, faces, smooth_shade=True, material='clay')
        mesh_colors = 0.6 * np.ones_like(vertices)
        mesh.add_color_quantity("Segmentation Mesh", mesh_colors, enabled=True)
        p = ps.register_point_cloud("Points", pcd, radius=radius)
        p.add_color_quantity("Segmentation PCD", colors, enabled=True)

    ps.look_at(R, (0,0,0))

    ps.set_ground_plane_height(vertices[:,1].min())
    ps.set_shadow_darkness(0.3)

    img = ps.screenshot_to_buffer(transparent_bg=True, vertical_flip=True)
    ps.remove_all_structures()
    
    return img


def render(vertices, faces, cols, pos=None, cat_dim=0, num_frames=3, pcd=None):
    if vertices is not None:
        if vertices.dtype == torch.float32:
            vertices = vertices.cpu().numpy()
    if faces is not None:
        if faces.dtype == torch.int32:
            faces = faces.cpu().numpy()
    if cols is not None:
        if cols.dtype == torch.float32:
            cols = cols.cpu().numpy()
    if pos is None:
        pos = np.array([3.5, 1, 0.])
    if PS is None:
        return None
    if vertices is not None and faces is not None:
        # build your mesh
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_triangle_normals()

        if not mesh.is_orientable():
            # compute which faces point inward
            V = np.asarray(mesh.vertices)
            T = np.asarray(mesh.triangles)
            N = np.asarray(mesh.triangle_normals)
            ctr = V.mean(axis=0)
            face_ctr = V[T].mean(axis=1)
            inward = np.einsum('ij,ij->i', face_ctr - ctr, N) < 0

            # flip those faces
            faces[inward] = faces[inward][:, ::-1]
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_triangle_normals()

            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

    def _render_views(vert, faces, cols):
        imgs = []
        for r in rotate_vector(pos, num_frames):
            img = torch.tensor(polyscope_render(PS, r, vert, faces, cols, pcd).copy())[None,...]
            imgs.append(img)
                
        return torch.cat(imgs, dim=cat_dim).permute(0,3,1,2).float() / 255
    return _render_views(vertices, faces, cols)

def save_video(tensor, path, fps, loop=False):
    if tensor.max() <= 1:
        tensor = tensor * 255
    if tensor.shape[1] == 4:
        tensor = tensor[:,:3,:,:]
    tensor = tensor.permute(0,2,3,1)
    video = tensor.cpu().numpy().astype(np.uint8)
    video_writer = imageio.get_writer(path, mode='I', fps=fps, codec='h264', quality=7)
    for j in range(len(video)):
        video_writer.append_data(video[j])
    if loop:
        for j in range(len(video)-1,-1,-1):
            video_writer.append_data(video[j])
    video_writer.close()

def label_to_col(num_components, _labels, do_one_hot=True, col_palette=None):
    if type(_labels) == torch.tensor or type(_labels) == torch.Tensor:
        _labels = _labels.cpu().numpy()
    if col_palette is None:
        col_palette = np.array(color_palette("husl", num_components), dtype=np.float32)
    if do_one_hot:
        labels = torch.nn.functional.one_hot(torch.tensor(_labels, dtype=torch.long), num_classes=num_components).numpy()
    else:
        labels = np.copy(_labels)
    col = (labels[...,None] * col_palette[None,None]).sum(axis=-2)
    # TODO: Clean up
    return torch.tensor(col[None]).to(dtype=torch.float32).permute(1,0,2,3)[0,0].cpu().numpy()


def get_predictive_clustering_img(num_components, model, data_src, data_tgt, src_i, tgt_i, diff3d=False, num_frames=60):
    v_src, f_src, feat_src, Feat_src, _, _ = get_data(model, data_src, src_i)
    v_tgt, f_tgt, feat_tgt, Feat_tgt, _, _ = get_data(model, data_tgt, tgt_i)

    if diff3d:
        feat_src = Feat_src
        feat_tgt = Feat_tgt

    kmeans_source_col, centroids = cluster_features(num_components, feat_src)
    kmeans_target_col, _ = cluster_features(num_components, feat_tgt, centroids=centroids)

    kmeans_source_img = render(v_src, f_src, kmeans_source_col, num_frames=num_frames)
    kmeans_target_img = render(v_tgt, f_tgt, kmeans_target_col, num_frames=num_frames)
    
    return [kmeans_source_img, kmeans_target_img]

@torch.no_grad()
def get_whole_dataset_clustering_img(num_components, model, data, indices, diff3d=False, centroids=None, num_frames=60):
    feats = []
    if centroids is None:
        with torch.no_grad():
            for bi in range(len(data)):
                _, _, F, _, _, _, _ = data[bi]
                if diff3d:
                    f = F
                else:
                    f = model.encode(F)[:,0]
                feats.append(f.cpu().numpy())
        _, centroids_whole = cluster_features(num_components, np.concatenate(feats), batched=False)
    else:
        centroids_whole = centroids

    imgs = []
    cols = []
    for i in indices:
        v, f, feat, Feat, _, _ = get_data(model, data, i)
        if diff3d:
            feat = Feat
        else:
            feat = feat
        col, _ = cluster_features(num_components, feat, centroids=centroids_whole)
        cols.append(col)
        img = render(v, f, col, num_frames=num_frames)
        imgs.append(img)

    return imgs, cols, centroids_whole, feats

@torch.no_grad()
def cluster_features(num_components, features, centroids=None, n_init="auto", seed=0, batched=False):
    if type(features) == torch.tensor or type(features) == torch.Tensor:
        features = features.cpu().numpy()
    if centroids is None:
        if batched:
            kmeans = MiniBatchKMeans(n_clusters=num_components, random_state=seed, n_init=n_init, max_iter=100, batch_size=65536).fit(features)
        else:
            kmeans = KMeans(n_clusters=num_components, random_state=seed, n_init=n_init).fit(features)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    else:
        labels = np.argmin(np.linalg.norm(features[:, np.newaxis] - centroids, axis=2), axis=1)

    return label_to_col(num_components, labels), centroids

def plot_pca_features_single(features, cols, plot_mask=None, **kwargs):
    if type(features) == torch.tensor or type(features) == torch.Tensor:
        features = features.cpu().numpy()
    if type(cols) == torch.tensor or type(cols) == torch.Tensor:
        cols = cols.cpu().numpy()

    ### PCA on joint features
    features_PCA = PCA(n_components=2)
    pca = features_PCA.fit(features)
    if plot_mask is not None:
        features = features[plot_mask]
        cols = cols[plot_mask]
    E = pca.transform(features)

    x1, y1 = E[:, 0], E[:, 1]
    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)

    plt.scatter(x1, y1, c=cols, marker='o', **kwargs)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.xticks([])
    plt.yticks([])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    return img, plt, fig

def get_correspondence_colors(v_src):
    a = 0.1
    b = 0.9
    v_min = np.min(v_src, axis=0)
    v_max = np.max(v_src, axis=0)
    v_normalized = (v_src - v_min) / (v_max - v_min)
    return  a + (b - a) * v_normalized

def get_correspondence_img(v_src, f_src, v_tgt, f_tgt, mapping, num_frames=3, skip_src_render=False, colors=None, pcd=False, joint=False):
    if colors is None:
        colors = get_correspondence_colors(v_src)
        cmap_target = colors[mapping]
    else:
        cmap_target = colors

    if pcd:
        f_tgt = None
        f_src = None
    
    imgs = render(v_tgt, f_tgt, cmap_target, num_frames=num_frames, pcd=v_tgt if joint else None)
    if not skip_src_render:
        source_img = render(v_src, f_src, colors, num_frames=num_frames, pcd=v_src if joint else None)
        imgs = torch.cat([source_img, imgs], dim=-1)

    return imgs

# Polyscope does not work via ssh, make it possible to turn it off.
value = os.getenv('no_render', 'false').strip().lower()
no_render = value in ('1', 'true', 'yes', 'on')

PS = None
if not no_render:
    PS = init_ps()