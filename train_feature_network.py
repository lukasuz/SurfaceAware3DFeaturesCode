from dataset import *
from utils import *
import torch
import os

from args import get_args
from visualisation import *
from tensorboardX import SummaryWriter
from tqdm import tqdm

def _train(args, data, save_path, val_p=0, val_pairs=[], val_w_train=False):
    if len(val_pairs) > 0:
        assert val_p == 0, "Either pass the validation pairs, they will be choosen randomly"
    do_val = val_p > 0 or len(val_pairs) > 0

    model_path = os.path.join(save_path, 'feature_network_i.pt')
    best_path = os.path.join(save_path, 'feature_network.pt')
    render_folder = os.path.join(save_path, 'train_progress')
    val_metric_path = os.path.join(save_path, 'val_metric.pt')
    
    os.makedirs(render_folder, exist_ok=True)

    model, ema, optimizer, start_i = get_feature_network(
        args, model_path, 
        update_after_step=args.FN_iters // 10, 
        update_every = 10,
        # update_model_with_ema_every = len(data),
        copy_ema=False,
    )
    if start_i >= args.FN_iters:
        return
    writer = SummaryWriter(log_dir=save_path)
    
    if val_p > 0:
        val_samples = int(val_p * len(data))
        if val_samples < 1:
            assert len(data) > 1, "If validation is intended, we need at least 2 samples in the data set."
            print("val_p too small, choosing one sample by default as validation. \n No correspondence images will be rendered")
            val_samples = 1

        indices = np.random.permutation(len(data))
        train_indices = indices[val_samples:]
        val_indices = indices[:val_samples]
        val_num = len(val_indices)

        if val_samples > 1:
            val_pairs = np.stack([
                np.arange(val_num), 
                np.roll(np.arange(val_num), 1)
            ]).T
    elif len(val_pairs) > 0:
        _val_flat = np.array(val_pairs).flatten()
        val_indices = np.array([i for i in range(len(data)) if i in _val_flat])
        if not args.train_on_validation:
            train_indices = np.array([i for i in range(len(data)) if i not in _val_flat])
        else:
            train_indices = np.array([i for i in range(len(data))])
    else:
        train_indices = np.arange(len(data))
        val_indices = None


    if val_indices is not None:
        for vi in val_indices:
            if args.train_on_validation:
                print(f"Validation set leaked into train set (Index: {vi}) but training continues due to args.train_on_validation")
            else:
                assert vi not in train_indices, f"Validation set leaked into train set (Index: {vi})."

    if val_w_train:
        do_val = True
        val_indices = train_indices
        
        val_pairs = list(zip(train_indices, np.roll(train_indices, 1))) + list(zip(np.roll(train_indices, 1), train_indices))

    cosine_loss = lambda x, y: (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)).mean()
    L_recon = cosine_loss
    L_constrastive = lambda f, D, mask: (D - (1 - torch.nn.functional.cosine_similarity(f[:,None], f[None, mask], dim=-1)) / 2).abs().mean()


    val_set = []
    @torch.no_grad()
    def do_validation(model, val_indices, write=False, render=PS is not None, eval_points=200):
        data.feature_noise_p = 0.
        data.fps_p = eval_points
        with torch.no_grad():
            val_recon_losses = []
            val_contrastive_losses = []
            val_configs = []
            for _i, vi in enumerate(val_indices):
                try:
                    vertices, faces, F, mask, D = val_set[_i]
                except:
                    vertices, faces, F, mask, D, _, _ = data[vi]
                    val_set.append([vertices, faces, F, mask, D])
                f, F_hat = model(F, norm=norm_data)
                val_configs.append([vertices, faces, f, F])
                
                val_recon_losses.append(L_recon(F, F_hat).cpu().numpy())
                val_contrastive_losses.append(L_constrastive(f, D, mask).cpu().numpy())

            if write:
                writer.add_scalar('Validation/reconstructon_loss', np.mean(val_recon_losses), i)
                writer.add_scalar('Validation/contrastive_loss', np.mean(val_contrastive_losses), i)
                writer.add_scalar('Validation/reconstructon_loss_std', np.std(val_recon_losses), i)
                writer.add_scalar('Validation/contrastive_loss_std', np.std(val_contrastive_losses), i)
                writer.flush()

            if render:
                for pair in val_pairs:
                    i_src = pair[0]
                    i_tgt = pair[1]
                    if i_src == i_tgt:
                        continue

                    v_src, f_src, features_source, feat_src_diff3f = val_configs[np.where(val_indices == i_src)[0].item()]
                    v_tgt, f_tgt, features_target, feat_tgt_diff3f = val_configs[np.where(val_indices == i_tgt)[0].item()]

                    surface_map_cos = get_point_correspondences(features_source, features_target)
                    surface_map_cos_diff3f = get_point_correspondences(feat_src_diff3f, feat_tgt_diff3f)

                    num_frames = 60
                    v_src, f_src, v_tgt, f_tgt = v_src.cpu().numpy(), f_src.cpu().numpy(), v_tgt.cpu().numpy(), f_tgt.cpu().numpy()
                    fmap_img = get_correspondence_img(v_src, f_src, v_tgt, f_tgt, surface_map_cos, num_frames=num_frames, joint=False)
                    fmap_img_diff3f = get_correspondence_img(v_src, f_src, v_tgt, f_tgt, surface_map_cos_diff3f, num_frames=num_frames, skip_src_render=True, joint=False)
                    fmap_img = torch.concat([fmap_img, fmap_img_diff3f], dim=-1)
                    save_video(fmap_img, os.path.join(render_folder, f'correspondence_cos_{i}i_{i_src}s-{i_tgt}t.mp4'), fps=30)
        data.feature_noise_p = args.feature_noise_p
        data.fps_p = args.fps_p

        return args.w_contrastive * np.mean(val_contrastive_losses) + np.mean(val_recon_losses)

    #### TRAIN #####
    reconstruction_losses = []
    contrastive_losses = []

    try:
        val_metric = torch.load(val_metric_path).item()
    except:
        print("Could not find validation metric file")
        val_metric = 100000000 

    data.norm = norm_data = True

    for i in tqdm(range(start_i, args.FN_iters+1), initial=start_i, total=args.FN_iters+1):
        optimizer.zero_grad()
        
        reconstruction_loss = torch.tensor([0.], device=args.device)
        contrastive_loss = torch.tensor([0.], device=args.device)
        smoothness_regularizer = torch.tensor([0.], device=args.device)
        point_loss = torch.tensor([0.], device=args.device)
        batch = 1
        for _ in range(batch):
            with torch.no_grad():
                b = np.random.choice(train_indices, (1,))
                _, _, F, mask, D, _, _ = data[b]
                
            f, F_hat = model(F, norm=True)

            # Reconstruction Loss
            if args.w_reconstruction > 0:
                reconstruction_loss += L_recon(F, F_hat)

            # Contrastive Loss
            if args.w_contrastive > 0:
                contrastive_loss = L_constrastive(f, D, mask)
                contrastive_loss += contrastive_loss
        
        reconstruction_loss = args.w_reconstruction * reconstruction_loss / batch
        contrastive_loss = args.w_contrastive * (contrastive_loss / batch)

        reconstruction_losses.append(reconstruction_loss.item())
        contrastive_losses.append(contrastive_loss.item())

        loss = reconstruction_loss + contrastive_loss + smoothness_regularizer + point_loss
        loss.backward()
        optimizer.step()
        ema.update()

        # Visualization
        if (i % args.FN_render_iter == 0):
            reconstruction_loss = np.mean(reconstruction_losses) if len(reconstruction_losses) > 0 else 0
            contrastive_loss = np.mean(contrastive_losses) if len(contrastive_losses) > 0 else 0
            reconstruction_losses = []
            contrastive_losses = []

            writer.add_scalar('Train/reconstructon_loss', reconstruction_loss, i)
            writer.add_scalar('Train/contrastive_loss', contrastive_loss, i)

            # write losses to tqdm
            tqdm.write(f'[{i}/{args.FN_iters}] Recon: {reconstruction_loss:.5f}, Contrastive: {contrastive_loss:.5f}')
            save_checkpoint(ema, optimizer, i, path=model_path)

            if do_val:
                _val_metric = do_validation(ema, val_indices, write=True).item()
                if _val_metric < val_metric: 
                    val_metric = _val_metric
                    save_checkpoint(ema, optimizer, i, path=best_path)
                    torch.save(torch.tensor(val_metric), val_metric_path)
            else:
                # do_validation(ema, train_indices, write=True)
                save_checkpoint(ema, optimizer, i, path=best_path)

    # save_checkpoint(model, optimizer, i, path=model_path)
    save_checkpoint(ema, optimizer, i, path=model_path)
    if do_val:
        do_validation(ema, val_indices, write=True)
    writer.close()

def train(args):
    save_path = get_experiment_folder(args)
    os.makedirs(save_path, exist_ok=True)
    seed_everything(args.seed)

    train_data = None
    # Get datasets, merge if multiple provided
    for data_name in args.train_data:
        if data_name not in DATA.keys():
            raise ValueError('Training data set not available')

        _train_data, _ = DATA[data_name](args)
        if train_data is None:
            train_data = _train_data
        else:
            train_data = ShapeDataset.merge_into(train_data, _train_data)

    train_data.deferred_masking(args.mask)

    # Parse validation pairs
    val_pairs = []
    if len(args.val_pairs) % 2 == 1:
        raise ValueError("Validation pairs must come in two pairs. For metrics only, pass the same index twice.")
    elif len(args.val_pairs) > 1:
        for i in range(0, len(args.val_pairs), 2):
            val_pairs.append([
                int(args.val_pairs[i]), int(args.val_pairs[i+1])
            ])

    print("Training data length:", len(train_data))
    print("Validation pairs", val_pairs)
    _train(args, train_data, save_path, val_pairs=val_pairs)

if __name__ == "__main__":
    args = get_args()
    train(args)
