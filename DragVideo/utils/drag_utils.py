# Adapted from https://github.com/Yujun-Shi/DragDiffusion/blob/main/utils/drag_utils.py

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.auto import tqdm


def point_tracking(F0, F1, handle_points, handle_points_init, args):
    assert handle_points.shape == handle_points_init.shape, f"handle_points and handle_points_init shoule be in same shape\n" \
                                                            f"handle_points is {handle_points.shape}; handle_points_init is: {handle_points_init.shape}"
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = int(pi[0]) - args.r_p, int(pi[0]) + args.r_p + 1
            c1, c2 = int(pi[1]) - args.r_p, int(pi[1]) + args.r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - args.r_p + row
            handle_points[i][1] = pi[1] - args.r_p + col

    return handle_points


# This function make sure that everything is within range
def check_handle_reach_target(handle_points, target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 1.0).all()


# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    Ib = feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1]
    Ic = feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1]
    Id = feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def _get_forward_unet_features(unet,
                               z,
                               t,
                               encoder_hidden_states,
                               layer_idx=[0],
                               interp_res=256):
    unet_output, all_interemediate_features = unet.get_features(z, t, encoder_hidden_states=encoder_hidden_states)

    all_return_features = []

    for idx in layer_idx:
        feat = all_interemediate_features[idx]
        all_frame_features = []
        for f in range(feat.shape[2]):
            feat_f = F.interpolate(feat[:, :, f, :, :], (interp_res, interp_res), mode='bilinear')
            all_frame_features.append(feat_f.unsqueeze(2))
        all_return_features.append(torch.cat(all_frame_features, dim=2))
    return_features = torch.cat(all_return_features, dim=1)
    return unet_output, return_features


def gen_slice(video_len, process_batch_size=3):
    idx_ls = list(range(video_len))
    return [idx_ls[i: i + process_batch_size] for i in range(0, video_len, process_batch_size)]


def _check_points_reach_target(handle_points, target_points):
    status = 0
    for i in range(len(handle_points)):
        status += 1 if not check_handle_reach_target(handle_points[i], target_points[i]) else 0
    return status == 0


# This function will perform drag with no condition, ignoring the batch size and gpu limit, better not directly use this function
# use the drag_diffusion_update for video drag

def perform_drag(unet,
                 scheduler,
                 t,
                 init_code,
                 handle_points,
                 target_points,
                 masks,
                 text_emb,
                 args):
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = _get_forward_unet_features(unet,
                                                     init_code,
                                                     t,
                                                     text_emb,
                                                     layer_idx=args.unet_feature_idx,
                                                     interp_res=args.sup_res)
        # Get the previous sampling for masking
        x_prev_0 = scheduler.step(unet_output, t, init_code).prev_sample

    handle_points_init = copy.deepcopy(handle_points)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    masks = masks.unsqueeze(0)
    interp_mask = F.interpolate(masks, (init_code.shape[3], init_code.shape[4]), mode='nearest')
    interp_mask = interp_mask.unsqueeze(0)
    # print("interp_mask shape after is: ", interp_mask.shape)

    for step_idx in tqdm(range(args.n_pix_step)):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = _get_forward_unet_features(unet,
                                                         init_code,
                                                         t,
                                                         text_emb,
                                                         layer_idx=args.unet_feature_idx,
                                                         interp_res=args.sup_res)
            x_prev_updated = scheduler.step(unet_output, t, init_code).prev_sample
            loss = 0.0
            chunck_size = len(F0[0, 0, :, 0, 0])

            if _check_points_reach_target(handle_points, target_points):
                print(f"early stop at step: {step_idx}")
                break

            for frame_idx in range(chunck_size):
                curr_F0 = F0[:, :, frame_idx, :, :]
                curr_F1 = F1[:, :, frame_idx, :, :]
                curr_handle_points = handle_points[frame_idx]
                curr_target_points = target_points[frame_idx]
                
                # print(f"Current handle points is:\n{curr_handle_points}")
                # print(f"Current target points is:\n{curr_target_points}")

                # perform point tracking if not the first step
                if step_idx != 0:
                    curr_handle_points = point_tracking(curr_F0,
                                                        curr_F1,
                                                        handle_points[frame_idx],
                                                        handle_points_init[frame_idx],
                                                        args)
                    handle_points[frame_idx] = curr_handle_points

                # perform the loss tracking
                for point_idx in range(len(curr_handle_points)):
                    pi, ti = curr_handle_points[point_idx], curr_target_points[point_idx]

                    # Skip this if distance between handle and target is less than 1:
                    if (ti - pi).norm() < 1:
                        continue

                    di = (ti - pi) / (ti - pi).norm()

                    # perform motion supervision
                    f0_patch = curr_F1[:, :,
                               (int(pi[0]) - args.r_m): (int(pi[0]) + args.r_m + 1),
                               (int(pi[1]) - args.r_m): (int(pi[1]) + args.r_m + 1)].detach()
                    f1_patch = interpolate_feature_patch(curr_F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                    loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)
            loss += args.lam * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            loss /= chunck_size

            # print(f"Total loss of step {step_idx} is: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        optimizer.zero_grad()
    return init_code


def drag_batch_update(unet,
                      scheduler,
                      init_code,
                      t,
                      handle_points,
                      target_points,
                      text_emb,
                      mask,
                      args,
                      device='cuda',
                      drag_batch=4):
    unet = unet.to(device)
    init_code = init_code.to(device)
    handle_points = handle_points.to(device)
    target_points = target_points.to(device)
    text_emb = text_emb.to(device)
    mask = torch.from_numpy(mask).to(device)

    init_code = init_code.requires_grad_(False)

    video_len = init_code.shape[2]

    # print(f"current mask shape: {mask.shape}")
    # print(f"text_emb shape: {text_emb.shape}")
    drag_batch_slice = gen_slice(video_len, drag_batch)

    for i, curr_batch in enumerate(drag_batch_slice):
        print(f"begin dragging batch: {i}")
        curr_init_code = init_code[:, :, curr_batch, :, :].detach().requires_grad_(True)
        curr_handle_points = handle_points[curr_batch]
        curr_target_points = target_points[curr_batch]
        curr_mask = mask[curr_batch, :, :]
        # print(f"current init_code shape: {curr_init_code.shape}")
        # print(f"current text_emb shape: {text_emb.shape}")
        modified_init_code = perform_drag(unet,
                                          scheduler,
                                          t,
                                          curr_init_code,
                                          curr_handle_points,
                                          curr_target_points,
                                          curr_mask,
                                          text_emb,
                                          args)
        init_code[:, :, curr_batch, :, :] = modified_init_code

    return init_code
