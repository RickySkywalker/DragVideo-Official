import gdown
import numpy as np
import os
import requests
import json
import torch
import torch.nn.functional as F
import math

from TrackAnything.track_anything import TrackingAnything
from TrackAnything.track_anything import parse_augment
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

from pips.nets.pips import Pips
import pips.utils.improc
import pips.saverloader



############################ pips ################################

def run_model(model, rgbs, points):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 512, 512
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    xy0 = []
    # print("points:",points.shape)
    
    xy0 = points.unsqueeze(0)
    N=len(points)
    
    _, S, C, H, W = rgbs.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:,0] = xy0[:,n] # B, 1, 2  # set first position 
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad

    prep_rgbs = pips.utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)      

    return trajs_e-pad

def tracking_pip(points, frames):
    init_dir = 'pips/reference_model'

    model = Pips(stride=4).cuda()
    if init_dir:
        _ = pips.saverloader.load(init_dir, model)
    model.eval()
    
    rgbs = []
    for s in range(len(frames)):
        im = frames[s]
        im = im.astype(np.uint8)
        rgbs.append(torch.from_numpy(im).permute(2,0,1))
                
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W

    with torch.no_grad():
        trajs_e = run_model(model, rgbs, points)

    model = None
    torch.cuda.empty_cache()

    return trajs_e[0]


def clip_point(point, offset, max_length):
    x, y = point
    dx, dy = offset
    
    new_x, new_y = x + dx, y + dy
    
    distance = math.sqrt(dx**2 + dy**2)

    if distance <= max_length:
        return np.array([new_x, new_y])
    
    scale_factor = max_length / distance
    
    clipped_dx, clipped_dy = dx * scale_factor, dy * scale_factor
    
    clipped_x, clipped_y = x + clipped_dx, y + clipped_dy
    
    return np.array([clipped_x, clipped_y])


def tracking(start_handle_points, start_target_points, end_handle_points, end_target_points, frames):

    start_vector = start_target_points - start_handle_points
    
    end_vector = end_target_points - end_handle_points
    if end_vector.shape[-1] != start_vector.shape[-1]:
        end_vector = start_vector

    video_len = frames.shape[0]
    # print(f"video shape: {frames.shape}")

    tracking_points = tracking_pip(start_handle_points, frames = frames).cpu()
    target_points = [start_target_points]


    # print("start vector", start_vector)
    # print("end vector", end_vector)
    

    for i in range(1, video_len):
        p = i / video_len
        handle_target_diff = start_vector * (1 - p) + end_vector * p

        curr_handle_points = tracking_points[i]
        # print("tracking point is", tracking_points[i])
        curr_target_points = []
        for j in range(curr_handle_points.shape[0]):
            curr_target_point = clip_point(curr_handle_points[j], handle_target_diff[j], 512)
            curr_target_points.append(curr_target_point)
        curr_target_points = np.array(curr_target_points)
        #curr_target_points = tracking_points[i] + handle_target_diff

        target_points.append(curr_target_points)

    # target_points.append(end_target_points)
    target_points = np.array(target_points)

    return tracking_points.numpy(), target_points


############################ Track-Anything ############################

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

def load_sam_model():
    # check and download checkpoints if needed
    tam_args = parse_augment()
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    sam_checkpoint = SAM_checkpoint_dict[tam_args.sam_model_type] 
    sam_checkpoint_url = SAM_checkpoint_url_dict[tam_args.sam_model_type] 
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
    e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
    e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


    folder ="TrackAnything/checkpoints"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
    return TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,tam_args)

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt