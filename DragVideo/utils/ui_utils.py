import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange

from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torchvision.io as io
from diffusers import DDIMScheduler, DDIMInverseScheduler

import sys
sys.path.append('pips')
sys.path.append('TrackAnything')
sys.path.append('TrackAnything/tracker')

from DragVideo.pipeline import DragVideoPipeline, init_modules
from DragVideo.LoRA_helper.LoRATrainHelper import LoRATrainHelper

from DragVideo.utils.track_utils import load_sam_model, get_prompt, tracking  ###
from DragVideo.utils.video_utils import VideoData, load_video
from DragVideo.utils.utils import get_best_device, point_and_save_video
from DragVideo.utils.utils import mask_painter, extend_mask, first_extend

from DragVideo.args import drag_args, DEVICE_ID


# -------------- general UI functionality --------------
def clear_all(length=480):
    return gr.Image.update(value=None), \
        gr.Image.update(value=None), \
        gr.Image.update(value=None), \
        gr.Video.update(value=None), \
        gr.Video.update(value=None), \
        gr.Video.update(value=None), \
        gr.Video.update(value=None), \
        [], [], None


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0)
    return out


def store_video(video, max_len, drag_batch, kps, save_dir, length=512):
    if video is None:
        return None, None, None, None, None, None, None, None, None, 24
    
    to_process_video, fps = load_video(video)

    video_len = min(max_len, to_process_video.shape[0])
    # print(f"video length: {video_len}")
    if kps == 0:
        preview_frames = to_process_video[:int(video_len)]
    else:
        preview_frames = to_process_video[:int(video_len)*int(fps//kps):int(fps//kps)]
    # print(f"preview frames shape: {preview_frames.shape}")

    start_frame = preview_frames[0].numpy()
    end_frame = preview_frames[-1].numpy()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    original_path = os.path.join(save_dir, 'original.mp4')

    io.write_video(original_path, preview_frames, fps=kps)
    
    return original_path, original_path, None, None, start_frame, end_frame, deepcopy(start_frame), deepcopy(end_frame), deepcopy(start_frame), fps
# once user upload an image, the original image is stored in `original_image`
# the same image is displayed in `input_image` for point clicking purpose


# user click the image to get points, and show the points on the image
def get_points_start(img,
                     sel_pix,
                     click_state,
                     painted_image,
                     mask,
                     logit,
                     evt: gr.SelectData):
    # collect the selected point
    img = img.copy()
    sel_pix.append(evt.index)
    if len(sel_pix) % 2 == 1:
        painted_image, mask, logit = sam_refine(deepcopy(img), "Positive", click_state, evt)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 2, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img), painted_image, mask, logit


def get_points_end(img,
                   sel_pix,
                   evt: gr.SelectData):
    # collect the selected point
    img = img.copy()
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 2, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


# clear all handle/target points
def undo_points(original_image):
    return original_image.copy(), []


# ------------------------------------------------------

sam_model = None


# use sam to get the mask
def sam_refine(origin_images, point_prompt, click_state, evt: gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    global sam_model
    if sam_model is None:
        sam_model = load_sam_model()
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])

    # prompt for sam model
    sam_model.samcontroler.sam_controler.reset_image()
    sam_model.samcontroler.sam_controler.set_image(origin_images)
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = sam_model.first_frame_click(
        image=origin_images,
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    return painted_image, mask, logit


def clear_click(origin_images, click_state):
    click_state = [[], []]
    template_frame = origin_images
    return template_frame, click_state


def train_lora_interface(original_video,
                         prompt,
                         model_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_rank,
                         lora_batch,
                         progress=gr.Progress()):
    drag_args.dreambooth_model_path = model_path

    # TODO: This is just a temp fix for gradio problem, need to fix it later
    device = get_best_device(DEVICE_ID)
    to_process_video, fps = load_video(original_video)

    video_data = VideoData(to_process_video, prompt, "", len_per_slice=lora_batch)
    unet, vae, text_encoder, tokenizer, pipeline = init_modules(drag_args, device)

    scheduler = DDIMScheduler.from_pretrained(drag_args.pretrained_model_path, subfolder="scheduler")

    lora_train_helper = LoRATrainHelper(unet,
                                        vae,
                                        text_encoder,
                                        tokenizer,
                                        scheduler,
                                        lora_rank=lora_rank,
                                        device=device)

    print('Start training LoRA... (progress bar is shown in gradio ui)')

    lora_train_helper.train(video_data,
                            lora_step,
                            lora_lr,
                            len_per_slice=lora_batch,
                            progress=progress)

    print('Training LoRA Done!')

    lora_train_helper.save_lora(lora_path)
    video_data = None
    unet = None
    vae = None
    text_encoder = None
    tokenizer = None
    pipeline = None
    torch.cuda.empty_cache()
    return "Training LoRA Done!"


def propagate(original_video,
              template_mask,
              prompt,
              points_start,
              points_end,
              kps,
              save_dir,
              radius,
              ):

    # TODO: This is just a temp fix for gradio problem, need to fix it later
    to_process_video, fps = load_video(original_video)
    # print(f"point propagate video shape: {to_process_video.shape}")

    video_data = VideoData(to_process_video, prompt, "")

    global sam_model
    if sam_model is None:
        sam_model = load_sam_model()
    sam_model.xmem.clear_memory()

    masks, logits, painted_images = sam_model.generator(images=video_data.video, template_mask=template_mask)
    # clear GPU memory
    sam_model.xmem.clear_memory()

    points_start = np.array(points_start)
    points_start = torch.from_numpy(points_start).float()

    points_end = np.array(points_end)
    points_end = torch.from_numpy(points_end).float()
    start_handle_point = points_start[::2]
    start_target_point = points_start[1::2]
    end_handle_point = points_end[::2]
    end_target_point = points_end[1::2]

    # print(f"current points: {start_handle_point, start_target_point, end_handle_point, end_target_point}")

    # Firstly, we perform all the point tracking to get points
    handle_points, target_points = tracking(
        start_handle_point,
        start_target_point,
        end_handle_point,
        end_target_point,
        video_data.video,
    )
    #extend the mask
    masked_img = []
    for i in range(len(masks)):
        # first extend the mask
        if radius>0:
            masks[i] = first_extend(masks[i], radius=radius)
        curr_mask = masks[i]
        for j in range (len(handle_points[i])):
            curr_handle_point = handle_points[i][j]
            curr_target_point = target_points[i][j]
            if(curr_mask[int(curr_target_point[1]), int(curr_target_point[0])] == 1):
                continue
            offset = curr_target_point- curr_handle_point
            masks[i] = extend_mask(masks[i], offset)
            painted_images[i] = mask_painter(video_data.video[i], masks[i])


    for i in range(len(masks)):
        image = video_data.video[i]
        image = video_data.video[i]
        if masks[i].sum() > 0:
            mask = np.uint8(masks[i] > 0)
            masked_img.append(mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.4))
        else:
            masked_img.append(image.copy())        
    
    # print("checking the point tracking result...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    point_and_save_video(np.array(painted_images), handle_points, target_points,
                         os.path.join(save_dir, 'preview1.mp4'), fps=kps)
    point_and_save_video(np.array(masked_img), handle_points, target_points,
                            os.path.join(save_dir, 'preview.mp4'), fps=kps)
    return os.path.join(save_dir, 'preview.mp4'), handle_points, target_points, masks, logits


def run_drag(original_video,
             masks,
             prompt,
             handle_points,
             target_points,
             kps,
             drag_batch,
             inverse_batch,
             inversion_strength,
             lam,
             latent_lr,
             n_pix_step,
             model_path,
             lora_path,
             guidance_scale,
             start_step,
             start_layer,
             save_dir,
             n_prompt=""
             ):
    drag_args.inversion_strength = inversion_strength
    drag_args.n_actual_steps = int(drag_args.n_inference_steps * inversion_strength)
    drag_args.dreambooth_model_path = model_path
    drag_args.lam = lam
    drag_args.lr = latent_lr

    global sam_model
    sam_model = None
    torch.cuda.empty_cache()

    # TODO: This is just a temp fix for gradio problem, need to fix it later
    to_process_video, fps = load_video(original_video)

    device = get_best_device(DEVICE_ID)

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(drag_args.pretrained_model_path,
                                                             subfolder="scheduler")
    scheduler = DDIMScheduler.from_pretrained(drag_args.pretrained_model_path, subfolder="scheduler")
    # reinit the unet
    unet, vae, text_encoder, tokenizer, pipeline = init_modules(drag_args, device)
    dragVideoPipeline = DragVideoPipeline(unet,
                                          vae,
                                          tokenizer,
                                          text_encoder,
                                          scheduler,
                                          drag_args,
                                          inverse_scheduler=inverse_scheduler,
                                          device=device,
                                          guidance_scale=guidance_scale,
                                          )

    if lora_path != None and lora_path != "":
        dragVideoPipeline.load_lora_to_unet(lora_path)

    masks = np.stack(masks)

    # print("the shape of the masks is:", masks.shape)
    drag_batch = int(drag_batch)
    # print(f"current drag batch is: {drag_batch}")
    video_data = VideoData(to_process_video,
                           prompt,
                           n_prompt=n_prompt,
                           masks=masks,
                           len_per_slice=inverse_batch)
    videos, masks, prompts, n_prompts = video_data.generate_batch()
    handle_points = deepcopy(handle_points)
    target_points = deepcopy(target_points)

    handle_points = torch.from_numpy(handle_points)
    target_points = torch.from_numpy(target_points)

    # print(f"handle points shape: {handle_points.shape}")
    # print(f"target points shape: {target_points.shape}")
    # print(f"video shape for first batch is: {videos[0].shape}")

    splited_points = [video.shape[0] for video in videos]
    batched_handle_points = torch.split(handle_points, splited_points)
    batched_target_points = torch.split(target_points, splited_points)

    processed_videos = []

    for i in range(len(videos)):
        torch.manual_seed(42)
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(drag_args.pretrained_model_path,
                                                                 subfolder="scheduler")
        scheduler = DDIMScheduler.from_pretrained(drag_args.pretrained_model_path, subfolder="scheduler")
        dragVideoPipeline.reinit_scheduler(scheduler, inverse_scheduler, num_inference_steps=drag_args.n_inference_steps)
        curr_batch_size = len(prompts[i])
        print(f"processing video slice {i}")
        curr_len = videos[i].shape[0]
        curr_video = torch.from_numpy(videos[i])
        curr_video = torch.FloatTensor(curr_video.unsqueeze(0).float())
        curr_masks = masks[i]
        # print(f"current video shape: {curr_video.shape}")
        prompt = [prompts[i]]
        n_prompt = [n_prompts[i]]

        curr_handle_points = batched_handle_points[i]
        curr_target_points = batched_target_points[i]

        # print(f"current handle points: {curr_handle_points}")
        # print("shape of the curr_mask is:", curr_masks.shape)

        # curr_masks = utils.utils.gen_uniform_mask((args.H, args.W), args.sup_res)

        curr_processed_video = dragVideoPipeline.process_drag_batch(curr_video,
                                                                    curr_handle_points,
                                                                    curr_target_points,
                                                                    curr_masks,
                                                                    prompt,
                                                                    drag_batch,
                                                                    n_pix_step=n_pix_step,
                                                                    n_prompts=n_prompt,
                                                                    start_step=start_step,
                                                                    start_layer=start_layer
                                                                    )

        dragVideoPipeline.reinit_unet()
        processed_videos += [curr_processed_video[1]]
        torch.cuda.empty_cache()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    processed_video = np.concatenate(processed_videos, axis=1)
    processed_video = rearrange(processed_video, "c f h w -> f h w c")
    # print(f"processed videos shape: {processed_video.shape}")
    io.write_video(os.path.join(save_dir, 'output.mp4'), processed_video, fps=kps)

    return os.path.join(save_dir, 'output.mp4')
