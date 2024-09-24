import numpy as np
import torch
import copy
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler
# from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from AnimateDiff.animatediff.models.unet import UNet3DConditionModel
from AnimateDiff.animatediff.utils.util import load_weights

from AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline

from einops import rearrange

from DragVideo.utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from DragVideo.utils.drag_utils import drag_batch_update
from DragVideo.utils.point_utils import process_points_to_sup_res

from DragVideo.args import drag_args
from DragVideo.utils.function_utils import preprocess_video, encode_video, encode_prompts


def init_modules(drag_args, device):
    if "v2" in drag_args.motion_module:
            inference_config = "AnimateDiff/configs/inference/inference-v2.yaml"
    elif "v3" in drag_args.motion_module:
        inference_config = "AnimateDiff/configs/inference/inference-v3.yaml"
    else:
        inference_config = "AnimateDiff/configs/inference/inference-v1.yaml"

    unet = UNet3DConditionModel.from_pretrained_2d(
        drag_args.pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.load(inference_config).unet_additional_kwargs
    )
    # if is_xformers_available() and torch.cuda.is_available():
    #     unet.enable_xformers_memory_efficient_attention()

    noise_scheduler_cls = DDIMScheduler
    
    noise_scheduler_kwargs = OmegaConf.load(inference_config).noise_scheduler_kwargs
    # if noise_scheduler_cls == EulerDiscreteScheduler:
    #     noise_scheduler_kwargs.pop("steps_offset")
    #     noise_scheduler_kwargs.pop("clip_sample")
    # elif noise_scheduler_cls == PNDMScheduler:
    #     noise_scheduler_kwargs.pop("clip_sample")

    pipeline = AnimationPipeline(
        unet=unet,
        vae=AutoencoderKL.from_pretrained(drag_args.pretrained_model_path, subfolder="vae"), 
        text_encoder=CLIPTextModel.from_pretrained(drag_args.pretrained_model_path, subfolder="text_encoder"), 
        tokenizer=CLIPTokenizer.from_pretrained(drag_args.pretrained_model_path, subfolder="tokenizer"), 
        scheduler=noise_scheduler_cls(**noise_scheduler_kwargs),
    )

    pipeline = load_weights(
        pipeline,
        motion_module_path=drag_args.motion_module,
        dreambooth_model_path='AnimateDiff/models/DreamBooth_LoRA/'+drag_args.dreambooth_model_path,
        lora_model_path=drag_args.lora_model_path,
        lora_alpha=drag_args.lora_alpha,
    )

    pipeline.to(device)
    # self.pipeline = pipeline
    print("done.")

    return pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, pipeline


class DragVideoPipeline:
    def __init__(self,
                 unet: UNet3DConditionModel,
                 vae: AutoencoderKL,
                 tokenizer: CLIPTokenizer,
                 text_encoder: CLIPTextModel,
                 scheduler: DDIMScheduler,
                 args,
                 inverse_scheduler: DDIMInverseScheduler = None,
                 device=torch.device("cpu"),
                 guidance_scale=7.0):

        self.device = device

        self.unet_store = copy.deepcopy(unet.to(torch.device("cpu")))

        self.num_inference_steps = args.n_inference_steps
        self.actual_inference_steps = args.n_actual_steps
        self.unet = unet.to(device)

        self.vae = vae.to(self.device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.scheduler = scheduler
        self.inverse_scheduler = inverse_scheduler
        self.guidance_scale = guidance_scale

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.scheduler.set_timesteps(self.num_inference_steps)
        if self.inverse_scheduler != None:
            self.inverse_scheduler.set_timesteps(self.num_inference_steps)

        self.args = args
        self.use_lora = False

    def load_lora_to_unet(self, lora_location):
        unet_lora_attn_procs = torch.load(lora_location, map_location="cpu")
        self.unet.set_attn_processor(unet_lora_attn_procs)
        self.unet = self.unet.to(self.device)
        self.unet_store = copy.deepcopy(self.unet)
        self.unet_store = self.unet_store.to(torch.device("cpu"))

        self.use_lora = True

    def reinit_unet(self, unet=None):
        if unet == None:
            self.unet = self.unet.to(torch.device("cpu"))
            self.unet = copy.deepcopy(self.unet_store)
            self.unet = self.unet.to(self.device)
            self.unet.requires_grad_(False)
        else:
            self.unet = self.unet.to(torch.device("cpu"))
            self.unet = unet.to(self.device)
            self.unet.requires_grad_(False)

    def reinit_scheduler(self,
                         scheduler: DDIMScheduler,
                         inverse_scheduler: DDIMInverseScheduler = None,
                         num_inference_steps=40):
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.num_inference_steps = num_inference_steps
        if inverse_scheduler != None:
            self.inverse_scheduler = inverse_scheduler
            self.inverse_scheduler.set_timesteps(num_inference_steps)

    def _preprocess_video(self, videos):
        return preprocess_video(videos)

    def _encode_video(self, video, video_len):
        return encode_video(video, self.vae, video_len, self.device)

    # We ban the usage of attention mask in this processing
    def _encode_prompt(self, prompts, n_prompts):
        return encode_prompts(prompts, n_prompts, self.tokenizer, self.text_encoder, self.device)

    def _add_noise_DDIMInversion(self, input, text_embeddings, begin_step=0, end_step=None):
        if end_step == None:
            end_step = self.actual_inference_steps

        if end_step >= self.num_inference_steps:
            end_step = self.num_inference_steps

        if self.inverse_scheduler == None:
            print(
                "There is no inverse scheduler when initializing the model, use default stable diffusion 1.5 DDIM Inverse scheduler")
            self.inverse_scheduler = DDIMInverseScheduler.from_pretrained("./AnimateDiff/models/StableDiffusion",
                                                                          subfolder="scheduler")
            self.inverse_scheduler.set_timesteps(self.num_inference_steps)

        latents = input

        for i, t in enumerate(tqdm(self.inverse_scheduler.timesteps)):
            if i <= begin_step:
                continue

            if i >= end_step:
                break

            # input = self.inverse_scheduler.scale_model_input(input)
            with torch.no_grad():
                torch.manual_seed(42)
                # expand the latents
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(
                    dtype=latents.dtype)

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def _remove_noise(self, latents, text_embeddings, starting_steps=None):
        batch_size = len(text_embeddings)

        print(f"initial noise sigma: {self.scheduler.init_noise_sigma}")
        latents *= self.scheduler.init_noise_sigma
        latents_dtype = latents.dtype

        # We do not use extra step kwargs here
        if starting_steps == None:
            starting_steps = self.num_inference_steps - self.actual_inference_steps


        # Denoising loop
        timesteps = self.scheduler.timesteps[starting_steps:]

        for i, t in enumerate(tqdm(timesteps)):
            torch.manual_seed(42)
            # expand the latents
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(
                dtype=latents_dtype)

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def _decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents

        print(f"latent shape: {latents.shape}")

        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        videos = []

        for frame_idx in tqdm(range(latents.shape[0])):
            videos += [self.vae.decode(latents[frame_idx: frame_idx + 1]).sample]

        videos = torch.cat(videos)
        videos = rearrange(videos, "(b f) c h w -> b c f h w", f=video_length)
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = videos.cpu().float().numpy()
        # rescale the video
        videos = ((videos) * 255).astype(np.uint8)
        return videos

    def _repeat_text_emb_MutualSelfAttn(self, text_embedding):
        to_return = []
        for i in range(len(text_embedding)):
            to_return += [text_embedding[i], text_embedding[i]]
        to_return = torch.stack(to_return)
        # print(f"text embedding shape after process is: {to_return.shape}")
        return to_return

    def process_drag_batch(self,
                           videos,
                           handle_points,
                           target_points,
                           mask,
                           prompts,
                           drag_batch,
                           n_pix_step=40,
                           n_prompts=None,
                           # These two parameters are for MutualSelfAttentionControl
                           start_step=0,
                           start_layer=10):
        self.args.n_pix_step = n_pix_step
        # make sure the video is in form b*f*h*w*c should be in form of 0~255 integers
        assert len(videos.shape) == 5, "Video have bad shape"
        # make sure the prompts and videos are in same batch_size
        assert len(prompts) == videos.shape[0], "Prompts length is different from video"

        # transform the video into torch.FloatTensor
        videos = self._preprocess_video(videos).float()
        video_length = videos.shape[1]

        # print("Current processing video shape is:", videos.shape)
        print("encoding video...")
        latents = self._encode_video(videos, video_length)
        # print("Latent shape is:", latents.shape)

        # attain batch size, the batch_size b of video should be the same as the number of pormots
        batch_size = len(prompts)

        # encode the prompts
        if n_prompts == None:
            n_prompts = [""] * batch_size
        text_embeddings = self._encode_prompt(prompts, n_prompts)
        latents = self._add_noise_DDIMInversion(latents, text_embeddings)

        # transforms points into sup_res, since the next input will be latents
        handle_points = process_points_to_sup_res(handle_points, (drag_args.H, drag_args.W),
                                                                 (drag_args.sup_res, drag_args.sup_res))
        target_points = process_points_to_sup_res(target_points, (drag_args.H, drag_args.W),
                                                                 (drag_args.sup_res, drag_args.sup_res))

        text_embeddings_for_drag = text_embeddings[1].unsqueeze(0)
        original_latents = latents.clone()
        print("start dragging...")
        original_latents = latents.clone()
        latents = drag_batch_update(self.unet,
                                               self.scheduler,
                                               latents,
                                               self.inverse_scheduler.timesteps[self.actual_inference_steps],
                                               handle_points,
                                               target_points,
                                               text_embeddings_for_drag,
                                               mask,
                                               self.args,
                                               device=self.device,
                                               drag_batch=drag_batch)
        torch.cuda.empty_cache()
        latents.detach_()

        print("begin denoising the video...")

        if self.args.msa:
            latents = torch.cat([original_latents, latents], dim=0)
            text_embeddings = self._repeat_text_emb_MutualSelfAttn(text_embeddings)

            # This is the section for MutualSelfAttentionControl
            editor = MutualSelfAttentionControl(start_step=start_step,
                                                start_layer=start_layer,
                                                total_steps=self.num_inference_steps,
                                                guidance_scale=self.guidance_scale,
                                                drag_batch=video_length)
            if self.use_lora:
                register_attention_editor_diffusers(self, editor, attn_processor="lora_attn_proc")
            else:
                register_attention_editor_diffusers(self, editor, attn_processor="attn_proc")

        with torch.no_grad():
            latents = self._remove_noise(latents,
                                         text_embeddings,
                                         starting_steps=self.num_inference_steps - self.actual_inference_steps)
            decoded_videos = self._decode_latents(latents)

        return decoded_videos