import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import torch.nn.functional as F

from .loaders import AttnProcsLayers, LoraLoaderMixin

from DragVideo.utils.function_utils import encode_prompts, preprocess_video, encode_video
from DragVideo.utils.video_utils import VideoData

from DragVideo.LoRA_helper.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)



class LoRATrainHelper:
    def __init__(self,
                 unet,
                 vae,
                 text_encoder,
                 tokenizer,
                 scheduler,
                 lora_rank=1,
                 device='cuda'):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        self.lora_rank = lora_rank

        self.unet = self.unet.to(device, dtype=torch.float16)
        self.vae = self.vae.to(device, dtype=torch.float16)
        self.text_encoder = self.text_encoder.to(device, dtype=torch.float16)

        # print("begin to load LoRA...")
        # self.unet_lora_attn_procs = self._preare_lora_attn_procs()

    # This function is used to prepare the LoRA, and load it to the unet
    def _preare_lora_attn_procs(self):
        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError("name must startwith mid_block, up_blocks or down_blocks")

            if isinstance(attn_processor, (AttnAddedKVProcessor,
                                           SlicedAttnAddedKVProcessor,
                                           AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.lora_rank
            )

        self.unet.set_attn_processor(unet_lora_attn_procs)
        return self.unet.attn_processors

    def train(self,
              train_data: VideoData,
              epoch,
              lr,
              betas=(0.9, 0.999),
              weight_decay=1E-2,
              eps=1E-8,
              len_per_slice=16,
              progress=None):

        # move everything to the desired device
        self.vae = self.vae.to(self.device, dtype=torch.float16)
        self.unet = self.unet.to(self.device, dtype=torch.float16)
        self.text_encoder = self.text_encoder.to(self.device, dtype=torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError("name must startwith mid_block, up_blocks or down_blocks")

            if isinstance(attn_processor, (AttnAddedKVProcessor,
                                           SlicedAttnAddedKVProcessor,
                                           AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.lora_rank
            )

        self.unet.set_attn_processor(unet_lora_attn_procs)

        unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)
        para_to_optimize = (unet_lora_layers.parameters())

        optimizer = torch.optim.AdamW(
            para_to_optimize,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )


        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch,
            num_cycles=1,
            power=1.0
        )


        # The following lines will prepare the batches for training
        train_data.set_len_per_slice(len_per_slice)
        videos, prompts, n_prompts = train_data.generate_batch()

        if progress == None:
            cur_tqdm = tqdm
        else:
            cur_tqdm = progress.tqdm

        for _ in cur_tqdm(range(epoch), desc="training LoRA"):
            self.unet.train()
            for i in range(len(videos)):
                optimizer.zero_grad()
                curr_video = train_data.perform_transform(videos[i])
                # print("current video shape: ", curr_video.shape)
                curr_prompts = [prompts[i]]
                curr_n_prompts = [n_prompts[i]]

                with torch.no_grad():
                    curr_text_embeddings = encode_prompts(
                        curr_prompts,
                        curr_n_prompts,
                        self.tokenizer,
                        self.text_encoder,
                        self.device
                    )
                curr_text_embeddings = curr_text_embeddings[1].unsqueeze(0)
                curr_text_embeddings = curr_text_embeddings.to(self.device, dtype=torch.float16)

                curr_video = torch.from_numpy(curr_video)
                curr_video = torch.FloatTensor(curr_video.unsqueeze(0).float()).to(self.device)
                curr_video = preprocess_video(curr_video)

                video_length = curr_video.shape[1]
                curr_video = curr_video.to(self.device, dtype=torch.float16)
                latents = encode_video(curr_video, self.vae, video_length, self.device)

                noise = torch.randn_like(latents)

                b, c, f, h, w = latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(0,
                                          self.scheduler.config.num_train_timesteps,
                                          (b,),
                                          device=self.device)
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

                noisy_latents = noisy_latents.to(self.device)
                self.unet = self.unet.to(self.device)

                model_pred = self.unet(noisy_latents, timesteps, curr_text_embeddings).sample

                # Get the target for loss depending on the prediction type
                if self.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.scheduler.config.prediction_type == "v_prediction":
                    target = self.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        self.unet = self.unet.to(torch.float32)

    def save_lora(self, path):
        torch.save(self.unet.attn_processors, path)


def check_optimizer_gradients(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                max_grad = torch.max(torch.abs(param.grad))
                if torch.isnan(max_grad) or torch.isinf(max_grad):
                    print('Gradient has NaN or Inf values')
                else:
                    print(f'Max absolute gradient: {max_grad.item()}')
