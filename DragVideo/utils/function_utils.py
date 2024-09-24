import torch
import torchvision.transforms as transforms
from einops import rearrange

# This function will encode the batched prompts and n_prompts, return the embeddings of them
def encode_prompts(prompts, n_prompts, tokenizer, text_encoder, device):
    # make sure all the prompts and negative prompts are in same length
    assert len(prompts) == len(n_prompts)
    batch_size = len(prompts)

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

    text_embeddings = text_encoder(
        text_input_ids.to(device)
    ).last_hidden_state

    max_length = text_input_ids.shape[-1]
    uncond_input = tokenizer(
        n_prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    uncond_embeddings = text_encoder(
        uncond_input.to(device)
    ).last_hidden_state

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def preprocess_video(videos):
    videos = videos / 255
    videos = rearrange(videos, "b f h w c -> b f c h w")
    transform = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    # print(f"video shape for preprocess: {videos.shape}")
    for i in range(len(videos)):
        videos[i] = transform(videos[i])
    videos = rearrange(videos, "b f c h w -> b f h w c")
    return videos


def encode_video(video, vae, video_len, device):
    pixel_val = rearrange(video.to(device), "b f h w c -> (b f) c h w")
    latents = vae.encode(pixel_val).latent_dist
    latents = latents.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_len)
    latents *= vae.config.scaling_factor
    return latents