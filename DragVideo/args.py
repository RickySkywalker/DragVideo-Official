VIDEO_LEN = 12
VIDEO_NAME = 'tmp'
DEVICE_ID = 0

class UI_args():
    def __init__(self):
        self.prompt = "best quality"
        self.lora_path = f"./results/{VIDEO_NAME}/lora.pt"
        self.save_path = f"./results/{VIDEO_NAME}"
        self.drag_batch = 12
        self.max_len = VIDEO_LEN
        self.lam = 0.1
        self.inv_batch = 12
        self.lr = 0.01
        self.drag_steps = 40
        self.guidance_scale = 1.0
        self.kps = 6
        self.pretrained_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.window_size = 480
        self.dreambooth_model_list = [
            "realisticVisionV60B1_v51VAE.safetensors",
            "majicmixRealistic_v4.safetensors",
            "leosamsFilmgirlUltra_velvia20Lora.safetensors",
            "toonyou_beta3.safetensors",
            "majicmixRealistic_v5Preview.safetensors",
            "rcnzCartoon3d_v10.safetensors",
            "lyriel_v16.safetensors",
            "leosamsHelloworldXL_filmGrain20.safetensors",
            "TUSUN.safetensors",
        ]

class Drag_args():
    def __init__(self):
        self.pretrained_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        # self.motion_module = "AnimateDiff/models/Motion_Module/mm_sd_v15_v2.ckpt"
        self.motion_module = "AnimateDiff/models/Motion_Module/v3_sd15_mm.ckpt"
        self.dreambooth_model_path = "realisticVisionV60B1_v51VAE.safetensors"
        self.lora_model_path = ""
        self.lora_alpha = 0.6
        self.video_name = VIDEO_NAME
        self.L = VIDEO_LEN
        self.W = 512
        self.H = 512

        self.n_inference_steps = 49
        self.n_actual_steps = 40
        self.inversion_strength = 0.75
        self.unet_feature_idx = [2]  # [2, 3]
        self.sup_res = 256
        self.r_m = 1
        self.r_p = 3
        self.r_d = 4
        self.lam = 0.1
        self.lr = 0.05
        self.n_pix_step = 40        # Drag Steps

        self.msa = True

class LoRA_args():
    def __init__(self):
        self.epoch = 100
        self.lr = 1E-4
        self.betas=(0.9, 0.999)
        self.weight_decay=1E-2
        self.eps=1E-8,
        self.len_per_slice=16

ui_args = UI_args()
drag_args = Drag_args()
lora_args = LoRA_args()