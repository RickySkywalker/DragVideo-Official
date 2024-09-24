import torchvision.io as io
import numpy as np
from torchvision import transforms
import torch

def load_video(file_path, target_shape=(512, 512)):
    video_tensor, audio_tensor, video_fps = io.read_video(file_path, pts_unit="sec")
    resize = transforms.Resize((512, 512))

    result_ls = []
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    for frame in video_tensor:
        result_ls.append(resize(frame))
    video_tensor = torch.stack(result_ls)
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    assert video_tensor.shape[1:3] == target_shape

    return video_tensor, video_fps['video_fps']

class VideoData:
    def __init__(self,
                 video,
                 prompt,
                 n_prompt=None,
                 masks=-1,
                 len_per_slice=16,
                 transform=None):
        assert isinstance(video, np.ndarray) or isinstance(video, torch.Tensor), "Please input a np array or torch tensor for the video"
        if isinstance(video, torch.Tensor):
            self.video = video.cpu().numpy()
        else:
            self.video = video
        self.prompt = prompt
        if n_prompt == None:
            print("No negative prompt is used, n_prompt is empty string")
            self.n_prompt = ""
        else:
            self.n_prompt = n_prompt
        self.len_per_slice = len_per_slice
        self.transform = transform
        self.masks = masks

        if transform == None:
            self.transform = transforms.Compose([

                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def set_len_per_slice(self, len_per_slice):
        self.len_per_slice = len_per_slice

    def perform_transform(self, video):
        for i in range(len(video)):
            curr_frame = transforms.ToPILImage()(video[i])
            curr_frame = self.transform(curr_frame)

            video[i] = curr_frame.permute(1, 2, 0)
        return video

    def generate_batch(self):
        if isinstance(self.masks, int):
            if self.video.shape[0]%self.len_per_slice:
                batches = np.array_split(self.video, self.video.shape[0]//self.len_per_slice + 1, axis=0)
            else:
                batches = np.array_split(self.video, self.video.shape[0]//self.len_per_slice, axis=0)
            batch_size = len(batches)
            prompts = [self.prompt] * batch_size
            n_prompts = [self.n_prompt] * batch_size
            batch_ls = [curr for curr in batches]
            return batch_ls, prompts, n_prompts
        else:
            if self.video.shape[0]%self.len_per_slice:
                batches = np.array_split(self.video, self.video.shape[0]//self.len_per_slice + 1, axis=0)
                mask_batch = np.array_split(self.masks, self.video.shape[0]//self.len_per_slice + 1, axis=0)
            else:
                batches = np.array_split(self.video, self.video.shape[0]//self.len_per_slice, axis=0)
                mask_batch = np.array_split(self.masks, self.video.shape[0]//self.len_per_slice, axis=0)
            batch_size = len(batches)
            prompts = [self.prompt] * batch_size
            n_prompts = [self.n_prompt] * batch_size
            batch_ls = [curr for curr in batches]
            return batch_ls ,mask_batch, prompts, n_prompts