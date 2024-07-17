from collections import defaultdict
import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing

def random_video_noise(t, c, h, w):
    """
    Generate random noise video.

    Parameters:
    t (int): Number of frames.
    c (int): Number of channels.
    h (int): Height of the video.
    w (int): Width of the video.

    Returns:
    torch.Tensor: Random noise video tensor.
    """
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        """
        Initialize the dataset.

        Parameters:
        args (Namespace): Arguments containing dataset configurations.
        transform (callable): A function/transform to apply on the video frames.
        temporal_sample (callable): A function to sample frames from the video.
        tokenizer (callable): A function to tokenize text.
        """
        self.image_data = args.image_data
        self.video_data = args.video_data
        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.v_decoder = DecordInit()

        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                self.img_cap_list = self.get_img_cap_list()
        else:
            self.img_cap_list = self.get_img_cap_list()

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        if self.num_frames != 1:
            return len(self.vid_cap_list)
        else:
            return len(self.img_cap_list)
        
    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Parameters:
        idx (int): Index of the item to retrieve.

        Returns:
        dict: Dictionary containing video data and image data.
        """
        try:
            video_data, image_data = {}, {}
            if self.num_frames != 1:
                video_data = self.get_video(idx)
                if self.use_image_num != 0:
                    if self.use_img_from_vid:
                        image_data = self.get_image_from_video(video_data)
                    else:
                        image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            return dict(video_data=video_data, image_data=image_data)
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video(self, idx):
        """
        Retrieve and process video from the dataset.

        Parameters:
        idx (int): Index of the video to retrieve.

        Returns:
        dict: Dictionary containing processed video and tokenized text.
        """
        video_path = self.vid_cap_list[idx]['path']
        frame_idx = self.vid_cap_list[idx]['frame_idx']
        video = self.decord_read(video_path, frame_idx)
        video = self.transform(video)  # T C H W -> T C H W

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = self.vid_cap_list[idx]['cap']

        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image_from_video(self, video_data):
        """
        Extract image frames from video data.

        Parameters:
        video_data (dict): Dictionary containing video data.

        Returns:
        dict: Dictionary containing extracted image frames and tokenized text.
        """
        select_image_idx = np.linspace(0, self.num_frames-1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i+1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        """
        Retrieve and process image from the dataset.

        Parameters:
        idx (int): Index of the image to retrieve.

        Returns:
        dict: Dictionary containing processed image and tokenized text.
        """
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        
        image = [Image.open(i['path']).convert('RGB') for i in image_data]  # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image]  # num_img [h, w, c]
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image]  # num_img [1 c h w]
        image = [self.transform(i) for i in image]  # num_img [1 C H W] -> num_img [1 C H W]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] for i in image_data]
        text = [text_preprocessing(cap) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids.append(text_tokens_and_mask['input_ids'])
            cond_mask.append(text_tokens_and_mask['attention_mask'])
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def tv_read(self, path, frame_idx=None):
        """
        Read video using torchvision.

        Parameters:
        path (str): Path to the video file.
        frame_idx (str): Frame indices to read (optional).

        Returns:
        torch.Tensor: Video tensor.
        """
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)

        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

        video = vframes[frame_indice]  # (T, C, H, W)

        return video
    
    def decord_read(self, path, frame_idx=None):
        """
        Read video using Decord.

        Parameters:
        path (str): Path to the video file.
        frame_idx (str): Frame indices to read (optional).

        Returns:
        torch.Tensor: Video tensor.
        """
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)

        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames

        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def get_vid_cap_list(self):
        """
        Build a list of video paths and captions from the dataset.

        Returns:
        list: List of dictionaries containing video paths and captions.
        """
        vid_cap_lists = []
        with open(self.video_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]

        # Case 1 (single-caption):
        # each line of .txt file is : "mp4 path, caption path"
        if folder_anno[0][1].endswith('txt'):
            for path, anno in tqdm(folder_anno):
                with open(anno, 'r') as f:
                    vid_cap = f.readline().rstrip()

                vid_cap_list = defaultdict(dict)
                vid_cap_list[0]['path'] = path
                vid_cap_list[0]['cap'] = vid_cap
                vid_cap_list[0]['frame_idx'] = None 

                vid_cap_lists.append(vid_cap_list)
        
        # Case 2 (multi-captions, OpenSora style): mp4 path, caption path
        # 
        elif folder_anno[0][1].endswith('json'):
            for folder, anno in folder_anno: 
                with open(anno, 'r') as f:
                    vid_cap_list = json.load(f)
                
                print(f'Building {anno}...')
                
                for i in tqdm(range(len(vid_cap_list))):
                    path = opj(folder, vid_cap_list[str(i)]['path'])
                    if os.path.exists(path.replace('.mp4', '_resize_1080p.mp4')):
                        path = path.replace('.mp4', '_resize_1080p.mp4')
                    vid_cap_list[str(i)]['path'] = path
                vid_cap_lists += list(vid_cap_list.values())
        
        return vid_cap_lists

    def get_img_cap_list(self):
        """
        Build a list of image paths and captions from the dataset.

        Returns:
        list: List of dictionaries containing image paths and captions.
        """
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        img_cap_lists = []
        with open(self.image_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                img_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(img_cap_list))):
                img_cap_list[i]['path'] = opj(folder, img_cap_list[i]['path'])
            img_cap_lists += img_cap_list
        img_cap_lists = [img_cap_lists[i: i+use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        return img_cap_lists[:-1]  # drop last to avoid error length
