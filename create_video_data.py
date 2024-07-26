import glob
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def process_video_dir(video_dir_output):
    video_dir, output_file = video_dir_output
    output_json_path = f"{output_file}/{os.path.basename(video_dir)}.json"

    if os.path.exists(output_json_path):
        logger.info(f'{output_json_path} already exists. Skipping...')
        return

    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    logger.info(f'Found {len(video_files)} video files from {video_dir}')

    json_dict = {}

    for i, video_file in enumerate(video_files):
        annotation_file = video_file.replace('.mp4', '.txt')
        caption = ""
        if os.path.exists(annotation_file):
            with open(annotation_file) as f:
                caption = f.readline().rstrip()
        json_dict[i] = {'path': os.path.basename(video_file), 'cap': caption, 'frame_idx': None}

    with open(output_json_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    logger.info(f'Written results to {output_json_path}')


def create_video_data_file(root_dir, output_file, num_workers=8):
    logger.info(f'Starting to create video data file from root directory: {root_dir}')
    video_dirs = glob.glob(os.path.join(root_dir, "*"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(executor.map(process_video_dir, [(video_dir, output_file) for video_dir in video_dirs]),
                 total=len(video_dirs)))


# 예시 사용
root_directory = '/moai/dataset/videos/videos/webvid/train_data/videos'  # 비디오 파일들이 있는 루트 디렉터리 경로
output_dir_name = '/moai/dataset/videos/videos/webvid/train_data/captions'

os.makedirs(output_dir_name, exist_ok=True)

logger.info('Creating video data file...')
create_video_data_file(root_directory, output_dir_name, num_workers=8)
logger.info('Video data file created successfully.')
