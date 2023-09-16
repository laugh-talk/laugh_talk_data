
"""
Downloader
"""

import os
import json

import cv2
import argparse


def download(video_path, ytb_id, proxy=None):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, defalut None
    """
    if proxy is not None:
        proxy_cmd = "--proxy {}".format(proxy)
    else:
        proxy_cmd = ""
    if not os.path.exists(video_path):
        down_video = " ".join([
            "yt-dlp",
            proxy_cmd,
            '-f', "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
            '--skip-unavailable-fragments',
            '--merge-output-format', 'mp4',
            "https://www.youtube.com/watch?v=" + ytb_id, "--output",
            video_path, "--external-downloader", "aria2c",
            "--external-downloader-args", '"-x 16 -k 1M"'
        ])
        print(down_video)
        status = os.system(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")
            download(raw_vid_path, vid_id, proxy)


def process_ffmpeg_actions(raw_vid_path, save_folder, save_vid_name, bbox, vid_actions):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    time: begin_sec, end_sec
    """
    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)

        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, leftx, right = bbox
        h = bottom - top
        w = right - leftx
        c = min(h, w) // 2
        c_h = (top + bottom) / 2
        c_w = (leftx + right) / 2

        top, bottom = c_h - c, c_h + c
        leftx, right = c_w - c, c_w + c
        return top, bottom, leftx, right

    def denorm(bbox, height, width):
        top = round(bbox[0] * height)
        bottom = round(bbox[1] * height)
        left = round(bbox[2] * width)
        right = round(bbox[3] * width)
        return top, bottom, left, right

    def compute_time(t):
        start_h, start_m, start_s = t[0].split(":")
        end_h, end_m, end_s = t[1].split(":")
        start_sec = int(start_h)*3600 + int(start_m)*60 + float(start_s) 
        end_sec = int(end_h) * 3600 + int(end_m) * 60 + float(end_s)

        return start_sec, end_sec

    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(denorm(expand(bbox, 0.02), height, width))

    # seg ['laugh', ['0:05:16', '0:05:17']]

    for seg in vid_actions:
        actions_class = seg[0]
        out_dir = os.path.join(save_folder,actions_class)
        os.makedirs(out_dir,exist_ok=True)
        out_path = os.path.join(out_dir, save_vid_name)

        cnt=0
        while os.path.exists(out_path):
            save_vid_name.split(".")[0]
            out_path = os.path.join(out_dir, save_vid_name.split(".")[0]+ str(cnt) + ".mp4")
            cnt += 1
        start_sec, end_sec = compute_time(seg[1])
        cmd = f"/home/sung/ffmpeg-git-20230621-amd64-static/ffmpeg -i {raw_vid_path} -vf crop=w={right - left}:h={bottom - top}:x={left}:y={top} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
        os.system(cmd)
    return out_path

def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    #for key, val in data_dict['clips_set1'].items():
    for key, val in data_dict.items():
        #save_name = key + ".mp4"
        save_name=key
        ytb_id = val['vid_id']
        time = val['time']
        bbox = [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]
        yield ytb_id, save_name, time, bbox



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="download_vid")
    parser.add_argument('--json_path', type=str, default="/local_data/urp1/aaa/data/laughtalk_290_data_info.json", help='Path for json')
    args = parser.parse_args()

    json_path = args.json_path  # json file path
    raw_vid_root = './downloaded_data/raw/'  # download raw video path
    #---------------------------------------------
    processed_vid_root = './downloaded_data/processed_data/'  # processed video path

    proxy = None  # proxy url example, set to None if not use

    os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(processed_vid_root, exist_ok=True)

    for vid_id, save_vid_name, time, bbox in load_data(json_path):
        raw_vid_path = os.path.join(raw_vid_root, vid_id + ".mp4")

        # Downloading is io bounded and processing is cpu bounded.
        # It is better to download all videos firstly and then process them via multiple cpu cores.
    
        download(raw_vid_path, vid_id, proxy)
        process_ffmpeg_actions(raw_vid_path, processed_vid_root, save_vid_name, bbox, time)
        os.remove(raw_vid_path)
 
