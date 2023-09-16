from scenedetect import detect, ContentDetector, split_video_ffmpeg
import shutil
import os
import glob
import cv2
import argparse

parser = argparse.ArgumentParser(description="Scendetect")

parser.add_argument('--videoFolder', type=str, default="/local_data/urp1/data/242_data/final_process/processed_talk", help='Path for inputs and outputs')
args = parser.parse_args()
args.videoFolder = os.path.join(args.videoFolder,"processed_talk")

def process_ffmpeg_actions(raw_vid_path, save_folder, save_vid_name, time_period):
    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))


    def compute_time(t):
        start_h, start_m, start_s = t[0].split(":")
        end_h, end_m, end_s = t[1].split(":")
        start_sec = int(start_h)*3600 + int(start_m)*60 + int(start_s) + 0.5
        end_sec = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)

        return start_sec, end_sec

    # seg ['laugh', ['0:05:16', '0:05:17', 1, 0]]
    for seg in time_period:

        out_path = os.path.join(out_dir, save_vid_name)
        cnt=0
        while os.path.exists(out_path):
            save_vid_name.split(".")[0]
            out_path = os.path.join(out_dir, save_vid_name.split(".")[0]+ str(cnt) + ".mp4")
            cnt += 1
        start_sec, end_sec = seg
        cmd = f"ffmpeg -i {raw_vid_path} -ss {start_sec} -to {end_sec} -loglevel error {out_path}"
        os.system(cmd)
        cap = cv2.VideoCapture(out_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps == 0:
            os.remove(out_path)
        else:
            duration = length/fps 
            if duration < 4:
                os.remove(out_path)
# ---------------------------------------------------------
cel_text = glob.glob(args.videoFolder + '/*.mp4')

out_dir = args.videoFolder
os.makedirs(out_dir, exist_ok=True)
path_len = len(out_dir) + 1

for video_name in cel_text:
    scene_list = detect(video_name, ContentDetector())
    save_vid_name = video_name[path_len:]
    
    if len(scene_list) == 0:
        continue
    else:
        process_ffmpeg_actions(video_name, out_dir, save_vid_name, scene_list)
        os.remove(video_name)
