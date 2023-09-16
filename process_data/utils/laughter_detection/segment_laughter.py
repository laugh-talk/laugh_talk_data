# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
import shutil
from tqdm import tqdm
import tgt
sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
import glob
import cv2
import pdb
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool

def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                    int(sec), int(end))

sample_rate = 8000

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--threshold', type=str, default='0.2')
parser.add_argument('--min_length', type=str, default='0.3')
parser.add_argument('--out_dir', type=str, default=None, help='Path for outputs')
parser.add_argument('--videoFolder', type=str, default=None, help='Path for inputs')
parser.add_argument('--save_to_audio_files', type=str, default='True')
parser.add_argument('--start_cnt', type=int, default=0)
parser.add_argument('--origin_VideoFolder', type=str, default="/local_data/urp1/data/242_data/final_process/processed_talk", help='Path for inputs, tmps and outputs')

args = parser.parse_args()

args.out_dir = os.path.join(args.videoFolder,"processed_data")
args.videoFolder = os.path.join(args.videoFolder,"processed_crop")
os.makedirs(args.out_dir,exist_ok=True)

model_path = args.model_path
config = configs.CONFIG_MAP[args.config]

threshold = float(args.threshold)
min_length = float(args.min_length)
save_to_audio_files = bool(strtobool(args.save_to_audio_files))

output_dir = args.out_dir

crop_time = {}
device = 'cpu'

model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
feature_fn = config['feature_fn']
model.set_device(device)

if os.path.exists(model_path):
    torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")
audio_dir = "./extract_audio/"
os.makedirs(audio_dir, exist_ok=True)
cel_text = glob.glob(args.videoFolder + '/*.mp4')
cnt = 0

for video_path in cel_text:
    cnt +=1

    if cnt<args.start_cnt:
       continue
    path_len = len(args.videoFolder) + 1
    video_name = video_path[path_len:-4]


    audio_path = audio_dir + video_name + ".wav" 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = length/fps
    crop_time[video_name]= {'time': []}
    time = crop_time[video_name]
    time[video_name] = video_name

    retval, frame = cap.read()
    if retval == False:
        continue

    while os.path.exists(audio_path) == False:
        try:
            cmd = f"ffmpeg -i {video_path} {audio_path}"
            os.system(cmd)
            if os.path.exists(audio_path)==False:
                continue

        except:
            continue

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)

    torch.backends.cudnn.enabled = False
    ##### Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs)/float(file_length)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(args.min_length), fps=fps)

    move = False

    if len(instances) > 0:
        full_res_y, full_res_sr = librosa.load(audio_path,sr=44100)
        wav_paths = []

        if save_to_audio_files:
            if output_dir is None:
                raise Exception("Need to specify an output directory to save audio files")
            else:
                for index, instance in enumerate(instances):
                    try:
                        
                        if duration > 10:
                            if instance[1] - instance[0] > 4:
                                move == True
                                out_path = output_dir + "/" + video_name + ".mp4" 
                                cnt_n = 0
                                while os.path.exists(out_path):
                                    out_path = os.path.join(output_dir, video_name + str(cnt) + ".mp4")
                                    cnt_n+=1
                                cmd = f"ffmpeg -i {video_path} -ss {secs_to_timestr(instance[0])} -to {secs_to_timestr(instance[1])} -loglevel error {out_path}"
                                os.system(cmd)
                                time.append((instance[0], instance[1]))
                        else:
                            if instance[1] - instance[0] > 2:
                                move = True
                    except:
                        continue
    if move == True and duration < 10:
        shutil.copy(video_path, output_dir + "/" + video_name + ".mp4")

    os.remove(audio_path)
shutil.rmtree(args.videoFolder)

