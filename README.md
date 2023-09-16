# Laugh Talk Data

# Dependencies
1. Build the environment
```
conda create -n LaughTalk python=3.7.9
conda activate LaughTalk
```
2. Clone this repo
```
git clone https://github.com/laugh-talk/Data.git
```
3. Install required python packages
```
cd Data
pip install -r requirements.txt
```
# Data
There are two dataset. 
The 504 dataset comprises two types of video recordings: one featuring a person's upper body engaged in conversation with a talk of laughter, and the other featuring a person's upper body engaged in talk while displaying a smiling expression without laughter.
The 292 dataset consists of the former type video.
We use 292 dataset to train and test model.

### JSON File Structure:
```
{
    "9j9nei1ECv0_7.mp4": 
    {
        "meta_video_name": "9j9nei1ECv0_7", 
        "vid_id": "9j9nei1ECv0",                                                        // youtube id
        "time": [["laugh", ["00:02:36.56", "00:02:42.53"]]],                            // action class, start and end times
        "bbox": {"top": 0.0, "bottom": 0.8889, "left": 0.2789, "right": 0.7789},        // bounding box
        "version": "v0.2"
        },
    "cuGs0oSqpBE_22.mp4":
     {
        "meta_video_name": "cuGs0oSqpBE_22", 
        "vid_id": "cuGs0oSqpBE", 
        "time": [["talk", ["00:03:38.34", "00:03:44.18"]]], 
        "bbox": {"top": 0.0, "bottom": 0.7278, "left": 0.3664, "right": 0.7758}, 
        "version": "v0.2"
        }
        "...",
        "..."

}
```
### Download Data
If you want to download 292 dataset, run the following
```
python download_vid.py --json_path "./laughtalk_292_data_info.json"
```

If you want to download 504 dataset, run the following
```
python download_vid.py --json_path "./laughtalk_504_data_info.json"
```

# Data Curation
Our data undergoes a four-stage preprocessing procedure:

1. we filter video to include only those containing active speakers, using TalkNet.
2. we split video at scene transitions.
3. we filter out video with excessive head movement or incomplete facial capture on the screen.
4. we leave video where laughter lasts for more than 0.1 seconds, utilizing laugh_detection.

If you want to run this preprocessing procedure, your custom dataset must consist of video of the upper body of a person.

### Using Your Own Dataset

If you wish to run a four-stage preprocessing procedure on your custom dataset, run the following
```
sh process_data/processed_data.sh
```
And enter the location of your dataset.
```
Video folder path : (Enter your dataset path)
```

# Citation
We use [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD/blob/main/README.md) for data preprocessing, please further cite:
```
@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {3927–3935},
  year={2021}
}
```

# Acknowledgement
We acknowledge the use of [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) for filtering videos with active speakers during the data preprocessing stage. For the preprocessing of videos containing laughter, we use of [Laughter-detection](https://github.com/jrgillick/laughter-detection).
We express our sincere gratitude to the authors for open-sourcing their code and sharing their excellent works.