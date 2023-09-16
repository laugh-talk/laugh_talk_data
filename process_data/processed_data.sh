SHELL_PATH=`pwd -P`
echo $SHELL_PATH

read -p "Video folder path : " VideoFolder


cd utils/TalkNet_ASD
python ./demo_talk.py --videoFolder $VideoFolder --outdir $SHELL_PATH
cd ..
python ./scendetect_data.py --videoFolder $SHELL_PATH
python ./scen_crop.py --videoFolder $SHELL_PATH
cd ./laughter_detection
python ./segment_laughter.py --videoFolder $SHELL_PATH