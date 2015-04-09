#!/bin/bash

vid_dir='/Users/mprat/Desktop/test_vids/';
# mkdir $vid_dir;

echo "Analyzing" $1;
start=`date +%s`;
echo $start;

vid_id=${1: -11};
vid_name='ID-'$vid_id;

echo "Downloading video"

mkdir $vid_dir$vid_name;

# download video / one second per frame
youtube-dl -o $vid_dir'ID-%(id)s.%(ext)s' 'http://www.youtube.com/watch?v='$vid_id;
# ffmpeg -i $vid_dir$vid_name.mp4 -f image2 -vf fps=fps=1 $vid_dir$vid_name/image_%08d.png;
# ffmpeg -i $vid_dir$vid_name.mp4 -vf showinfo $vid_dir$vid_name/all_%08d.png 2>&1 | grep --line-buffered -o 'pts_time:[0-9]*.[0-9]*' | gsed -u -E 's/[a-z_:]*//' > $vid_dir$vid_name/exact_times.txt;
ffmpeg -i $vid_dir$vid_name.mp4 -vf showinfo -s 300x300 $vid_dir$vid_name/all_%08d.png 2>&1 | grep --line-buffered -o 'pts_time:[0-9]*.[0-9]*' | gsed -u -E 's/[a-z_:]*//' > $vid_dir$vid_name/exact_times.txt;


# id=`python download_video.py --url $1`;
# cd ../MATLAB/;
# echo "Getting features from one frame per second"
# matlab -nosplash -nodesktop -r "run vid_name_to_all_frames_features('$id'); quit;";
# cd ../python/;
# echo "Constructing JSON from one frame per second"
# python construct_json_from_prediction_labels.py --id $id;
# echo "Downloading millisecond-level accurate frames"
# python get_millis_frames.py --id $id;
# cd ../MATLAB/;
# echo "Getting features from millisecond-level accuracy"
# matlab -nosplash -nodesktop -r "run vid_name_to_millisecond_images('$id'); quit;";
# cd ../python/;
# echo "Re-constructing JSON"
# python construct_json_millis.py --id $id;

echo "Done";
end=`date +%s`
echo $(($end-$start))