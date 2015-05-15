EdVidParser
======

Usage
-----

To analyze a video, do the following steps:

1. Setup the configuration file for the video directory where you want the video file to be saved.
2. Download the video with
```
python download_video.py --url YOUTUBE_VIDEO_URL
```
The video will be saves as `ID-YOUTUBE_ID.mp4` and saved into the video directory you specified in Step 1.
3. Once the video has been downloaded, analyze the video with
```
python vid_from_file.py --filename PATH_TO_VIDEO
```
The final JSON will be saved in the video directory you specified in a folder named `jsons`.



Installation requirements
-----
Required tools:

* youtube-dl >= 2014.09.09
* ffmpeg >= 2.3.3

Coming soon: A vagrant image