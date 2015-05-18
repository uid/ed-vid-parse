EdVidParser
======

Installation
-----

Vagrant
---
To use the provided Vagrantfile, install VirtualBox and Vagrant. If you want to install the dependencies on your own machine, see below.

The `Vagrantfile` in the repository has an automatic install script for all the software packages and dependencies on an Ubuntu 14.04 machine using VirtualBox. To use, simply
```
vagrant up
```
And wait for it to finish. It should take about 30 minutes. Once it is finished, you can use your machine with 
```
vagrant ssh
```

See the Vagrant docs for more details.

Local machine install
---
Coming soon.


Usage
-----

If using Vagrant, do: 
```
vagrant ssh
cd /vagrant/
```

To analyze a video:

1. Setup the configuration file (`config.yml`) for the video directory where you want the video file to be saved.
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