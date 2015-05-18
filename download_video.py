import argparse
import os
import os.path
import subprocess
import sys

# check if video exists
# download the video
# expects command-line argument "--url YOUTUBE_URL"

# youtube_url is the URL of the video on youtube
# vid_dir is the directory the video will be saved in (should end in "/")
# vid_id is the 11-character ID number for videos on youtube
def download_video(youtube_url, vid_dir, vid_id):
	try:
		vid_filename = vid_dir + 'ID-' + vid_id + '.mp4'
		if not os.path.isfile(vid_filename):
			subprocess.call(["youtube-dl", "-o", vid_filename, youtube_url])
		else:
			print "Already exists. Not downloading again."
	except:
		print "There was a problem, most likely with the URL. The video might not exist anymore."
		sys.exit(0)

# for now, a hacky way of finding the video ID
def url_to_vid_id(youtube_url):
	try:
		vid_id = youtube_url[-11:]
		return vid_id
	except:
		print "Something went wrong. The URL is probably in the wrong format. Exiting."
		sys.exit(0)

if __name__ == '__main__':
	# parse the command-line arguments
	parser = argparse.ArgumentParser("Download a video from youtube and save it")
	parser.add_argument('--url', metavar="url", type=str)

	args = parser.parse_args()

	if not args.url:
		print "Must provide a URL. Exiting."
		sys.exit(0)
	else:
		youtube_url = args.url

	vid_id = url_to_vid_id(youtube_url)

	# video directory is always vids/
	vid_dir = 'vids/'


	download_video(youtube_url, vid_dir, vid_id)