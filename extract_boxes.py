import scipy
import numpy
from scipy import io
import os
import os.path
import re
import argparse
import sys
import pprint

pp = pprint.PrettyPrinter(indent=2)

# add path of the vidutils directory
# sys.path.append('../vidutils/python')
# import vidutils

# include caffe
caffe_root = '/Users/mprat/Documents/repos/caffe-master/'
sys.path.insert(0, caffe_root + 'python')

import caffe

# 
# Command-line argument Parser
# 
parser = argparse.ArgumentParser(description='Turn the frame-level video classification into a JSON file.')
parser.add_argument('--id', metavar="videoID", type=str)
parser.add_argument('--url', metavar="videoURL", type=str)
# parser.add_argument('--ID', dest="id_given", const=True, default=False, help="Use this flag if you just want to use the youtube video ID.")

args = parser.parse_args()

# can either specify a command-line argument or type the name / ID of a video here
if not args.id and not args.url:
	vid_name = "ID-EMaTF9-ArJY"
	vid_url = "http://www.youtube.com/watch?v=" + vid_name[3:]
elif args.url:
	vid_name = vidutils.get_id_from_url(args.url)
	# vid_name = "ID-" + args.url[31:] #if there is a www (if there is NO www, then should be 27)
elif args.id:
	vid_name = args.id
else:
	print "Something horrible happened."
	sys.exit(0)


# 
# Variables needed for this task
#
vid_dir = '/Users/mprat/Desktop/test_vids/' # directory where the videos live
# vid_dir = '/data/vision/torralba/mooc-video/videos/'
# data_dir = '../data/' # directory where the .mat files live

# caffe stuff
MODEL_FILE = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/placesCNN/places205CNN_deploy_FC7_upgraded_one.prototxt'
CAFFEMODEL_FILE = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/placesCNN/places205CNN_iter_300000_upgraded.caffemodel'
MEAN_FILE = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/bvlc_caffenet/ilsvrc_2012_mean.npy'

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, CAFFEMODEL_FILE, caffe.TEST)

