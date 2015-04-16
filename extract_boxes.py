import scipy
import numpy as np
from scipy import io
import os
import os.path
import re
import argparse
import sys
import pprint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

pp = pprint.PrettyPrinter(indent=2)

# add path of the vidutils directory
# sys.path.append('../vidutils/python')
# import vidutils

# include caffe
# caffe_root = '/home/ubuntu/caffe/'
# repo_root = '/home/ubuntu/vid-to-json-deploy/'
# model_root = '/home/ubuntu/models/'
caffe_root = '/Users/mprat/Documents/repos/caffe/'
repo_root = '/Users/mprat/Documents/repos/vid-to-json-deploy/'
model_root = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/placesCNN/'
weights_root = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/conv5_weights/'
vid_dir = '/Users/mprat/Desktop/test_vids/'

people_weightfile_name = 'places-people-train-activ-target-3.mat'
people_weight_data = scipy.io.loadmat(weights_root + people_weightfile_name, squeeze_me=True)
people_weight_data = people_weight_data['weights']
nonzero_people_weights = np.nonzero(people_weight_data)[0]

thresh = 0.01

sys.path.insert(0, caffe_root + 'python')
import caffe

# # 
# # Command-line argument Parser
# # 
# parser = argparse.ArgumentParser(description='Turn the frame-level video classification into a JSON file.')
# parser.add_argument('--id', metavar="videoID", type=str)
# parser.add_argument('--url', metavar="videoURL", type=str)
# # parser.add_argument('--ID', dest="id_given", const=True, default=False, help="Use this flag if you just want to use the youtube video ID.")

# args = parser.parse_args()

# # can either specify a command-line argument or type the name / ID of a video here
# if not args.id and not args.url:
# 	vid_name = "ID-EMaTF9-ArJY"
# 	vid_url = "http://www.youtube.com/watch?v=" + vid_name[3:]
# elif args.url:
# 	vid_name = vidutils.get_id_from_url(args.url)
# 	# vid_name = "ID-" + args.url[31:] #if there is a www (if there is NO www, then should be 27)
# elif args.id:
# 	vid_name = args.id
# else:
# 	print "Something horrible happened."
# 	sys.exit(0)

# 
# Variables needed for this task
#
# vid_dir = '/Users/mprat/Desktop/test_vids/' # directory where the videos live
# vid_dir = '/data/vision/torralba/mooc-video/videos/'
# data_dir = '../data/' # directory where the .mat files live

# caffe stuff
MODEL_FILE = model_root + 'places205CNN_deploy_FC7_upgraded_one.prototxt'
CAFFEMODEL_FILE = model_root + 'places205CNN_iter_300000_upgraded.caffemodel'
MEAN_FILE = model_root + 'ilsvrc_2012_mean.npy'

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, CAFFEMODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1)) # channels, then width, then height
transformer.set_mean('data', np.load(model_root + 'ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# net.blobs['data'].data[...]

# let's assume you start with a folder of images
# read one image at a time, compute it
# show next one, etc.

vid_name = "ID-EMaTF9-ArJY-temp"
image_folder = vid_dir + vid_name + "/"
num_imgs = 150

# plt.subplot(1, 2, 1);
# plt.subplot(1, 2, 2);
# img_axis = plt.gca()
# bounding_box_patch = Rectangle((0, 0), 0, 0, fill=None, alpha=1)
# img_axis.add_patch(bounding_box_patch)
plt.ion()
plt.show(block=False)

for i in range(1,num_imgs+1):
	filename = image_folder + "all_{:08d}.png".format(i)
	print filename
	# input the data
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
	# send it through the net
	out = net.forward()

	# extract the conv5 features
	conv5_features = net.blobs['conv5'].data # the shape is (1, 256, 13, 13)
	conv5_features = conv5_features[0]

	final_mask = np.zeros((13, 13))

	# multiply them by the weights
	for nonzero_index in nonzero_people_weights:
		final_mask += conv5_features[nonzero_index] * people_weight_data[nonzero_index]


	# transform to heatmap
	heatmap = scipy.ndimage.zoom(final_mask, 227/13.0, order=1)
	# plt.subplot(1, 2, 1)
	plt.imshow(heatmap)

	# transform to bounding box
	[y_index, x_index] = np.where(heatmap > thresh)
	if (len(y_index) > 0 and len(x_index) > 0):
		box = [min(x_index), min(y_index), max(x_index), max(y_index)]
		box[2] = box[2] - box[0]
		box[3] = box[3] - box[1]
		# bounding_box_patch.set_bounds(box[0], box[1]+box[3], box[2], -1*box[3])
		print box

	# show the original image
	# plt.subplot(1, 2, 2)
	# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
	plt.draw()