import numpy as np
import cv2
import sys
import scipy
from scipy import io
import skimage
from skimage import transform
import json
import pprint
import os
import subprocess
import re
import math
import argparse

# extract people and content from the video
# expects command-line argument "--filename FILENAME"

pp = pprint.PrettyPrinter(indent=2)

display = False
show_score = True
one_mask_size = 55
crop_size = 227

caffe_root = '/home/vagrant/caffe/'
repo_root = '/vagrant/'
model_root = '/vagrant/models/'
vid_dir = '/vagrant/vids/'
json_dir = 'jsons/'

person_net_name = 'places'
person_neuron_nums = [1511, 1731] #, 1927, 1844, 1606, 1979, 910, 256]
person_weight_data = [0.0057, 0.0212] #, 0.0160, 0.0175, 0.0145, -0.00125, -0.0057, -0.00325, -0.008]
person_heatmap_thresh = 0.03
person_score_thresh = 7000
person_num_boxes = 10

content_net_name = 'places'
content_neuron_nums = [1606, 1979, 910, 256]
content_weight_data = [0.0025, 0.0091, 0.0063, 0.016]
content_heatmap_thresh = 0.02
content_score_thresh = 8000
content_num_boxes = 10

sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe stuff
HYBRIDPLACES_MODEL_FILE = model_root + 'hybridCNN/hybridCNN_deploy_FC7_updated_one.prototxt'
HYBRIDPLACES_CAFFEMODEL_FILE = model_root + 'hybridCNN/hybridCNN_iter_700000_upgraded.caffemodel'

PLACES_MODEL_FILE = model_root + 'placesCNN/places205CNN_deploy_FC7_upgraded_one.prototxt'
PLACES_CAFFEMODEL_FILE = model_root + 'placesCNN/places205CNN_iter_300000_upgraded.caffemodel'
MEAN_FILE = model_root + 'placesCNN/ilsvrc_2012_mean.npy'

# PLACES_TUNED_MODEL_FILE = model_root + 'placesCNN/places205CNN_deploy_FC7_upgraded_one.prototxt'
# PLACES_TUNED_CAFFEMODEL_FILE = model_root + 'placesCNN/finetune_ed_scenes_iter_100000.caffemodel'

caffe.set_mode_cpu()

hybridplaces_net = caffe.Net(HYBRIDPLACES_MODEL_FILE, HYBRIDPLACES_CAFFEMODEL_FILE, caffe.TEST)
places_net = caffe.Net(PLACES_MODEL_FILE, PLACES_CAFFEMODEL_FILE, caffe.TEST)
# places_tuned_net = caffe.Net(PLACES_TUNED_MODEL_FILE, PLACES_TUNED_CAFFEMODEL_FILE, caffe.TEST)

model_mean = np.load(repo_root + 'ilsvrc_2012_mean.npy')
model_mean = skimage.transform.resize(model_mean, (crop_size, crop_size, 3), preserve_range=True)
model_mean = model_mean.transpose((2, 0, 1))

def get_video_length_secs(vid_name, vid_dir):
	probe = subprocess.check_output(["ffprobe", "-v", "quiet",  "-show_format",  vid_dir + vid_name + ".mp4"])
	return int(float(re.search("duration=\d+.?\d+", probe).group(0)[9:]))

def neuron_index_to_layer_name(i):
	layer_name = ''
	in_layer_index = 0

	real_filter_name = '';
	if (i <= 96):
		layer_name = 'conv1'
		in_layer_index = i
		# real_filter_name = sprintf('conv1-neuron-%0.0f', i);
	elif (i <= 96+256):
		layer_name = 'conv2'
		in_layer_index = i - 96
		# real_filter_name = sprintf('conv2-neuron-%0.0f', i - 96);
	elif (i <= 96+256+384):
		layer_name = 'conv3'
		in_layer_index = i - 96 - 256
		# real_filter_name = sprintf('conv3-neuron-%0.0f', i - 96 - 256);
	elif (i <= 96+256+384+384):
		layer_name = 'conv4'
		in_layer_index = i - 96 - 256 - 384
		# real_filter_name = sprintf('conv4-neuron-%0.0f', i - 96 - 256 - 384);
	elif (i <= 96+256+384+384+256):
		layer_name = 'conv5'
		in_layer_index = i - 96 - 256 - 384 - 384
		# real_filter_name = sprintf('conv5-neuron-%0.0f', i - 96 - 256 - 384 - 384);
	elif (i <= 96+256+384+384+256+96):
		layer_name = 'pool1';
		in_layer_index = i - 96 - 256 - 384 - 384 - 256
		# real_filter_name = sprintf('pool1-neuron-%0.0f', i - 96 - 256 - 384 - 384 - 256);
	elif (i <= 96+256+384+384+256+96+256):
		layer_name = 'pool2'
		in_layer_index = i - 96 - 256 - 384 - 384 - 256 - 96
		# real_filter_name = sprintf('pool2-neuron-%0.0f', i - 96 - 256 - 384 - 384 - 256 - 96);
	elif (i <= 96+256+384+384+256+96+256+256):
		layer_name = 'pool5'
		in_layer_index = i - 96 - 256 - 384 - 384 - 256 - 96 - 256
		# real_filter_name = sprintf('pool5-neuron-%0.0f', i - 96 - 256 - 384 - 384 - 256 - 96 - 256);

	return [layer_name, in_layer_index]

def pick(boxes, overlap_thresh):
	if (boxes.shape[0] == 0):
		return []
	elif (boxes.shape[0] == 1):
		return [0]
	else:
		pick = []

		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]
		scores = boxes[:, 4]

		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		sorted_indices = np.argsort(scores)

		while len(sorted_indices) > 0:
			last = len(sorted_indices) - 1
			i = sorted_indices[last]
			pick.append(i)

			xx1 = np.maximum(x1[i], x1[sorted_indices[:last]])
			yy1 = np.maximum(y1[i], y1[sorted_indices[:last]])
			xx2 = np.minimum(x2[i], x2[sorted_indices[:last]])
			yy2 = np.minimum(y2[i], y2[sorted_indices[:last]])

			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)

			overlap = (w * h) / area[sorted_indices[:last]]

			sorted_indices = np.delete(sorted_indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

		return pick


def process_heatmap(heatmap, heatmap_thresh, num_boxes, score_thresh=0):
	heatmap_binary = np.copy(heatmap)
	heatmap_binary[heatmap >= heatmap_thresh] = 1
	heatmap_binary[heatmap < heatmap_thresh] = 0

	[L, num] = scipy.ndimage.measurements.label(heatmap_binary)

	bbs = []
	scores = []

	for i in range(1, num+1):
		one_comp = L == i
		temp_heatmap = np.zeros(np.shape(heatmap))
		temp_heatmap[one_comp] = heatmap[one_comp]

		sum_conn_comp_activ = np.sum(temp_heatmap)

		breakpoints = np.percentile(heatmap[one_comp], np.linspace(0, 50 - num_boxes, num_boxes))

		for j in range(num_boxes):
			[rows, cols] = np.where(temp_heatmap > breakpoints[j])
			single_bb = [0, 0, 0, 0, 0]
			if ((rows.shape[0] > 0) and (cols.shape[0] > 0)):
				single_bb[0] = np.min(cols)
				single_bb[1] = np.min(rows)
				single_bb[2] = np.max(cols)
				single_bb[3] = np.max(rows)
				score_relevant = np.sum(temp_heatmap[single_bb[1]:single_bb[3]][single_bb[0]:single_bb[2]])
				single_bb[4] = score_relevant*score_relevant / sum_conn_comp_activ

				# only add valid scores
				if single_bb[4] >= score_thresh:
					bbs.append(single_bb)

	bbs = np.array(bbs)

	if (len(bbs) > 0):
		picked_indices = pick(bbs, 0.5)

		scores = bbs[picked_indices, 4]
		bbs = bbs[picked_indices, 0:4]

		bbs[:,2] = bbs[:,2] - bbs[:,0]
		bbs[:,3] = bbs[:,3] - bbs[:,1]

		# valid = scores > score_thresh
		# bbs = bbs[valid, :]
		# score = scores[valid]

	return [bbs, scores]

def extract_box(frame):
	person_mask = np.zeros((one_mask_size, one_mask_size))
	content_mask = np.zeros((one_mask_size, one_mask_size))

	f = skimage.transform.resize(frame, (crop_size, crop_size, 3), preserve_range=True)
	f = f.transpose((2, 0, 1))
	f = f - model_mean;
	# input the data
	hybridplaces_net.blobs['data'].data[...] = f
	places_net.blobs['data'].data[...] = f
	# places_tuned_net.blobs['data'].data[...] = f
	# send it through the net
	out_hybridplaces = hybridplaces_net.forward()
	out_places = places_net.forward()
	# out_places_tuned = places_tuned_net.forward()

	# people
	for index in range(len(person_neuron_nums)):
		neuron_num = person_neuron_nums[index]
		weight = person_weight_data[index]

		[layer_name, in_layer_index] = neuron_index_to_layer_name(neuron_num)
		activ = places_net.blobs[layer_name].data
		activ = activ[0]

		one_activ = activ[in_layer_index]
		one_activ_scaled = scipy.ndimage.zoom(one_activ, float(one_mask_size) / one_activ.shape[0], order = 1)
		person_mask += one_activ_scaled * weight

	# content
	for index in range(len(content_neuron_nums)):
		neuron_num = content_neuron_nums[index]
		weight = content_weight_data[index]

		[layer_name, in_layer_index] = neuron_index_to_layer_name(neuron_num)
		activ = places_net.blobs[layer_name].data
		activ = activ[0]

		one_activ = activ[in_layer_index]
		one_activ_scaled = scipy.ndimage.zoom(one_activ, float(one_mask_size) / one_activ.shape[0], order = 1)
		content_mask += one_activ_scaled * weight

	# transform to heatmap
	person_heatmap = scipy.ndimage.zoom(person_mask, 227.0/one_mask_size, order=1)
	content_heatmap = scipy.ndimage.zoom(content_mask, 227.0/one_mask_size, order=1)

	[content_box, content_scores] = process_heatmap(content_heatmap, content_heatmap_thresh, content_num_boxes, content_score_thresh)
	[person_box, person_scores] = process_heatmap(person_heatmap, person_heatmap_thresh, person_num_boxes, person_score_thresh)

	if show_score:
		print "content box inside extract, ",  content_box
		print "content box scores, ",  content_scores

		print "person box inside extract, ", person_box
		print "person box scores, ", person_scores

	# plt.subplot(1, 2, 1)
	# plt.imshow(heatmap)
	if display:
		cv2.imshow('heatmap person', person_heatmap)
		cv2.imshow('heatmap content', content_heatmap)

	return (person_box, content_box)

def run_video(vid_name):
	segment_list = []
	content_prev = 0
	person_prev = 0

	cap = cv2.VideoCapture(vid_dir + vid_name + '.mp4')

	index = 0
	section_id = 0
	section_info = {"id": section_id, "start_time": 0, "end_time": 0, 
		"content": content_prev, "person": person_prev}

	while (cap.isOpened()):
		ret, frame = cap.read()

		if ret:
			person = 0
			content = 0

			cur_msec = int(math.floor(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))) # 0 is the index for getting the msec. or it's "video capture timestamp"
			frame_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
			frame_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
			print "cur_msec = ", cur_msec

			# only extract the person box every 10 frames
			# do some caffe stuff
			if index%10 == 0:
				(person_box, content_box) = extract_box(frame)
				print "Person: ", person_box
				print "Content: ", content_box
			if len(person_box) > 0:
				for j in range(person_box.shape[0]):
					one_box = person_box[j]
					cv2.rectangle(frame, (int(one_box[0]*frame.shape[1]/227.0), int(one_box[1]*frame.shape[0]/227.0)), (int(one_box[2]*frame.shape[1]/227.0), int(one_box[3]*frame.shape[0]/227.0)), (0, 255, 0))
				person = 1

			if len(content_box) > 0:
				# print content_box
				for j in range(content_box.shape[0]):
					one_box = content_box[j]
					cv2.rectangle(frame, (int(one_box[0]*frame.shape[1]/227.0), int(one_box[1]*frame.shape[0]/227.0)), (int(one_box[2]*frame.shape[1]/227.0), int(one_box[3]*frame.shape[0]/227.0)), (0, 255, 0))
				content = 1

			if display:
				cv2.imshow('frame', frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			index += 1

			if person == person_prev and content == content_prev:
				section_info["end_time"] = cur_msec
				section_info["person"] = person
				section_info["content"] = content
			elif index > 2:
				segment_list.append(section_info)
				print segment_list
				section_id += 1
				section_info = {"id": section_id, "start_time": cur_msec, "end_time": cur_msec, 
					"content": content, "person": person}

			person_prev = person
			content_prev = content
		else:
			segment_list.append(section_info)
			pp.pprint(segment_list)
			break

	cap.release()
	cv2.destroyAllWindows()

	final_dict = {"youtubeID": vid_name[3:], "length": get_video_length_secs(vid_name, vid_dir), "segments": segment_list}
	# pp.pprint(final_dict)
	return final_dict

if __name__ == '__main__':
	# vid_name = "ID-e7IErqC25nU"
	# vid_name = "ID-BvooIjkNJ24"
	# vid_name = "ID-waIE0L9vfiI"
	# vid_name = "ID-zhKN60gDjk8"
	# vid_name = "ID-bfpZRBTo5xc"
	# vid_name = "ID-bGWgqvhUfPU"
	# vid_name = "ID-BcioL4magDg"
	# vid_name = "ID-EMaTF9-ArJY"
	# vid_name = "ID-6mj9wWjAqz0"
	# vid_name = 'ID-b7KNIA4w9lE'
	# vid_name = "ID-epNXEXIFIYQ"
	# vid_name = "ID-ESKcD9x2Wrg"
	# vid_name = "ID-eu5pb97DhGs"
	# vid_name = "ID-JWWDvL9-zbk"
	# vid_name = "ID-paAXl2Ie_as"

	# parse the command-line arguments
	parser = argparse.ArgumentParser("Turn a video into a JSON with person and content annotations")
	parser.add_argument('--filename', metavar="filename", type=str)

	args = parser.parse_args()

	if not args.filename:
		print "Must provide a filename. Exiting."
		sys.exit(0)
	else:
		filename = args.filename

	vid_name = filename.split('/')[-1][:-4]

	final_dict = run_video(vid_name)
	# pp.pprint(final_dict)
	with open(json_dir + vid_name + '.json', 'w') as outfile:
		json.dump(final_dict, outfile)