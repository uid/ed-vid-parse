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

# add path of the vidutils directory
sys.path.append('../vid-to-json-cleaned/vidutils/python')
import vidutils

pp = pprint.PrettyPrinter(indent=2)

vid_name = "ID-EMaTF9-ArJY"
# vid_name = "ID-6mj9wWjAqz0"
# vid_name = 'ID-b7KNIA4w9lE'
# vid_name = "ID-epNXEXIFIYQ"
# vid_name = "ID-ESKcD9x2Wrg"
# vid_name = "ID-eu5pb97DhGs"
# vid_name = "ID-JWWDvL9-zbk"
# vid_name = "ID-paAXl2Ie_as"

display = True
one_mask_size = 55

caffe_root = '/Users/mprat/Documents/repos/caffe/'
repo_root = '/Users/mprat/Documents/repos/vid-to-json-deploy/'
model_root = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/'
weights_root = '/Users/mprat/Documents/repos/vid-to-json-cleaned/data/conv5_weights/'
vid_dir = '/Users/mprat/Desktop/test_vids/'

person_net_name = 'places'
person_net_neuron_param = 2
people_weightfile_name = person_net_name + '-person-train-activ-target-' + str(person_net_neuron_param) + '.mat'
people_weight_data = scipy.io.loadmat(weights_root + people_weightfile_name, squeeze_me=True)
people_weight_data = people_weight_data['weights']
nonzero_people_weights = np.nonzero(people_weight_data)[0]
people_thresh = 0.1

content_net_name = 'places'
# content_net_neuron_param = 1
# content_weightfile_name = content_net_name + '-content-train-activ-target-' + str(content_net_neuron_param) + '.mat'
# content_weight_data = scipy.io.loadmat(weights_root + content_weightfile_name, squeeze_me=True)
# content_weight_data = content_weight_data['weights']
# nonzero_content_weights = np.nonzero(content_weight_data)[0]
content_neuron_nums = [1606, 1979, 910]
content_weight_data = [0.0025, 0.0114, 0.0065]
content_heatmap_thresh = 0.01
content_score_thresh = 1000
content_num_boxes = 10

sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe stuff
HYBRIDPLACES_MODEL_FILE = model_root + 'hybridCNN/hybridCNN_deploy_FC7_updated_one.prototxt'
HYBRIDPLACES_CAFFEMODEL_FILE = model_root + 'hybridCNN/hybridCNN_iter_700000_upgraded.caffemodel'

PLACES_MODEL_FILE = model_root + 'placesCNN/places205CNN_deploy_FC7_upgraded_one.prototxt'
PLACES_CAFFEMODEL_FILE = model_root + 'placesCNN/places205CNN_iter_300000_upgraded.caffemodel'
MEAN_FILE = model_root + 'placesCNN/ilsvrc_2012_mean.npy'

PLACES_TUNED_MODEL_FILE = model_root + 'placesCNN/places205CNN_deploy_FC7_upgraded_one.prototxt'
PLACES_TUNED_CAFFEMODEL_FILE = model_root + 'placesCNN/finetune_ed_scenes_iter_100000.caffemodel'

caffe.set_mode_cpu()

hybridplaces_net = caffe.Net(HYBRIDPLACES_MODEL_FILE, HYBRIDPLACES_CAFFEMODEL_FILE, caffe.TEST)
places_net = caffe.Net(PLACES_MODEL_FILE, PLACES_CAFFEMODEL_FILE, caffe.TEST)
places_tuned_net = caffe.Net(PLACES_TUNED_MODEL_FILE, PLACES_TUNED_CAFFEMODEL_FILE, caffe.TEST)

# transformer = caffe.io.Transformer({'data': places_net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', np.load(model_root + 'placesCNN/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_raw_scale('data', 255) # don't need this because opencv reads in 255 scale anyway
# transformer.set_channel_swap('data', (2, 1, 0)) # opencv is also read in BGR order
model_mean = np.load(model_root + 'placesCNN/ilsvrc_2012_mean.npy')

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
	people_mask = np.zeros((one_mask_size, one_mask_size))
	content_mask = np.zeros((one_mask_size, one_mask_size))

	f = skimage.transform.resize(frame, (227, 227, 3), preserve_range=True)
	f = f.transpose((2, 0, 1))
	f = f - model_mean;
	# input the data
	hybridplaces_net.blobs['data'].data[...] = f
	places_net.blobs['data'].data[...] = f
	places_tuned_net.blobs['data'].data[...] = f
	# send it through the net
	out_hybridplaces = hybridplaces_net.forward()
	out_places = places_net.forward()
	out_places_tuned = places_tuned_net.forward()

	# people

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


		# # cv2.imshow('net_init', transformer.deprocess('data', net.blobs['data'].data[0]))

		# # extract the conv5 features
		# hybridplaces_conv5_features = hybridplaces_net.blobs['conv5'].data
		# hybridplaces_conv5_features = hybridplaces_conv5_features[0]

		# places_conv5_features = places_net.blobs['conv5'].data # the shape is (1, 256, 13, 13)
		# places_conv5_features = places_conv5_features[0]

		# places_tuned_conv5_features = places_tuned_net.blobs['conv5'].data
		# places_tuned_conv5_features = places_tuned_conv5_features[0]

		# if (person_net_name == 'places'):
		# 	person_conv5_features = places_conv5_features
		# elif (person_net_name == 'places-tuned'):
		# 	person_conv5_features = places_tuned_conv5_features
		# elif (person_net_name == 'hybridplaces'):
		# 	person_conv5_features = hybridplaces_conv5_features
		# else:
		# 	sys.exit(0)

		# if (content_net_name == 'places'):
		# 	content_conv5_features = places_conv5_features
		# elif (content_net_name == 'places-tuned'):
		# 	content_conv5_features = places_tuned_conv5_features
		# elif (content_net_name == 'hybridplaces'):
		# 	content_conv5_features = hybridplaces_conv5_features
		# else:
		# 	sys.exit(0)


		# # multiply them by the weights
		# for nonzero_index in nonzero_people_weights:
		# 	people_mask += person_conv5_features[nonzero_index] * people_weight_data[nonzero_index]
		# for nonzero_index in nonzero_content_weights:
		# 	content_mask += content_conv5_features[nonzero_index] * content_weight_data[nonzero_index]


	# transform to heatmap
	# person_heatmap = scipy.ndimage.zoom(people_mask, 227.0/one_mask_size, order=1)
	content_heatmap = scipy.ndimage.zoom(content_mask, 227.0/one_mask_size, order=1)

	[content_box, content_scores] = process_heatmap(content_heatmap, content_heatmap_thresh, content_num_boxes, content_score_thresh)

	print "content box inside extract, ",  content_box
	print "content box scores, ",  content_scores

	# plt.subplot(1, 2, 1)
	# plt.imshow(heatmap)
	if display:
		# cv2.imshow('heatmap person', person_heatmap)
		cv2.imshow('heatmap content', content_heatmap)

	# print "Person bounds: ", np.amax(person_heatmap), np.amin(person_heatmap)
	# print "Content bounds: ", np.amax(content_heatmap), np.amin(content_heatmap)

	# transform to bounding box
	# [y_index_person, x_index_person] = np.where(person_heatmap > people_thresh)
	# [y_index_content, x_index_content] = np.where(content_heatmap > content_thresh)
	person_box = []
	# content_box = []
	# if (len(y_index_person) > 0 and len(x_index_person) > 0):
		# person_box = [min(x_index_person), min(y_index_person), max(x_index_person), max(y_index_person)]
	# if (len(y_index_content) > 0 and len(x_index_content) > 0):
		# content_box = [min(x_index_content), min(y_index_content), max(x_index_content), max(y_index_content)]

	return (person_box, content_box)

def run_video():
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

			cur_msec = cap.get(0) # 0 is the index for getting the msec
			frame_width = cap.get(3);
			frame_height = cap.get(4);
			print "cur_msec = ", cur_msec

			# only extract the person box every 10 frames
			# do some caffe stuff
			if index%5 == 0:
				(person_box, content_box) = extract_box(frame)
				print "Person: ", person_box
				print "Content: ", content_box
			if len(person_box) > 0:
				# cv2.rectangle(frame, (int(person_box[0]*frame.shape[1]/227.0), int(person_box[1]*frame.shape[0]/227.0)), (int(person_box[2]*frame.shape[1]/227.0), int(person_box[3]*frame.shape[0]/227.0)), (255, 0, 0))
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
			else:
				segment_list.append(section_info)
				print segment_list
				section_id += 1
				section_info = {"id": section_id, "start_time": cur_msec, "end_time": cur_msec, 
					"content": content, "person": person}

			person_prev = person
			content_prev = content
		else:
			break

	cap.release()
	cv2.destroyAllWindows()

	final_dict = {"youtubeID": vid_name[3:], "length": vidutils.get_video_length_secs(vid_name, vid_dir), "segments": segment_list}
	# pp.pprint(final_dict)
	return final_dict

if __name__ == '__main__':
	final_dict = run_video()
	pp.pprint(final_dict)
	json_dir = vid_dir + 'jsons/'
	if not os.path.isdir(json_dir):
		os.system('mkdir ' + json_dir)
	with open(json_dir + vid_name + '.json', 'w') as outfile:
		json.dump(final_dict, outfile)