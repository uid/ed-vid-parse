import numpy as np
import cv2
import sys
import scipy
from scipy import io

global frame

vid_name = "ID-EMaTF9-ArJY"

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

# caffe stuff
MODEL_FILE = model_root + 'places205CNN_deploy_FC7_upgraded_one.prototxt'
CAFFEMODEL_FILE = model_root + 'places205CNN_iter_300000_upgraded.caffemodel'
MEAN_FILE = model_root + 'ilsvrc_2012_mean.npy'

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, CAFFEMODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1)) # channels, then width, then height
transformer.set_mean('data', np.load(model_root + 'ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_raw_scale('data', 255) # don't need this because opencv reads in 255 scale anyway
transformer.set_channel_swap('data', (2, 1, 0))

def extract_person_box(frame):
	# input the data
	net.blobs['data'].data[...] = transformer.preprocess('data', frame)
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
	# plt.imshow(heatmap)
	cv2.imshow('heatmap', heatmap)

	# transform to bounding box
	[y_index, x_index] = np.where(heatmap > thresh)
	if (len(y_index) > 0 and len(x_index) > 0):
		box = [min(x_index), min(y_index), max(x_index), max(y_index)]
		box[2] = box[2] - box[0]
		box[3] = box[3] - box[1]
		# bounding_box_patch.set_bounds(box[0], box[1]+box[3], box[2], -1*box[3])
		return box
	return []

def run_video():
	cap = cv2.VideoCapture(vid_dir + vid_name + '.mp4')

	index = 0
	while (cap.isOpened()):
		ret, frame = cap.read()

		cv2.imshow('frame', frame)

		# only extract the person box every 10 frames
		# do some caffe stuff
		if index%10 == 0:
			person_box = extract_person_box(frame)
			print person_box

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


		index += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	run_video()