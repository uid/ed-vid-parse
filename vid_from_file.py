import numpy as np
import cv2
import sys
import scipy
from scipy import io
import skimage
from skimage import transform

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

content_weightfile_name = 'places-content-train-activ-target-50.mat'
content_weight_data = scipy.io.loadmat(weights_root + content_weightfile_name, squeeze_me=True)
content_weight_data = content_weight_data['weights']
nonzero_content_weights = np.nonzero(content_weight_data)[0]

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
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(model_root + 'ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_raw_scale('data', 255) # don't need this because opencv reads in 255 scale anyway
# transformer.set_channel_swap('data', (2, 1, 0)) # opencv is also read in BGR order
model_mean = np.load(model_root + 'ilsvrc_2012_mean.npy')

def extract_box(frame):
	f = skimage.transform.resize(frame, (227, 227, 3), preserve_range=True)
	f = f.transpose((2, 0, 1))
	f = f - model_mean;
	# input the data
	net.blobs['data'].data[...] = f
	# send it through the net
	out = net.forward()

	# cv2.imshow('net_init', transformer.deprocess('data', net.blobs['data'].data[0]))

	# extract the conv5 features
	conv5_features = net.blobs['conv5'].data # the shape is (1, 256, 13, 13)
	conv5_features = conv5_features[0]

	people_mask = np.zeros((13, 13))
	content_mask = np.zeros((13, 13))

	# multiply them by the weights
	for nonzero_index in nonzero_people_weights:
		people_mask += conv5_features[nonzero_index] * people_weight_data[nonzero_index]
	for nonzero_index in nonzero_content_weights:
		content_mask += conv5_features[nonzero_index] * content_weight_data[nonzero_index]


	# transform to heatmap
	person_heatmap = scipy.ndimage.zoom(people_mask, 227/13.0, order=1)
	content_heatmap = scipy.ndimage.zoom(content_mask, 227/13.0, order=1)

	# plt.subplot(1, 2, 1)
	# plt.imshow(heatmap)
	# cv2.imshow('heatmap', heatmap)

	# transform to bounding box
	[y_index_person, x_index_person] = np.where(person_heatmap > thresh)
	[y_index_content, x_index_content] = np.where(content_heatmap > thresh)
	person_box = []
	content_box = []
	if (len(y_index_person) > 0 and len(x_index_person) > 0):
		person_box = [min(x_index_person), min(y_index_person), max(x_index_person), max(y_index_person)]
	if (len(y_index_content) > 0 and len(x_index_content) > 0):
		content_box = [min(x_index_content), min(y_index_content), max(x_index_content), max(y_index_content)]

	return (person_box, content_box)

def run_video():
	cap = cv2.VideoCapture(vid_dir + vid_name + '.mp4')

	index = 0
	while (cap.isOpened()):
		ret, frame = cap.read()

		# only extract the person box every 10 frames
		# do some caffe stuff
		if index%20 == 0:
			(person_box, content_box) = extract_box(frame)
			print "Person: ", person_box
			print "Content: ", content_box
		if len(person_box) > 0:
			cv2.rectangle(frame, (int(person_box[0]*frame.shape[1]/227.0), int(person_box[1]*frame.shape[0]/227.0)), (int(person_box[2]*frame.shape[1]/227.0), int(person_box[3]*frame.shape[0]/227.0)), (255, 0, 0))

		if len(content_box) > 0:
			cv2.rectangle(frame, (int(content_box[0]*frame.shape[1]/227.0), int(content_box[1]*frame.shape[0]/227.0)), (int(content_box[2]*frame.shape[1]/227.0), int(content_box[3]*frame.shape[0]/227.0)), (0, 255, 0))

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


		index += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	run_video()