Vagrant.configure("2") do |config|

  config.vm.box = "ubuntu/trusty64"
  # config.vm.box_url = "http://files.vagrantup.com/ubuntu/trusty64.box"

  # enable symlinks in the shared folder between guest and host
  config.vm.provider :virtualbox do |vb|
    vb.customize [ "setextradata", :id,
                  "VBoxInternal2/SharedFoldersEnableSymlinksCreate/v-root", "1" ]
    vb.memory = 2048
  end

  config.vm.provision :shell, :inline => <<SCRIPT

echo "Starting install"
apt-get update
apt-get install -y git
apt-get install -y build-essential
apt-get install -y gfortran
apt-get install -y libprotobuf-dev
apt-get install -y libleveldb-dev 
apt-get install -y libsnappy-dev 
apt-get install -y libopencv-dev 
apt-get install -y python-opencv
apt-get install -y libboost-all-dev 
apt-get install -y libhdf5-serial-dev
apt-get install -y libgflags-dev 
apt-get install -y libgoogle-glog-dev 
apt-get install -y liblmdb-dev 
apt-get install -y protobuf-compiler
apt-get install -y libatlas-base-dev
apt-get install -y python-dev

# install these to not get the SSL warnings on installs
apt-get install -y libssl-dev
apt-get install -y libffi-dev

# install pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
# remove the file we downloaded to get pip
rm get-pip.py

# upgrade pip
pip install --upgrade pip

# install ffmpeg and youtube-dl
# on 14.04, apparently ffmpeg was removed. so:
add-apt-repository ppa:mc3man/trusty-media
apt-get update
apt-get install -y ffmpeg

pip install youtube-dl

# do all the model fetching and stuff
cd /vagrant/
mkdir vids/
mkdir jsons/
mkdir models/
cd models/
mkdir imagenet/
cd imagenet/
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget https://www.dropbox.com/s/4dichqj3e7lyunn/ilsvrc_2012_mean.npy
cd ../
wget http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz
mkdir placesCNN/
tar -xvf placesCNN_upgraded.tar.gz -C placesCNN
rm placesCNN_upgraded.tar.gz
wget http://places.csail.mit.edu/model/hybridCNN_upgraded.tar.gz
mkdir hybridCNN/
tar -xvf hybridCNN_upgraded.tar.gz -C hybridCNN
rm hybridCNN_upgraded.tar.gz
cd placesCNN/
wget https://www.dropbox.com/s/21s6i61n5s4jc22/places205CNN_deploy_FC7_upgraded_one.prototxt
cd ../hybridCNN/
wget https://www.dropbox.com/s/zekwp0whb5rebng/hybridCNN_deploy_FC7_updated_one.prototxt

# copy the caffe clone from my github repo to make sure that
# there are no problems with versining
# install the caffe library in the root of the VM
cd
git clone -b csail-openstack-caffe https://github.com/mprat/caffe
cd caffe/python
for req in $(cat requirements.txt); do pip install $req; done
cd ../
make all
make pycaffe
make test
make runtest


# disable libdc1394 error with this hacky solution
# from stackoverflow: http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394
ln /dev/null /dev/raw1394

SCRIPT

end