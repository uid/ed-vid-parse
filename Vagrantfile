Vagrant.configure("2") do |config|

  config.vm.box = "ubuntu/trusty64"
  # config.vm.box_url = "http://files.vagrantup.com/ubuntu/trusty64.box"

  # enable symlinks in the shared folder between guest and host
  config.vm.provider :virtualbox do |vb|
   vb.customize [ "setextradata", :id,
                  "VBoxInternal2/SharedFoldersEnableSymlinksCreate/v-root", "1" ]
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
apt-get install -y libboost-all-dev 
apt-get install -y libhdf5-serial-dev
apt-get install -y libgflags-dev 
apt-get install -y libgoogle-glog-dev 
apt-get install -y liblmdb-dev 
apt-get install -y protobuf-compiler
apt-get install -y libatlas-base-dev

# install pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py

# install ffmpeg and youtube-dl
# on 14.04, apparently ffmpeg was removed. so:
add-apt-repository ppa:mc3man/trusty-media
apt-get update
apt-get install -y ffmpeg

pip install youtube-dl

# copy the caffe clone from my github repo to make sure that
# there are no problems with versining
git clone -b csail-openstack-caffe https://github.com/mprat/caffe
cd caffe/python
for req in $(cat requirements.txt); do sudo pip install $req; done
cd ../
make all
make test
make runtest
make pycaffe

# do all the model fetching and stuff
mkdir models/
cd models/
mkdir imagenet/
cd imagenet/
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget https://www.dropbox.com/s/4dichqj3e7lyunn/ilsvrc_2012_mean.npy
cd ..
wget http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz
tar -xvf placesCNN_upgraded.tar.gz -C placesCNN --strip-components=1
wget https://www.dropbox.com/s/21s6i61n5s4jc22/places205CNN_deploy_FC7_upgraded_one.prototxt



SCRIPT

end