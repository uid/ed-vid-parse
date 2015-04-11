sudo apt-get update

sudo apt-get install build-essential git gfortran

sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

sudo apt-get install libatlas-base-dev

# no need for CUDA since CPU / cloud / cluster / whatever

# install pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

# install ffmpeg and youtube-dl
# on 14.04, apparently ffmpeg was removed. so:
# sudo apt-get install ffmpeg
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get dist-upgrade
sudo pip install youtube-dl


# probably want to keep a clone / copy of the makefile somewhere
# on my own github rather than cloning / copy from scratch
git clone -b csail-openstack-caffe https://github.com/mprat/caffe
cd caffe/python
for req in $(cat requirements.txt); do sudo pip install $req; done
cd ../
make all
make test
make runtest