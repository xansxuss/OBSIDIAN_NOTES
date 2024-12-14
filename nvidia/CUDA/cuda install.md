1. 顯示GPU資訊並查詢支援的CUDA版本
	1. sudo lshw -numeric -C display
2. 清除既有nvidia driver
	1. sudo apt-get purge nvidia*
	2. sudo apt-get --purge autoremove "\*nvidia\*"
3. 加入GPU ppa
	1. sudo add-apt-repository ppa:graphics-drivers
4. packages更新
	1. sudo apt-get update
5. 列出支援的GPU driver版本
	1. ubuntu-drivers list
6. nvidia-driver-470版本安裝
	1. sudo apt install nvidia-driver-470
7. 重新啟動
	1. sudo reboot
8. 檢查nvidia driver版本
	1. nvidia-smi
9. cuda 11.6 安裝
	1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
	2. sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
	3. wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
	4. sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
	5. sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
	6. sudo apt-get update && sudo apt-get -y install cuda
10. 檢查 cuda安裝路徑
	1. ll /usr/local/cuda/bin/
11. 修改bashrc
	1. vi ~/.bashrc
	2. add:
		1. export PATH=/usr/local/cuda/bin:$PATH
		2. export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
		3.source ~/.bashrc
12. Downlod cudnn cuda 11.6 version
	1. https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
13. 安裝cudnn
	1. sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
	2. sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	3. sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

### reference
https://medium.com/@scofield44165/ubuntu-20-04%E4%B8%AD%E5%AE%89%E8%A3%9Dnvidia-driver-cuda-11-4-2%E7%89%88-cudnn-install-nvidia-driver-460-cuda-11-4-2-cudnn-6569ab816cc5

/sbin/ldconfig.real: /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8 is not a symbolic link
https://blog.csdn.net/jy1023408440/article/details/107258942


1. [How to install CUDA, cuDNN and TensorFlow on Ubuntu 22.04 (2023)](https://medium.com/@gokul.a.krishnan/how-to-install-cuda-cudnn-and-tensorflow-on-ubuntu-22-04-2023-20fdfdb96907)

sudo apt-get install nvidia-cuda-toolkit
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2