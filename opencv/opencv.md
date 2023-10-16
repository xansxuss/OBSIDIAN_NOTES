[opencv_install](https://blog.csdn.net/weixin_44384491/article/details/121142093)
[# Using CMake to build and install OpenCV for Python and C++ in Ubuntu 20.04](https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a)

CMAKE:
cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=1 -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D OPENCV_ENABLE_NONFREE=ON -D CUDNN_INCLUDE_DIR=/usr/include -D CUDNN_VERSION=8.2.0 -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn_static_v8.a -D CUDA_ARCH_BIN=8.6 -D WITH_OPENCL=ON -D WITH_OPENGL=ON-D WITH_IPP=ON -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_VTK=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.19/modules/ ..

[opencv with cuda](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
[【小白用】在Ubuntu上安装OpenCV任何版本+Contrib库+CUDA兼容](https://waltpeter.github.io/open-cv-basic/install-opencv-with-contrib-ubuntu/index.html)
[opencv cmake option flags](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)