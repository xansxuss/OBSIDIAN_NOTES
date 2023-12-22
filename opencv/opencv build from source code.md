[opencv_install](https://blog.csdn.net/weixin_44384491/article/details/121142093)
[# Using CMake to build and install OpenCV for Python and C++ in Ubuntu 20.04](https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a)

CMAKE:
cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=1 -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D OPENCV_ENABLE_NONFREE=ON -D CUDNN_INCLUDE_DIR=/usr/include -D CUDNN_VERSION=8.2.0 -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn_static_v8.a -D CUDA_ARCH_BIN=8.6 -D WITH_OPENCL=ON -D WITH_OPENGL=ON-D WITH_IPP=ON -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_VTK=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.19/modules/ ..

[opencv with cuda](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
[【小白用】在Ubuntu上安装OpenCV任何版本+Contrib库+CUDA兼容](https://waltpeter.github.io/open-cv-basic/install-opencv-with-contrib-ubuntu/index.html)
[opencv cmake option flags](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)


for android
https://mdeore.medium.com/latest-android-ndk-to-build-opencv-ccecd11efa82
https://blog.devgenius.io/opencv-on-android-tiny-with-optimization-enabled-932460acfe38
cmake .. -DANDROID_TOOLCHAIN=clang -DTARGET_SOC=rk3588 -DCMAKE_SYSTEM_NAME=Android -DCMAKE_TOOLCHAIN_FILE=~/workspaces/android/NDK/android-ndk-r18b/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release -D WITH_OPENCL=ON -D WITH_OPENGL=ON -DBUILD_ANDROID_PROJECTS=OFF -D BUILD_opencv_videostab=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_superres=OFF  -D BUILD_opencv_stitching=OFF -D BUILD_opencv_shape=OFF -D WITH_CUDA=OFF -D WITH_MATLAB=OFF -D BUILD_ANDROID_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D ANDROID_STL=c++_shared -D BUILD_SHARED_LIBS=ON -D BUILD_opencv_objdetect=OFF -D BUILD_opencv_video=OFF -D BUILD_opencv_videoio=OFF -D BUILD_opencv_features2d=ON -D BUILD_opencv_flann=OFF -D BUILD_opencv_highgui=ON -D BUILD_opencv_ml=OFF -D BUILD_opencv_photo=OFF -D BUILD_opencv_python=OFF
