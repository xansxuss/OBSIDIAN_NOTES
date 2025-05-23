[How to Configure Ubuntu 22.04 for CUDA Programming and OpenCV: A Comprehensive Guide](https://medium.com/@adari.girishkumar/how-to-configure-ubuntu-22-04-for-cuda-programming-and-opencv4-a-comprehensive-guide-e1eb89cbc21f)


#新版opencv須注意nvidia video codec SDK(nvidia 影像編譯庫) 有無安裝，不安裝-D WITH_NVCUVID=OFF
** 將nvidia video codec SDK Lib and Interface copy to /usr/local/cuda-*/lib and /usr/local/cuda-*/include
nvidia video codec SDK 下載網址"https://developer.nvidia.com/video-codec-sdk-archive"

[build FFmpeg](https://medium.com/@vladakuc/compile-opencv-4-7-0-with-ffmpeg-5-compiled-from-the-source-in-ubuntu-434a0bde0ab6)
[build gstreamer](https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c)
[Using CMake to build and install OpenCV for Python and C++ in Ubuntu 20.04](https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a)

[opencv_install](https://blog.csdn.net/weixin_44384491/article/details/121142093)


CMAKE:
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D  OPENCV_GENERATE_PKGCONFIG=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D with_GTK=ON -D WITH_GSTREAMER=ON -DWITH_GIF=ON -DWITH_AVIF=ON -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CUDA_ARCH_BIN=8.6 -D CUDA_ARCH_PTX=  -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_FREETYPE=TRUE -D OPENCV_ENABLE_NONFREE=ON -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 ..
make -j$(nproc)

[opencv with cuda](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
[【小白用】在Ubuntu上安装OpenCV任何版本+Contrib库+CUDA兼容](https://waltpeter.github.io/open-cv-basic/install-opencv-with-contrib-ubuntu/index.html)
[opencv cmake option flags](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)


for android
https://mdeore.medium.com/latest-android-ndk-to-build-opencv-ccecd11efa82
https://blog.devgenius.io/opencv-on-android-tiny-with-optimization-enabled-932460acfe38
cmake .. -DANDROID_TOOLCHAIN=clang -DTARGET_SOC=rk3588 -DCMAKE_SYSTEM_NAME=Android -DCMAKE_TOOLCHAIN_FILE=~/workspaces/android/NDK/android-ndk-r18b/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release -D WITH_OPENCL=ON -D WITH_OPENGL=ON -DBUILD_ANDROID_PROJECTS=OFF -D BUILD_opencv_videostab=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_superres=OFF  -D BUILD_opencv_stitching=OFF -D BUILD_opencv_shape=OFF -D WITH_CUDA=OFF -D WITH_MATLAB=OFF -D BUILD_ANDROID_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D ANDROID_STL=c++_shared -D BUILD_SHARED_LIBS=ON -D BUILD_opencv_objdetect=OFF -D BUILD_opencv_video=OFF -D BUILD_opencv_videoio=OFF -D BUILD_opencv_features2d=ON -D BUILD_opencv_flann=OFF -D BUILD_opencv_highgui=ON -D BUILD_opencv_ml=OFF -D BUILD_opencv_photo=OFF -D BUILD_opencv_python=OFF

for cross complier
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/workspaces_data/repo/medatek_genio_700/mediatek/mtk_infer/toolchain.cmake \                                
  -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DWITH_GSTREAMER=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \                                  
  -DWITH_FREETYPE=ON \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DPYTHON3_EXECUTABLE=$SYSROOT/usr/bin/python3 \
  -DPYTHON3_INCLUDE_DIR=$SYSROOT/usr/include/python3.10 \
  -DPYTHON3_LIBRARY=$SYSROOT/usr/lib/libpython3.so