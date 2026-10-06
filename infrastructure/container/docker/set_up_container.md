```
docker run -it -d \
-v /srv:/srv -v /mnt:/mnt -v /temp:/temp -v /home/eray/repo:/workspaces_data/repo -v /home/eray/project:/workspaces_data/project \
--ipc=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --device /dev/dri --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=all --restart unless-stopped \
-u $(id -u):$(id -g) \
-w /workspaces_data --name pytorch_2_3_1 565ac28ad01e /bin/bash
```

加``` -u $(id -u):$(id -g) ```第一次啟動
```
docker exec -u 0 -it cudaImage bash -c "groupadd -g $(id -g) eray && useradd -l -u $(id -u) -g eray -m eray && usermod -aG sudo eray"
```

```
# 以 root 身份進入已經在執行的容器
docker exec -u 0 -it pytorch_2_3_1 bash
```


```
docker create -v /workspaces_data --name data_volume_container -v /home/shared/:/workspaces_data/shared -v /home/eray/repo/:/workspaces_data/repo 
-v /home/eray/project/:/workspaces_data/project ubuntu:18.04
```



nvidia jetson

```
docker run -it -d \
    --runtime nvidia \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/l4t-jetpack:r36.3.0 \
    /bin/bash
```

jetson R364 ds 71

``` dockerfile
# =====================================================================
# STAGE 1: Compilation Environment (Builder)
# =====================================================================
FROM nvcr.io/nvidia/deepstream-l4t:7.1-triton-multiarch AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安裝編譯所需之開發者依賴套件 (Dev Libraries)
RUN apt-get update -qq && apt-get install -y --no-install-recommends --fix-missing \
    libmp3lame0 \
    libsdl2-2.0-0 \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    zlib1g \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstrtspserver-1.0-0 \
    gstreamer1.0-rtsp \
    gstreamer1.0-tools \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libgtk-3-0 \
    libfreetype6 \
    libpng16-16 \
    python3 \
    python3-pip \
    cmake \
    && rm -rf /var/lib/apt/lists/*
# 2. 下載 OpenCV 與 OpenCV_contrib 4.10.0
WORKDIR /tmp/opencv_build
RUN git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git

# 3. 編譯與安裝 OpenCV (針對 Jetson Orin 最佳化設定)
#    - CUDA_ARCH_BIN=8.7 (Orin 專屬架構)
#    - 自動尋找系統中的 aarch64 Python 與 Freetype 庫
WORKDIR /tmp/opencv_build/opencv/build
RUN cmake .. \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D WITH_GTK=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GIF=ON \
    -D WITH_AVIF=ON \
    -D WITH_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDA_ARCH_BIN=8.7 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_build/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_JAVA=OFF && \
    make -j$(nproc) && \
    make install



# =====================================================================
# STAGE 2: Lightweight Deployment Environment (Runtime)
# =====================================================================
FROM nvcr.io/nvidia/deepstream-l4t:7.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive
# 1. 補齊 Stage 2 執行 OpenCV/Qt/GStreamer 所需的動態庫 (Runtime Shared Libraries)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libmp3lame0 \
    libsdl2-dev \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    zlib1g \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstrtspserver-1.0-0 \
    gstreamer1.0-rtsp \
    gstreamer1.0-tools \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libgtk-3-0 \
    libfreetype6 \
    libpng-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. 從 Builder Stage 複製編譯完成的 OpenCV artifact 與標頭檔/PkgConfig
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/opencv4 /usr/local/include/opencv4
COPY --from=builder /usr/local/lib/pkgconfig/ /usr/local/lib/pkgconfig/

# 補複製 Python3 binding (若你需要 Python 開發/呼叫 OpenCV)
COPY --from=builder /usr/local/lib/python3.*/dist-packages/cv2 /usr/local/lib/python3.10/dist-packages/cv2

# 3. 刷新 Dynamic Linker Cache 讓系統認識新放入的 /usr/local/lib shared objects
RUN ldconfig

```


### 打包、推送與壓縮自動化腳本

建立一個 `deploy.sh` 腳本一次完成標籤、推送與本機備份：

``` Bash
#!/bin/bash
set -e

# ================= Configuration =================
IMAGE_NAME="my-deepstream-app"
TAG="latest"
DOCKER_USER="your_dockerhub_username" # 請修改為你的 Docker Hub 帳號
OUTPUT_FILENAME="${IMAGE_NAME}_${TAG}.tar.gz"
# ================================================

echo "=== 1. Building Image ==="
docker build -t ${IMAGE_NAME}:${TAG} .

echo "=== 2. Tagging & Pushing to Docker Hub ==="
docker tag ${IMAGE_NAME}:${TAG} ${DOCKER_USER}/${IMAGE_NAME}:${TAG}
docker push ${DOCKER_USER}/${IMAGE_NAME}:${TAG}

echo "=== 3. Exporting & Compressing to ${OUTPUT_FILENAME} ==="
if command -v pigz &> /dev/null; then
    docker save ${IMAGE_NAME}:${TAG} | pigz -9 -p $(nproc) > ${OUTPUT_FILENAME}
else
    docker save ${IMAGE_NAME}:${TAG} | gzip -9 > ${OUTPUT_FILENAME}
fi

echo "=== Success! Output file size: ==="
ls -lh ${OUTPUT_FILENAME}
```