``` bash
dpkg --add-architecture arm64
apt-get update
apt-get install -y build-essential cmake git vim curl pkg-config python3 python3-pip wget unzip  crossbuild-essential-arm64 cmake  pkg-config && rm -rf /var/lib/apt/lists/*
pip3 install meson ninja
```
opencv dependency

``` bash
# 加入 FFmpeg 與相關編解碼依賴
apt-get install -y \
    libavcodec-dev:arm64 \
    libavformat-dev:arm64 \
    libavutil-dev:arm64 \
    libswscale-dev:arm64 \
    libavresample-dev:arm64 \
    libpostproc-dev:arm64 \
    # GStreamer 依賴
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    # 其他常用影像庫
    libjpeg-dev:arm64 \
    libpng-dev:arm64 \
    libtiff-dev:arm64 \
    libgtk-3-dev:arm64 \
    libglib2.0-dev:arm64 \
    libcairo2-dev:arm64 \
    libpango1.0-dev:arm64 \
    libatk1.0-dev:arm64 \
    libgdk-pixbuf2.0-dev:arm64
```



create toolchain-arm64.cmake

``` cmake 
# 目標平台
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# --- 關鍵修改 1: 重新定義路徑概念 ---
# 在 Multi-arch 環境下，Sysroot 實際上就是根目錄 /
# 我們透過 CMAKE_FIND_ROOT_PATH 來包含多個搜尋路徑
set(CMAKE_SYSROOT /)
set(CMAKE_FIND_ROOT_PATH 
    /usr/aarch64-linux-gnu 
    /usr/lib/aarch64-linux-gnu
    /usr/include/aarch64-linux-gnu
)

# 交叉編譯器 (建議使用路徑變數確保一致性)
set(CROSS_TRIPLE aarch64-linux-gnu)
set(CMAKE_C_COMPILER   /usr/bin/${CROSS_TRIPLE}-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/${CROSS_TRIPLE}-g++)

# 工具鏈工具
set(CMAKE_AR      /usr/bin/${CROSS_TRIPLE}-ar)
set(CMAKE_AS      /usr/bin/${CROSS_TRIPLE}-as)
set(CMAKE_NM      /usr/bin/${CROSS_TRIPLE}-nm)
set(CMAKE_LINKER  /usr/bin/${CROSS_TRIPLE}-ld)
set(CMAKE_STRIP   /usr/bin/${CROSS_TRIPLE}-strip)
set(CMAKE_RANLIB  /usr/bin/${CROSS_TRIPLE}-ranlib)

# CMake 對搜尋策略的設定
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# --- 關鍵修改 2: 修正 pkg-config 配置 ---
# 確保 pkg-config 指向 arm64 的 .pc 檔案路徑
set(ENV{PKG_CONFIG_SYSROOT_DIR} /)
set(ENV{PKG_CONFIG_LIBDIR} 
    /usr/lib/${CROSS_TRIPLE}/pkgconfig:
    /usr/share/pkgconfig
)
# 清空 PKG_CONFIG_PATH 避免污染
set(ENV{PKG_CONFIG_PATH} "")

# 如果你有自己封裝 aarch64-linux-gnu-pkg-config 腳本
set(PKG_CONFIG_EXECUTABLE /usr/bin/${CROSS_TRIPLE}-pkg-config)
```


``` bash
#!/bin/bash

# 1. 自動偵測 Triplet (增加靈活性，若改名為 arm-linux-gnueabihf-pkg-config 也能用)
TRIPLE=$(basename "$0" | sed 's/-pkg-config//')

# 2. 定義 PKG_CONFIG_LIBDIR
# 包含目標架構專屬路徑 (Priority 1) 與 架構無關路徑 (Priority 2)
# 使用 PKG_CONFIG_LIBDIR 會完全取代 pkg-config 預設的 Host 搜尋路徑
ARM_SPECIFIC_DIR="/usr/lib/${TRIPLE}/pkgconfig"
SHARE_DIR="/usr/share/pkgconfig"

export PKG_CONFIG_LIBDIR="${ARM_SPECIFIC_DIR}:${SHARE_DIR}"

# 3. 處理自行編譯的庫 (選用)
# 如果你有額外編譯的庫在非標準路徑，請用 PKG_CONFIG_PATH
# export PKG_CONFIG_PATH="/opt/my-libs/lib/pkgconfig"

# 4. 設定系統參數
# 允許輸出編譯與連結參數
export PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1
export PKG_CONFIG_ALLOW_SYSTEM_LIBS=1

# 5. 確保不會誤用 Host 的路徑 (加強保險)
# 如果有使用 sysroot，可以取消註解下一行
# export PKG_CONFIG_SYSROOT_DIR="/"

# 6. 執行真正的 pkg-config
exec pkg-config "$@"

```
