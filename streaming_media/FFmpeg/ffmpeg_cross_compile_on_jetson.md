``` bash
# 1. 清理變數
unset LDFLAGS CPPFLAGS CFLAGS

# 2. 鎖定 pkg-config
export PKG_CONFIG_DIR=""
export PKG_CONFIG_LIBDIR="/usr/local/jetson_sysroot/usr/lib/aarch64-linux-gnu/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="/usr/local/jetson_sysroot"

# 1. 清理先前的編譯殘留
make clean

# 2. 重新執行配置（核心：加入了倒數第三行的 --strip）
./configure \
  --prefix=../ffmpeg-deb/ \
  --enable-cross-compile \
  --target-os=linux \
  --arch=aarch64 \
  --cc=aarch64-linux-gnu-gcc \
  --cxx=aarch64-linux-gnu-g++ \
  --pkg-config=pkg-config \
  --sysroot=/usr/local/jetson_sysroot \
  --extra-cflags="-I/usr/local/jetson_sysroot/usr/include -I/usr/local/jetson_sysroot/usr/include/aarch64-linux-gnu -I/usr/local/jetson_sysroot/usr/src/jetson_multimedia_api/include" \
  --extra-ldflags="-L/usr/local/jetson_sysroot/usr/lib/aarch64-linux-gnu -L/usr/local/jetson_sysroot/usr/lib/aarch64-linux-gnu/tegra -Wl,-rpath-link,/usr/local/jetson_sysroot/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,/usr/local/jetson_sysroot/lib/aarch64-linux-gnu" \
  --extra-libs="-lm -lv4lconvert -ljpeg" \
  --enable-shared \
  --disable-static \
  --enable-gpl \
  --enable-nonfree \
  --enable-v4l2-m2m \
  --enable-libv4l2 \
  --strip=aarch64-linux-gnu-strip
```