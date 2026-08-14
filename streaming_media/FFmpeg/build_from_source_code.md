### x86:

### arm
Jetson orin series(3rd series)
1. compile and install to system
```bash
sudo apt install nvidia-l4t-jetson-multimedia-api
git clone https://github.com/Keylost/jetson-ffmpeg.git
cd jetson-ffmpeg
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../
./ffpatch.sh ../ffmpeg-6.0.1
cd ../ffmpeg
./configure \
  --prefix=/usr/local \
  --arch=aarch64 \
  --target-os=linux \
  --enable-shared \
  --enable-pic \
  --enable-neon \
  --enable-gpl \
  --enable-nonfree \
  --enable-nvmpi \
  --enable-libx264 \
  --enable-libx265 \
  --extra-cflags="-march=armv8.2-a"
make -j$(nproc)
sudo make install
# 用 nvmpi 硬體編碼 
ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 -c:v h264_nvmpi -b:v 8M out_nvmpi.mp4
```

2. package
```bash
mkdir -p ./ffmpeg-nvmpi-deb/DEBIAN
mkdir -p ./ffmpeg-nvmpi-deb/usr/local
mkdir -p ./ffmpeg-nvmpi-deb/usr/local/lib usr/local/include 
mkdir -p ./ffmpeg-nvmpi-deb/usr/local/share/man/man3
cp -a /usr/local/bin usr/local/
cp -a /usr/local/lib/libav* usr/local/lib/
cp -a /usr/local/lib/libsw* usr/local/lib/
cp -a /usr/local/lib/libpostproc* usr/local/lib/
cp -a /usr/local/lib/pkgconfig usr/local/lib/
cp -a /usr/local/include/libav* usr/local/include/
cp -a /usr/local/include/libsw* usr/local/include/
cp -a /usr/local/include/libpostproc* usr/local/include/
cp -a /usr/local/share/man/man3/libav* usr/local/share/man/man3/ 2>/dev/null
cat > DEBIAN/control << 'EOF'
Package: ffmpeg-nvmpi
Version: 6.1-orin-nx-1
Architecture: arm64
Maintainer:Neo 
Depends: libx264-163, libx265-199, nvidia-l4t-jetson-multimedia-api
Conflicts: ffmpeg, ffmpeg-custom
Replaces: ffmpeg, ffmpeg-custom
Provides: ffmpeg
Section: video
Priority: optional
Description: FFmpeg with Jetson nvmpi hardware codec support
 自編 FFmpeg，搭配 Jetson Multimedia API (nvmpi) 硬體編解碼，
 適用於 Jetson Orin NX (JetPack 6.0, L4T R36.3.0)。
EOF
# 加上安裝後腳本,自動更新 ldconfig
cat > DEBIAN/postinst << 'EOF'
#!/bin/bash
ldconfig
EOF
chmod 755 DEBIAN/postinst
cd ..
dpkg-deb --build --root-owner-group ffmpeg-nvmpi-deb
mv ffmpeg-nvmpi-deb.deb ffmpeg-nvmpi_6.1-orin-nx-1_arm64.deb
```

Jetson series(2nd series)
1. compile and install to system
```bash
```