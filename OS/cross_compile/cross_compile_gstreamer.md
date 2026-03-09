核心觀念（先釘住這三件事）

GStreamer 是一整個生態系，不是一個 lib

gstreamer (core)

gst-plugins-base

gst-plugins-good

gst-plugins-bad

gst-plugins-ugly
少一個，runtime 就會開始鬧脾氣。

交叉編譯 ≠ target apt install

你是在 x86 編譯

但 headers / libs 必須來自 ARM sysroot

不能混到 host 的 /usr/lib/x86_64-linux-gnu

pkg-config 是關鍵反派

99% 的失敗來自它指錯路

所以一定要「騙它」只看 sysroot

cross file（Meson 的靈魂）

aarch64.cross

[binaries]
c = 'aarch64-linux-gnu-gcc'
cpp = 'aarch64-linux-gnu-g++'
ar = 'aarch64-linux-gnu-ar'
strip = 'aarch64-linux-gnu-strip'
pkgconfig = 'pkg-config'

[host_machine]
system = 'linux'
cpu_family = 'aarch64'
cpu = 'armv8-a'
endian = 'little'

[properties]
sys_root = '/usr/aarch64-linux-gnu'
c_args = ['--sysroot=/usr/aarch64-linux-gnu']
cpp_args = ['--sysroot=/usr/aarch64-linux-gnu']

編譯 GStreamer core
git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git
cd gstreamer
git checkout 1.22.8   # 舉例，選你要的版本

meson setup build-arm64 \
  --cross-file ../aarch64.cross \
  --prefix=${SYSROOT}

ninja -C build-arm64
ninja -C build-arm64 install DESTDIR=$SYSROOT