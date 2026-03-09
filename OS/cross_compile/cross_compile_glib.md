前提假設

目標架構：aarch64 / ARMv8

Sysroot：/usr/aarch64-linux-gnu

交叉編譯器：aarch64-linux-gnu-gcc / aarch64-linux-gnu-g++

GLib 版本：2.74.x

Meson & Ninja 已安裝於主機

1️⃣ 準備交叉工具與 sysroot

確認工具鏈：

which aarch64-linux-gnu-gcc
which aarch64-linux-gnu-g++


確認 sysroot 有必要的 libiconv：

ls /usr/aarch64-linux-gnu/lib/libiconv.*
ls /usr/aarch64-linux-gnu/include/iconv.h


有就好，無需 pkg-config。交叉編譯用內建 iconv。

2️⃣ 建立 Meson Cross File

檔名：aarch64.cross，放在 GLib 原碼旁或方便的位置：

[binaries]
c = '/usr/bin/aarch64-linux-gnu-gcc'
cpp = '/usr/bin/aarch64-linux-gnu-g++'
ar = '/usr/bin/aarch64-linux-gnu-ar'
strip = '/usr/bin/aarch64-linux-gnu-strip'
pkgconfig = '/usr/bin/aarch64-linux-gnu-pkg-config'

[host_machine]
system = 'linux'
cpu_family = 'aarch64'
cpu = 'armv8-a'
endian = 'little'

[built-in options]
c_args = []
cpp_args = []


這個 cross file 指定了編譯器、sysroot 路徑，並用內建 iconv。

3️⃣ 下載 GLib 原始碼
wget https://download.gnome.org/sources/glib/2.74/glib-2.74.1.tar.xz
tar xf glib-2.74.1.tar.xz
cd glib-2.74.1

4️⃣ 設定 Meson Build

設定 build 目錄，並 prefix 指向 sysroot，保證安裝不碰 x86 系統：

meson setup build-arm64 \
  --cross-file ../aarch64.cross \
  --prefix=/usr/aarch64-linux-gnu



5️⃣ 編譯
ninja -C build-arm64

6️⃣ 安裝到 sysroot（安全方式）

ninja -C build-arm64 install

or

使用 Meson install --destdir，確保 x86 系統不被污染：

meson install -C build-arm64 --destdir=/usr/aarch64-linux-gnu


結果：

/usr/aarch64-linux-gnu/usr/lib/libglib-2.0.so
/usr/aarch64-linux-gnu/usr/lib/libgio-2.0.so
/usr/aarch64-linux-gnu/usr/include/glib-2.0


所有 symlink 都在 sysroot /usr/aarch64-linux-gnu 下，不會碰 /usr/local/lib。

7️⃣ 驗證
file /usr/aarch64-linux-gnu/usr/lib/libglib-2.0.so
# output: ELF 64-bit LSB shared object, ARM aarch64


檢查 include：

ls /usr/aarch64-linux-gnu/usr/include/glib-2.0

8️⃣ 後續建議

交叉編譯依賴 GLib 的套件（如 GStreamer）時，設定環境：

export PKG_CONFIG_SYSROOT_DIR=/usr/aarch64-linux-gnu
export PKG_CONFIG_PATH=$PKG_CONFIG_SYSROOT_DIR/usr/lib/pkgconfig:$PKG_CONFIG_PATH


這樣 pkg-config 就能找到剛安裝的 GLib。