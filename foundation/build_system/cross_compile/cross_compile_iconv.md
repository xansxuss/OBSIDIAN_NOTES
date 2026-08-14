1️⃣ 確認 sysroot 位置

假設你的交叉編譯 sysroot 是：

/usr/aarch64-linux-gnu


你需要把 libiconv 的頭文件、庫檔、pkg-config 放到 sysroot 對應目錄：

/usr/aarch64-linux-gnu/include/iconv.h
/usr/aarch64-linux-gnu/lib/libiconv.so
/usr/aarch64-linux-gnu/lib/pkgconfig/iconv.pc

2️⃣ 安裝 ARM64 版本的 libiconv

Debian/Ubuntu 官方沒有單獨的 libiconv 套件，通常是 glibc 提供 iconv，或者你自己編譯 libiconv。建議做法是交叉編譯 libiconv：

a. 下載 libiconv 原始碼
wget https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.17.tar.gz
tar xf libiconv-1.17.tar.gz
cd libiconv-1.17

b. 交叉編譯

假設你的交叉編譯器是 aarch64-linux-gnu-gcc，sysroot 是 /usr/aarch64-linux-gnu：

./configure \
  --host=aarch64-linux-gnu \
  --prefix=/usr/aarch64-linux-gnu \
  --enable-shared \
  --disable-static
make -j$(nproc)
make install


這樣會把 libiconv.so、iconv.h 和 pkg-config 文件 安裝到 sysroot。

注意：--prefix 指向 sysroot，Meson / pkg-config 才能找到。

3️⃣ 設定交叉編譯 pkg-config
export PKG_CONFIG_SYSROOT_DIR=/usr/aarch64-linux-gnu
export PKG_CONFIG_LIBDIR=/usr/aarch64-linux-gnu/lib/pkgconfig
aarch64-linux-gnu-pkg-config --modversion libiconv


應該可以看到版本號，表示 pkg-config 找到了。

4️⃣ 重新編譯 GLib
meson setup build-arm64 \
  --cross-file ../aarch64.cross \
  --prefix=/usr \
  -Dintrospection=disabled \
  -Dgtk_doc=disabled
