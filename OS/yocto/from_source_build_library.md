下載 Perl 原始碼
前往 Perl 的官方網站 Perl 官方下載頁面。

tar -xvzf perl-5.X.X.tar.gz
cd perl-5.X.X

./Configure -des -Dprefix=/usr/local/perl

重要選項：
-des：使用預設設定，並啟用交互模式。
-Dprefix=/path/to/install：指定安裝目錄（默認為 /usr/local）。
可選配置：
-Dusethreads：啟用多執行緒支援。
-Doptimize='-O2'：優化編譯速度。
-DDEBUGGING：啟用調試模式（僅在需要時使用）。

make -j$(nproc)
make test
sudo make install
/usr/local/perl/bin/perl -v
export PATH=/usr/local/perl/bin:$PATH
/usr/local/perl/bin/cpan install lib


1. zlib
下載: zlib 最新版本
編譯指令:
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
2. ncurses
下載: ncurses 最新版本
編譯指令:
./configure --prefix=/usr/local --with-shared --with-termlib
make -j$(nproc)
sudo make install
3. gdbm
下載: GNU gdbm 最新版本
編譯指令:
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
4. libnss
下載: Mozilla NSS 最新版本
編譯指令:
cd nss
make BUILD_OPT=1 USE_SYSTEM_ZLIB=1 -j$(nproc)
sudo make install
5. OpenSSL
下載: OpenSSL 最新版本
編譯指令:
./config --prefix=/usr/local --openssldir=/usr/local/ssl
make -j$(nproc)
sudo make install
6. readline
下載: GNU readline 最新版本
編譯指令:
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
7. libffi
下載: libffi 最新版本
編譯指令:
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
8. bzip2
下載: bzip2 最新版本
編譯指令:
make -j$(nproc)
sudo make install PREFIX=/usr/local

python3-dev 
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
export CFLAGS="-fPIC"
export LDFLAGS="-fPIC"
tar -xvf Python-3.10.13.tgz
cd Python-3.10.13
./configure --enable-optimizations --prefix=/usr/local
make -j$(nproc)
sudo make altinstall


# opencv

git clone https://github.com/uclouvain/openjpeg.git
cd openjpeg
mkdir build
cd build
cmake ..
make
sudo make install

Glib

git clone https://gitlab.gnome.org/GNOME/glib.git
cd glib

mkdir build
cd build
meson setup .. --prefix=/usr --buildtype=release
ninja
sudo ninja install


pkg-config

tar zxvf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2.tar.gz
./configure --with-internal-glib
make
make install