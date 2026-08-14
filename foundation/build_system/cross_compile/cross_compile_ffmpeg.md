setting config

``` bash
./configure --prefix=${SYSROOT} --target-os=linux --arch=aarch64 --enable-cross-compile --cross-prefix=aarch64-linux-gnu- --cc=aarch64-linux-gnu-gcc --cxx=aarch64-linux-gnu-g++ --nm=aarch64-linux-gnu-nm --ar=aarch64-linux-gnu-ar --strip=aarch64-linux-gnu-strip --pkg-config=pkg-config --enable-shared --disable-static --disable-doc --disable-programs --enable-pic
make -j$(nproc)
make install
```


