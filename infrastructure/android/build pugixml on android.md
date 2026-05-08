1. modify https://github.com/zeux/pugixml.git fit android.bp compiler
2. modify ~/proj/3588_Android12_SDK/build/make/target/product/gsi/32.txt add VNDK-product: libpugixml.so
3. modify ~/proj/3588_Android12_SDK/build/make/target/product/gsi/current.txt add VNDK-product: libpugixml.so
4. scp pugixml folder to ~/proj/3588_Android12_SDK/external/
5. cd to ~/proj/3588_Android12_SDK/ and rebuild