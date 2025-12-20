wget -c https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz
tar -zxvf cmake-3.26.3.tar.gz
cd cmake-3.26.3/

./bootstrap
make
sudo make install

cmake --version