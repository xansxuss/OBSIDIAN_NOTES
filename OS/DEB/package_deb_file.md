使用 checkinstall 替代 make install
安裝 OpenCV（或任何 CMake 專案）時，建議：
``` bash
sudo apt install checkinstall
sudo checkinstall
```
這樣會自動把 make install 過程包成 .deb，之後可直接用 dpkg -r 卸載乾淨。

