### issue

``` bash
sudo dpkg -i /media/eray/storage/home_root_backup/Downloads/deepstream-7.1_7.1.0-1_amd64.deb
Selecting previously unselected package deepstream-7.1.
(Reading database ... 257934 files and directories currently installed.)
Preparing to unpack .../deepstream-7.1_7.1.0-1_amd64.deb ...
Unpacking deepstream-7.1 (7.1.0-1) ...
dpkg: dependency problems prevent configuration of deepstream-7.1:
 deepstream-7.1 depends on libyaml-cpp-dev (>= 0.6.2); however:
  Package libyaml-cpp-dev is not installed.

dpkg: error processing package deepstream-7.1 (--install):
 dependency problems - leaving unconfigured
Errors were encountered while processing:
 deepstream-7.1
```
deepstream 要libyaml-cpp-dev 0.7.0
ubuntu24.04 只有libyaml-cpp-dev 0.8.0

```
sudo updatedb
locate libyaml-cpp.so
/usr/local/lib/libyaml-cpp.so
/usr/local/lib/libyaml-cpp.so.0.8
/usr/local/lib/libyaml-cpp.so.0.8.0
```

目前在安裝 DeepStream 7.1 時遇到了一個常見問題：缺少依賴套件 libyaml-cpp-dev (>= 0.6.2)，導致 .deb 套件無法正確設定（configure）。

### 第一步：移除舊版 yaml-cpp 0.8
查看哪些套件用到 yaml-cpp：
``` bash
dpkg -S libyaml-cpp.so
``` 
💣 移除 apt 安裝的 0.8（如果存在）：
``` bash
sudo apt remove --purge libyaml-cpp-dev libyaml-cpp0.8
```
🔥 移除你之前手動編譯安裝的 yaml-cpp 0.8（如果還殘留）：
``` bash
sudo rm -f /usr/local/lib/libyaml-cpp.so*
sudo rm -f /usr/lib/x86_64-linux-gnu/libyaml-cpp.so*
sudo ldconfig
```
確認都清空：
``` bash
locate libyaml-cpp.so
```

### 第二步：編譯 yaml-cpp v0.7.0
``` bash
# 下載 yaml-cpp 原始碼
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
git checkout yaml-cpp-0.7.0
```
#### 建立 build 目錄並編譯
``` bash
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make -j$(nproc)
``` 
#### 安裝到 /usr/local/lib
``` bash
sudo make install
sudo ldconfig
```
確認版本與路徑：
``` bash
ls -l /usr/local/lib/libyaml-cpp.so*
```
應該會看到：

``` bash
/usr/local/lib/libyaml-cpp.so -> libyaml-cpp.so.0.7
/usr/local/lib/libyaml-cpp.so.0.7
/usr/local/lib/libyaml-cpp.so.0.7.0
```

### 第三步：用 equivs 建立虛擬套件解決依賴（不破壞系統）

讓 dpkg 知道依賴已滿足，可用 equivs 自製一個假的 libyaml-cpp-dev 套件。
``` bash
sudo apt-get install equivs
mkdir tmp-libyaml
cd tmp-libyaml
equivs-control libyaml-cpp-dev
```
編輯 libyaml-cpp-dev 這檔案，修改內容（至少要有套件名與版本）：
``` bash
vi libyaml-cpp-dev
```
``` bash
Section: misc
Priority: optional
Standards-Version: 3.9.2
Package: libyaml-cpp-dev
Version: 0.7.0
Maintainer: Your Name <youremail@example.com>
Architecture: all
Description: Dummy package to satisfy DeepStream libyaml-cpp-dev dependency
``` 
然後建包並安裝：
``` bash
equivs-build libyaml-cpp-dev
```
這會在當前目錄產生一個 .deb，像是：
``` bash
libyaml-cpp-dev_0.7.0_all.deb
```
``` bash
sudo dpkg -i libyaml-cpp-dev_0.7.0_all.deb
```
這樣就會讓 dpkg 認為你有裝 libyaml-cpp-dev，不會報依賴錯誤。

避免 apt 再裝回 0.8
你可以將它 pin 起來防止 apt 自動裝回：
``` bash
echo -e "Package: libyaml-cpp*\nPin: release *\nPin-Priority: -1" | sudo tee /etc/apt/preferences.d/no-yaml-cpp-08
```

