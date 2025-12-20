mtk 刷機環境設定
1. To install Genio tools on a Linux host, you need:
    Ubuntu 18.04 or later LTS version
    Administrator privilege
    Internet connection
    Git 1.8 or later
    Python 3.8 or later
    pip3 20.3 or later
    Fastboot 28.0.2 or later
2. echo -n 'SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="201c", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="0003", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="gpio", MODE="0660", TAG+="uaccess"
' | sudo tee /etc/udev/rules.d/72-aiot.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
3. echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="201c", MODE="0660", $ GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/96-rity.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo usermod -a -G plugdev $USER
4. sudo usermod -a -G dialout $USER
5. pip3 install -U genio-tools
   1. genio-config
    fastboot: OK
    udev rules: OK

mtk 刷機
genio-flash --load-dtbo gpu-mali.dtbo --load-dtbo apusys.dtbo --load-dtbo video.dtbo --load-dtbo camera-imx214-csi0.dtbo --load-dtbo display-dp.dtbo

set static ip 

in root
/lib/systemd/network/
/run/systemd/network/

vi 00-eth0.network

[Match] 
Name=eth0 

[Network] 
Address=192.168.33.125 
Gateway=192.168.33.2 
DNS=8.8.8.8
Netmask=255.255.255.0