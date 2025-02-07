mtk 燒機
genio-flash --load-dtbo gpu-mali.dtbo --load-dtbo apusys.dtbo --load-dtbo video.dtbo --load-dtbo camera-imx214-csi0.dtbo --load-dtbo display-dp.dtbo

set static ip 

in root
/lib/systemd/network/

vi 00-eth0.network

[Match] 
Name=eth0 

[Network] 
Address=192.168.33.125 
Gateway=192.168.33.2 
DNS=8.8.8.8
Netmask=255.255.255.0