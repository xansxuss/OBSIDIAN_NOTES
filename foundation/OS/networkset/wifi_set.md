1. 查看wifi裝置裝態 
``` bash 
sudo dmesg | grep wlan # 看驅動有沒有載入
ip link # 確認 wlan0 出現
iwconfig # 確認無線網卡狀態
``` 
2. 配置 wpa_supplicant.conf
你可以放一個預設的 WiFi 配置檔在 rootfs，例如：
``` bash
ctrl_interface=/var/run/wpa_supplicant
update_config=1

network={
    ssid="你的SSID"
    psk="你的密碼"
    key_mgmt=WPA-PSK
}
```
3. 啟動流程
``` bash 
# 1. 殺掉舊的
killall wpa_supplicant
rm -rf /var/run/wpa_supplicant/*

# 2. 啟用介面
ip link set wlp4s0 up

# 3. 啟動 wpa_supplicant（加點 debug 輸出）
wpa_supplicant -i wlp4s0 -c /etc/wpa_supplicant.conf -D nl80211,wext -f /tmp/wpa.log &

# 4. 確認連上 AP（這是關鍵！）
iw wlp4s0 link

# 5. 拿 IP
dhclient wlp4s0

# 6. 確認 IP
ip addr show wlp4s0
```