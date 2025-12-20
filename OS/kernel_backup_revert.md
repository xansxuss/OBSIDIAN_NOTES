✅ 方法 1：完整複製 kernel 與 modules（最快、最通用）

備份 /usr/lib/modules/<version> + /boot 中的 kernel 檔（Ubuntu 24.04 的 kernel 主體搬到 /usr/lib/modules 裡，但 /boot 還是有必要）。

備份指令：

``` bash
sudo mkdir -p /home/eray/kernel_backup/6.14.0-35
sudo cp -a /usr/lib/modules/6.14.0-35-generic /home/eray/kernel_backup/6.14.0-35/
sudo cp -a /boot/*6.14.0-35* /home/eray/kernel_backup/6.14.0-35/ 2>/dev/null || true
```

成品：

``` bash
~/kernel_backup/6.14.0-35/
```

包含：

- kernel modules
- System.map
- config
- initrd
- vmlinuz（如果系統有放在 /boot）

🔥 方法 2：打包成壓縮檔（最乾淨）

``` bash
cd /usr/lib/modules
sudo tar -czvf ~/kernel_backup/linux-6.14.0-35-backup.tar.gz 6.14.0-35-generic
```

（如果 /boot 有 kernel file，也一起進 tar）

``` bash
sudo tar -rzvf ~/kernel_backup/linux-6.14.0-35-backup.tar.gz /boot/*6.14.0-35* 2>/dev/null || true
```

🛡️ 方法 3：完整 Debian Package 備份（最專業、可直接還原）

把 kernel package 本體重新抓回來備份：

``` bash
mkdir -p ~/kernel_backup/deb
apt download linux-image-6.14.0-35-generic linux-headers-6.14.0-35-generic
mv *.deb ~/kernel_backup/deb/
```

這方法的好處：

未來要還原時，只需要 sudo dpkg -i *.deb

不必靠 apt repository 是否還存在舊版本

🔄 若要還原 kernel（完整指令）

（以方式 1 / 2 的備份為例）

``` bash
sudo cp -a ~/kernel_backup/6.14.0-35/6.14.0-35-generic /usr/lib/modules/
sudo cp -a ~/kernel_backup/6.14.0-35/*6.14.0-35* /boot/
sudo update-initramfs -c -k 6.14.0-35-generic
sudo update-grub
```