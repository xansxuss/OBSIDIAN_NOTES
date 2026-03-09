速鎖定誰在偷吃你的 eMMC / NVMe / SD 卡。每一招都有明確戰術意義。

先從全局視角下手，避免盲人摸象：

``` bash
df -hT
```

重點看 overlay、rootfs、mmcblk*、nvme*。Jetson + Docker 時，overlay 幾乎一定是嫌疑犯。

1️⃣ Rootfs 一層層掃描（嵌入式基本功）

``` bash
du -h --max-depth=1 / | sort -hr
```

如果你看到 /usr 或 /var 異常肥，先別急著怪人生，通常是 toolchain 或 log。

進一步鑽：

``` bash
du -h --max-depth=1 /usr | sort -hr
du -h --max-depth=1 /var | sort -hr
```

2️⃣ Yocto 專屬：Build 目錄是空間黑洞

Yocto 沒壞，它只是「很誠實地吃空間」。

``` bash
du -h --max-depth=1 build | sort -hr
```

通常兇手排行：

- tmp/

- sstate-cache/

- downloads/

快速止血（不重編世界）：

``` bash
rm -rf build/tmp/work/*
```

想徹底瘦身但會重建：

``` bash
rm -rf build/tmp build/sstate-cache
```

3️⃣ Docker / containerd：overlay2 的深淵

Jetson 上最常見的失控點。

``` bash
du -h --max-depth=1 /var/lib/docker | sort -hr
```

重點關注：

``` bash
/var/lib/docker/overlay2
/var/lib/docker/containers
/var/lib/docker/volumes
```

容器日誌爆炸檢查：

``` bash
du -h /var/lib/docker/containers/*/*-json.log | sort -hr
```

直接清垃圾（安全版）：

``` bash
docker system df
docker system prune
```

狠一點但清醒：

``` bash
docker system prune -a
```

4️⃣ Jetson 特有：CUDA / TensorRT / Sample 殘留

JetPack 很貼心，也很會留下紀念品。

``` bash
du -h --max-depth=2 /usr/local | sort -hr
```

常見嫌疑區：

- `/usr/local/cuda-*`

- `/usr/src/jetson_multimedia_api`

- `/opt/nvidia`

5️⃣ Log 與 runtime 暗黑物質

嵌入式系統最愛「慢慢撐爆你」。

``` bash
du -h --max-depth=1 /var/log | sort -hr
journalctl --disk-usage
```

立刻降溫：

``` bash
journalctl --vacuum-size=100M
```

6️⃣ 找出超肥檔案（通殺技）

``` bash
find / -xdev -type f -size +100M -exec du -h {} + | sort -hr | head -20
```

-xdev 很重要，不然你會掃進 NFS 或 USB，然後懷疑人生。

7️⃣ OverlayFS / 容器寫爆 rootfs 的真相確認

``` bash
mount | grep overlay
```

如果 rootfs 是 overlay，刪檔不等於回收空間，重啟或重建 image 才是真正回春。

8️⃣ 工程師防復發心法（真的有用）

- Docker 設 log size limit（不然 JSON log 會變成史詩）

- Yocto build 和 rootfs 分磁碟

- Jetson 上不要把 dataset 丟在 /

- eMMC 系統，定期 du /var