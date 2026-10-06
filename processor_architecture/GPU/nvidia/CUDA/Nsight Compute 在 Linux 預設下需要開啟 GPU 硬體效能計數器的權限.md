## Nsight Compute 在 Linux 預設下需要「開啟 GPU 硬體效能計數器的權限」，不然就會被驅動鎖起來（NVIDIA 從驅動 418 開始加的安全限制）。

1. 臨時開啟（root）
直接在 root 下執行：

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 --gom=0
sudo nvidia-smi -i 0 --reset-gpu-clocks
```

然後打開 performance counter：

```bash
sudo nvidia-smi -i 0 --query-gpu=perf_policy --format=csv
```

但這個指令實際上在新驅動裡被移到 /proc 路徑設定，對 Nsight 來說更推薦 udev 方法。

2. 官方推薦方式（udev rule）
NVIDIA 文件建議用 udev rule 給當前使用者權限：
建立檔案：

```bash
sudo nano /etc/udev/rules.d/70-nvidia-perf.rules
```

內容：

```udev
# Allow access to NVIDIA performance counters
SUBSYSTEM=="drm", KERNEL=="card*", ATTR{device}=="0x10de*", MODE="0666"
```

儲存後：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

3. Docker 環境注意
你是在 Docker 裡跑（root@6a33a1cb4c53），
Nsight Compute 要抓硬體計數器必須：

Docker 啟動加 --cap-add=SYS_ADMIN 和 --device /dev/nvidia*

驅動要允許容器內讀 performance counter（同樣需要宿主機開 udev rule）

範例：

```bash
docker run --rm -it \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  nvcr.io/nvidia/nsight-compute:latest
```

4. 如果你只是要 Rpt / SMem 不看全性能
你可以直接用 nvcc 編譯 kernel 時加：

```bash
nvcc -Xptxas=-v your_kernel.cu
```

這會輸出：

```python
ptxas info    : Used 64 registers, 32768 bytes smem, 0 bytes cmem[0]
```