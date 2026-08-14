# Jetson 零複製 AI 影像管線（純 C 風格 + TensorRT）

H.264 Packet → NVDEC(V4L2 ioctl) → EGLImage → CUDA Pointer(零複製) → TensorRT →
VIC(色彩轉換) → NVENC(V4L2 ioctl) → 輸出 bitstream

## 檔案說明

| 檔案 | 說明 | 語言 |
|---|---|---|
| `src/v4l2_decoder.c` | 直接操作 V4L2 M2M ioctl 控制 NVDEC，CAPTURE plane 用 `NvBufSurfaceCreate` 配置 DMA buffer | 純 C |
| `src/v4l2_encoder.c` | 直接操作 V4L2 M2M ioctl 控制 NVENC | 純 C |
| `src/egl_cuda_bridge.c` | DMA fd → EGLImage → CUDA pointer，零複製核心 | 純 C |
| `src/vic_transform.c` | 呼叫 `NvBufSurfTransform` 做 VIC 硬體色彩轉換 | 純 C |
| `src/trt_c_api.cpp` | TensorRT 推論；**TensorRT 官方僅提供 C++ API（`nvinfer1`），無法避免用 .cpp**，但對外用 `extern "C"` 包裝成純 C 介面，`main.c` 呼叫時不需接觸任何 C++ 語法 | C++（僅此檔案） |
| `src/main.c` | 主迴圈，串起整條管線 | 純 C |

## 已知簡化 / 使用前必查項目

1. **Device node 路徑**：`/dev/nvhost-nvdec`、`/dev/nvhost-msenc` 這兩個路徑依 L4T 版本可能不同（部分版本是 `/dev/v4l2-nvdec0`、`/dev/v4l2-nvenc0`）。實機測試前務必執行：
   ```
   v4l2-ctl --list-devices
   ```
   確認實際節點名稱後修改 `main.c` 裡的字串。

2. **動態解析度變更未處理**：正式串流來源解析度可能中途改變，需訂閱 `V4L2_EVENT_SOURCE_CHANGE` 並重新 `VIDIOC_S_FMT` + 重建 buffer，目前範例假設解析度固定。

3. **AI 疊框邏輯未實作**：`trt_output_dev_ptr` 只示範資料流向，實際 bounding box 畫框（寫入 RGBA surface）需另外實作 CUDA kernel。

4. **OUTPUT buffer 排隊策略簡化**：`v4l2_decoder_feed_packet` / `v4l2_encoder_submit_frame` 目前用「先嘗試 DQBUF、失敗就用 index 0」的簡化邏輯，正式版建議維護明確的 free-list，避免 buffer 尚未被 driver 消化完就被覆寫。

5. **Bitstream 輸出**：範例用 `fwrite` 寫到 stdout 示範資料已經產出，正式串流請接 GStreamer `appsrc` 或直接用 RTSP library 推流。

## 編譯

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

需要 Jetson MMAPI 標頭（`/usr/src/jetson_multimedia_api/include`）與對應的 `nvbufsurface` / `nvbufsurftransform` / `nvbuf_utils` 系統函式庫存在（隨 L4T / JetPack 安裝）。

## 執行

```bash
./pipeline_app input.h264 model.engine > output.h264
```
