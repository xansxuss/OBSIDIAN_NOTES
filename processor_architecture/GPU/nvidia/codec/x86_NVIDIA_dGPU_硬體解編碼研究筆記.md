

> 範圍：本筆記只涵蓋 **x86 桌機/伺服器 + 獨立顯卡（dGPU）** 這條線，用的是 NVIDIA Video Codec SDK（`cuviddec.h` / `nvcuvid.h` / `nvEncodeAPI.h`）。 Jetson（Xavier/Orin，V4L2 M2M / Jetson Multimedia API）是完全不同的架構， 另外整理。

---

## 1. 背景與目標

- 目的：用 NVDEC（硬體解碼）+ NVENC（硬體編碼）打造一條「解碼 → （可選）處理 → 編碼」的 pipeline，畫面全程留在 GPU 記憶體，不落地到主機（zero host copy）。
- C++ 風格限制：不使用 STL 容器（`std::vector`/`std::queue`/`std::mutex` 等）， 改用固定大小陣列、自製環狀緩衝、POSIX `pthread`。
- 最終支援項目：
    - 本地檔案 batch 轉檔
    - `rtsp://` 即時串流轉檔
    - H.264 / HEVC 雙輸出
    - 多路平行處理（多路 thread 共用同一個 CUDA primary context）

---

## 2. 核心技術棧

|元件|用途|標頭檔|
|---|---|---|
|NVDEC（解碼）|硬體解碼 H.264/HEVC|`cuviddec.h`、`nvcuvid.h`|
|NVENC（編碼）|硬體編碼 H.264/HEVC|`nvEncodeAPI.h`|
|CUDA Driver API|管理 context、裝置記憶體|`cuda.h`|
|libavformat（FFmpeg）|拆封裝容器（mp4/mkv/rtsp）|`libavformat/avformat.h`|
|libavcodec bsf|AVCC → Annex-B 格式轉換|`libavcodec/bsf.h`|
|pthread|多路平行、環狀緩衝鎖|`pthread.h`|

**關鍵認知**：NVDEC 走 **callback 模型**（`cuvidCreateVideoParser` 設定 `pfnSequenceCallback`/`pfnDecodePicture`/`pfnDisplayPicture`），跟 Jetson 那邊 V4L2 M2M 的「output/capture plane 佇列」模型完全不同，兩者程式碼不能互通。

---

## 3. 資料流架構

```
檔案/RTSP
   │  (libavformat demux)
   ▼
Demuxer ──► 取出 Annex-B bitstream (H.264/HEVC)
   │
   ▼
MyDecoder (cuvid)
   │  pfnDisplayPicture callback → 存進自製環狀緩衝
   ▼
PopFrame() → CUdeviceptr (NV12，GPU 記憶體，Y 平面接著 UV 平面)
   │
   │  ★ 全程不 cudaMemcpy 回主機 ★
   ▼
NvEncoder (nvEncodeAPI)
   │  nvEncRegisterResource() 直接註冊這個 CUdeviceptr
   ▼
輸出 .h264 / .hevc（raw elementary stream）
```

畫面從解碼輸出到編碼輸入，中間**沒有任何格式轉換**：NVDEC 輸出的 NV12 剛好就是 NVENC 吃的格式，`CUdeviceptr` 直接傳遞即可。

---

## 4. 專案檔案結構與各自職責

```
nvdec_demo/
├── CMakeLists.txt
├── src/
│   ├── CodecType.h        # 平台中立的 codec 列舉（H264/HEVC），
│   │                      # x86/Jetson 共用，Demuxer 不綁定任何 SDK 型別
│   ├── Demuxer.h/.cpp     # libavformat 拆封裝 + bsf 轉 Annex-B，支援 rtsp://
│   ├── MyDecoder.h/.cpp   # cuvid wrapper：不用 STL，環狀緩衝存 callback 結果
│   ├── NvEncoder.h/.cpp   # nvEncodeAPI wrapper：register 快取、H264/HEVC
│   ├── Pipeline.h/.cpp    # 單路 demux→decode→encode 迴圈，pthread 進入點
│   ├── BatchSurface.h     # 參考 NvBufSurface 結構設計的批次描述容器（框架，
│   │                      # 尚未接進 Pipeline，留給未來多路合批後處理用）
│   ├── main.cpp           # 解析命令列、建立多路 thread（nvdec_transcode）
│   ├── debug_ppm_main.cpp # 單路解碼除錯工具（存 PPM 驗證畫面正確性）
│   └── nv12_to_rgb.cu/.h  # CUDA kernel：NV12→RGB（僅除錯工具用）
```

兩個執行檔：

- **`nvdec_transcode`**：正式功能，demux→NVDEC→NVENC，支援多路/RTSP/雙編碼
- **`nvdec_ppm_debug`**：除錯用，解碼後轉 RGB 存成 PPM 圖檔肉眼驗證

---

## 5. 關鍵設計決策與理由

### 5.1 為什麼 callback 結果要放進自製環狀緩衝？

`cuvidParseVideoData()` 內部同步觸發 `pfnDisplayPicture` callback，但呼叫時機 跟送入壓縮資料的時機不一定一致（B frame 需要重排序），所以要有個緩衝把 callback 吐出來的畫面暫存，供外部迴圈之後用 `PopFrame()` 依序取出。 用固定大小陣列 + `pthread_mutex_t` 取代 `std::queue`。

### 5.2 NVENC 的 register/map 生命週期

- `nvEncRegisterResource`：真正貴的操作，要跟 driver 建立映射表
- `nvEncMapInputResource`/`Unmap`：相對便宜，每張畫面都要做

**最佳化重點**：因為 NVDEC 內部的 decode surface 位址集合是有限的 （`ulNumDecodeSurfaces` 固定張數），`NvEncoder` 用固定大小陣列做 「CUdeviceptr → 已註冊資源」的快取，只在第一次遇到某個位址時才 `RegisterResource`，之後重複出現的位址直接複用，只做 map/unmap。

### 5.3 Demuxer 平台中立化

一開始 `Demuxer.h` 直接 include `cuviddec.h` 取用 `cudaVideoCodec` 列舉， 後來為了讓 Jetson 那邊的 `V4l2Decoder` 也能共用同一份 `Demuxer`， 改成獨立的 `CodecType.h`（`VideoCodecType::kH264/kHEVC`）， x86/Jetson 各自在自己的 `Pipeline`/`main` 裡把這個列舉轉成該平台的型別。

### 5.4 多路平行的 CUDA context 策略

用 `cuDevicePrimaryCtxRetain()` 拿一個 primary context，所有 thread 共用， 每個 thread 一開始呼叫 `cuCtxSetCurrent()` 把它設成自己的 current context。 沒有幫每個 thread 各建一個獨立 context（多 GPU 情境才需要考慮這件事）。

### 5.5 RTSP 支援

底層還是 `libavformat`，只多做兩件事：

- `av_dict_set(&options, "rtsp_transport", "tcp", 0)`：避免 UDP 掉包
- `avformat_network_init()` / `avformat_network_deinit()`：全域呼叫一次

---

## 6. 使用方式速查

```bash
# 編譯
mkdir build && cd build
cmake .. -DNVCODEC_DIR=/path/to/Video_Codec_SDK_xx.x.x
make -j

# 單路檔案轉 H.264
./nvdec_transcode input.mp4 output.h264 h264

# 單路轉 HEVC，指定位元率
./nvdec_transcode input.mp4 output.hevc hevc 4000000

# 多路：本地檔案 + RTSP 同時處理
./nvdec_transcode input.mp4 out1.h264 h264 6000000 \
                   rtsp://192.168.1.10/stream1 cam1.hevc hevc 4000000

# 除錯：驗證解碼畫面正確性
./nvdec_ppm_debug input.mp4 5
```

---

## 7. 已知限制 / 待辦事項

- [ ] `BatchSurface.h` 只是框架，還沒真正接進 `Pipeline.cpp`； 而且要注意 **NVENC/NVDEC 本身不支援跨路合批**，這個結構的價值 在於未來接 CUDA 後處理 kernel 或 TensorRT 批次推論，不是拿來加速 解碼/編碼本身
- [ ] 沒有處理 RTSP 斷線重連，目前斷線該路 thread 直接結束
- [ ] 沒有保留音訊軌（目前只處理視訊）
- [ ] 全部程式碼**沒有實機編譯測試過**（開發環境沒有 NVIDIA GPU）， 已知可能需要現場調整：SDK 版本造成的 preset GUID 差異、 struct 欄位名稱在不同 SDK 版本間的細微差異

---

## 8. 參考資料

- NVIDIA Video Codec SDK 官方下載：[https://developer.nvidia.com/video-codec-sdk](https://developer.nvidia.com/video-codec-sdk)
- NVDEC/NVENC 官方 Programming Guide（`Video_Codec_SDK` 內附 PDF）
- FFmpeg libavformat/libavcodec 官方文件：[https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)