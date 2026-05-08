`nv3dsink` 是 **NVIDIA Jetson 平台**（特別是使用 JetPack + GStreamer 的系統）上專用的硬體加速 **影像顯示 (video sink)** 元件。它是 NVIDIA GStreamer plugin (`nvgstplugins`) 的一部分，用來直接將影像透過 GPU 顯示在螢幕上（通常是 EGL 或 framebuffer）。

---

### 🧩 基本介紹

|屬性|說明|
|---|---|
|元件名稱|`nv3dsink`|
|所屬 Plugin|`nvgstplugins`|
|支援平台|Jetson (Nano / Xavier / Orin 等)|
|作用|顯示 GPU 記憶體中的畫面（通常是 NVMM buffer）|
|Memory Type|支援 `video/x-raw(memory:NVMM)`|
|硬體加速|是（使用 VIC / GPU compositing）|

---

### 🚀 常見使用範例

#### 1️⃣ 播放本地影片

``` cpp
gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nv3dsink
```

#### 2️⃣ 播放 RTSP 串流

``` cpp
gst-launch-1.0 rtspsrc location=rtsp://192.168.0.10:8554/stream latency=200 ! \     rtph264depay ! h264parse ! nvv4l2decoder ! nv3dsink
```

#### 3️⃣ 顯示 CUDA 或 OpenCV 處理後的畫面

如果前面 pipeline 使用 `nvvideoconvert`：

``` cpp
... ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nv3dsink
```

---

### ⚙️ 常見屬性（`gst-inspect-1.0 nv3dsink`）

| 屬性                                                 | 型別      | 說明              |
| -------------------------------------------------- | ------- | --------------- |
| `sync`                                             | boolean | 是否同步播放（預設 true） |
| `max-lateness`                                     | gint64  | 允許延遲時間（預設 -1）   |
| `show-preroll-frame`                               | boolean | 是否顯示預捲畫面        |
| `display-id`                                       | int     | 選擇顯示螢幕（多顯示器情況）  |
| `overlay-x`, `overlay-y`, `overlay-w`, `overlay-h` | int     | 顯示位置與大小         |

---

### 🔄 與其他 sink 比較

| Sink            | 平台         | 特性                      |
| --------------- | ---------- | ----------------------- |
| `ximagesink`    | CPU        | 一般 X11 視窗顯示（無硬體加速）      |
| `nveglglessink` | GPU / EGL  | OpenGL 硬體加速顯示，支援透明與複合   |
| **`nv3dsink`**  | Jetson GPU | 專為 Jetson 設計，效能最佳，零拷貝顯示 |
| `fakesink`      | 無輸出        | 偵錯用                     |

---

### 🧠 小技巧

- 若你想在 headless Jetson 系統（無 GUI）顯示，可用：

    ``` cpp
    nv3dsink sync=false
    ```

    讓它不等待 vsync（適合測速或背景執行）
    
- 若要在 OpenCV / CUDA 與 GStreamer 混用，可搭配：

    ``` cpp
    appsink ! cv::cuda::GpuMat
    ```

    或反向：

    ``` cpp
    appsrc ! nvvideoconvert ! nv3dsink
    ```

- 若你想要在 x86 模擬 Jetson pipeline，可改用：

    ``` cpp
    nveglglessink
    ```

    因為 `nv3dsink` 只存在於 Jetson 的 JetPack SDK。
