1. image_format : image/jpeg
                  video/mpeg {mpegversion: [(int)1,(int)2,4]}
                  video/mpegts
                  video/x-bayer {format: [(string)bggr, (string)gbrg, (string)grbg, (string)rggb]}
                  video/x-dv
                  video/x-fwht
                  video/x-h263
                  video/x-h264 {stream-format: [(string)byte-stream, (string)avc]}
                  video/x-h265
                  video/x-pwc1
                  video/x-pwc2
                  video/x-raw {format: [(string)RGB16, (string)BGR, (string)RGB, (string)ABGR, (string)xBGR, (string)RGBA, (string)RGBx, (string)GRAY8, (string)GRAY16_LE, (string)GRAY16_BE, (string)YVU9, (string)YV12, (string)YUY2, (string)YVYU, (string)UYVY, (string)Y42B, (string)Y41B, (string)YUV9, (string)I422, (string)NV12_64Z32, (string)NV24, (string)NV12_16L32S, (string)NV61, (string)NV16, (string)NV21, (string)NV12, (string)I420, (string)ARGB, (string)xRGB, (string)BGRA, (string)BGRx, (string)BGR15, (string)RGB15]}
                  video/x-sonix
                  video/x-vp8
                  video/x-vp9
                  video/x-wmv
                  Video/x-raw(format:Interlaced) {format:[(string)RGB16, (string)BGR, (string)RGB, (string)ABGR, (string)xBGR, (string)RGBA, (string)RGBx, (string)GRAY8, (string)GRAY16_LE, (string)GRAY16_BE, (string)YVU9, (string)YV12, (string)YUY2, (string)YVYU, (string)UYVY, (string)Y42B, (string)Y41B, (string)YUV9, (string)I422, (string)NV12_64Z32, (string)NV24, (string)NV12_16L32S, (string)NV61, (string)NV16, (string)NV21, (string)NV12, (string)I420, (string)ARGB, (string)xRGB, (string)BGRA, (string)BGRx, (string)BGR15, (string)RGB15]}

URI 處理能力：
    - 支援的 URI 協議：v4l2
        - v4l2src 支援透過 URI 協議從指定的 Video4Linux2 裝置（例如 /dev/video0）擷取影像流。
擴充功能：
    - 支援I\O：
      - GstURIHandler：允許元素處理 URI。
      - GstTuner：可以控制和配置調諧器。
      - GstColorBalance：允許調整顏色平衡。
      - GstVideoOrientation：處理視頻方向的變換。
Pads：
    - SRC pad：
      - src：輸出影像流的源，支援多種影像格式，像是 JPEG、MPEG、Raw 格式等。
      - 這個 pad 是從設備輸出的視頻流。
元素屬性：
1. blocksize：每個緩衝區讀取的字節大小，默認為 4096。
2. brightness：圖片的亮度，範圍為 -2147483648 到 2147483647，默認為 0。
3. contrast：圖片對比度，範圍為 -2147483648 到 2147483647，默認為 0。
4. device：視頻設備的路徑，默認為 /dev/video0。
5. device-fd：設備的文件描述符，範圍為 -1 到 2147483647，默認為 -1。
6. device-name：設備的名稱，默認為 null。
7. do-timestamp：是否對緩衝區應用當前流時間，默認為 false。
8. extra-controls：額外的 v4l2 控制參數（CIDs），可以用來設置更詳細的設備配置。
9. force-aspect-ratio：強制應用像素長寬比，默認為 true。
10. hue：色調或色平衡，範圍為 -2147483648 到 2147483647，默認為 0。
11. io-mode：I/O 模式選項，包括自動、讀寫、映射等。
12. norm：視頻標準（例如 NTSC、PAL 等），默認為 none。
13. num-buffers：輸出緩衝區數量，默認為 -1，表示無限制。

訊號：
    - "prepare-format"：當元素準備格式時觸發，允許用戶定義如何處理視頻流的格式。
裝置類型標誌：
    - flags：標誌顯示設備是否支持視頻捕捉、回放、疊加等功能。
    - 支援捕捉、回放、視頻覆蓋等功能。

extra-controls : 
    1. 查詢選項 : v4l2-ctl --list-ctrls -d /dev/video*
    - example : 
        用戶控制項 (User Controls)
            亮度 (brightness)
            設置亮度範圍是 0 到 100，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="brightness=50" ! videoconvert ! autovideosink ```
            對比度 (contrast)
            設置對比度範圍是 0 到 10，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="contrast=5" ! videoconvert ! autovideosink ```
            飽和度 (saturation)
            設置飽和度範圍是 0 到 10，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="saturation=8" ! videoconvert ! autovideosink ```
            色相 (hue)
            設置色相範圍是 0 到 100，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="hue=20" ! videoconvert ! autovideosink ```
            白平衡自動 (white_balance_automatic)
            設置是否啟用自動白平衡。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="white_balance_automatic=0" ! videoconvert ! autovideosink ```
            曝光 (exposure)
            曝光範圍是 -40 到 40，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="exposure=10" ! videoconvert ! autovideosink ```
            電源線頻率 (power_line_frequency)
            設置為自動模式 (Auto) 或其他選項（0-3）。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="power_line_frequency=1" ! videoconvert ! autovideosink ```
            白平衡溫度 (white_balance_temperature)
            設置白平衡的色溫範圍是 2700 到 6500，默認值是 6500。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="white_balance_temperature=5000" ! videoconvert ! autovideosink ```
            銳度 (sharpness)
            設置銳度範圍是 0 到 10，默認值是 0。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="sharpness=7" ! videoconvert ! autovideosink ```
            ISO (iso)
            設置 ISO 範圍是 100 到 6400，默認值是 100。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="iso=800" ! videoconvert ! autovideosink ```
        相機控制項 (Camera Controls)
            自動曝光 (auto_exposure)
            設置為自動模式或其他選項（0 或 1）。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="auto_exposure=1" ! videoconvert ! autovideosink ``` 
            曝光時間 (exposure_time_absolute)
            設置曝光時間的絕對值，範圍是 100,000 到 100,000,000。            ` ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="exposure_time_absolute=50000000" ! videoconvert ! autovideosink ```
            焦距 (focus_absolute)
            設置焦距，範圍是 0 到 255。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="focus_absolute=50" ! videoconvert ! autovideosink ```
            自動連續焦點 (focus_automatic_continuous)
            設置是否啟用自動連續焦點（0 或 1）。
            ``` gst-launch-1.0 v4l2src device=/dev/video0 extra-controls="focus_automatic_continuous=1" ```
io-mode : I/O mode
    - flags: readable, writable
    - Enum "GstV4l2IOMode" Default: 0, "auto"
    - (0): auto             - GST_V4L2_IO_AUTO 自動選擇最佳模式。
    - (1): rw               - GST_V4L2_IO_RW 使用基本的讀寫操作模式。
    - (2): mmap             - GST_V4L2_IO_MMAP 使用記憶體映射模式，通常用於高效處理。
    - (3): userptr          - GST_V4L2_IO_USERPTR 允許應用程式提供緩衝區。
    - (4): dmabuf           - GST_V4L2_IO_DMABUF 用於高效的 DMA buffer 操作，通常涉及硬體加速。
    - (5): dmabuf-import    - GST_V4L2_IO_DMABUF_IMPORT 用於匯入外部的 DMA buffer。
