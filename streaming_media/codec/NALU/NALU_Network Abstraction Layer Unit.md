在 H.264 與 H.265 (HEVC) 視訊編碼標準中，**NALU**（Network Abstraction Layer Unit，網路抽象層單位）是==視訊資料傳輸與儲存的**基本核心單元**==。

NALU 的核心結構

每個 NALU 都是由**標頭（Header）**與**負載（Payload）**組成：

- **NALU Header**：說明此資料包的**類型**與**重要性**（H.264 佔 1 位元組；H.265 佔 2 位元組）。
- **RBSP (Raw Byte Sequence Payload)**：實際的封裝資料，長度不固定。

---

NALU 的兩大主要分類

1. VCL NALU (Video Coding Layer)

- **核心功能**：存放實際壓縮後的**視訊畫面資料**。
- **常見內容**：IDR 關鍵影格（I-frame）、非關鍵影格（P-frame、B-frame）的切片（Slice）數據。

2. Non-VCL NALU

- **核心功能**：存放解碼器不可或缺的**控制與參數資訊**。
- **關鍵類型**：
    - **SPS (Sequence Parameter Set)**：序列參數集。記錄解析度、影格數、Profile 與 Level 等全域資訊。
    - **PPS (Picture Parameter Set)**：圖像參數集。記錄熵編碼類型、分塊對齊等單張畫面的控制參數。
    - **VPS (Video Parameter Set)**：視訊參數集（**僅限 H.265/HEVC**）。負責多層次視訊（如 3D 或可伸縮編碼）的參數協調。
    - **SEI (Supplemental Enhancement Information)**：補充增強資訊。存放時間戳記、字幕或旋轉等輔助數據。

---

NALU 的兩種傳輸型態

為了讓解碼器在持續接收的位元流中正確識別出獨立的 NALU，通常會有以下兩種封裝方式：

1. Annex B 格式（常見於本地檔案如 `.h264`, `.mp4` 或 TS 串流）

- 每個 NALU 前面會加上固定長度的**起始碼（Start Code）**。
- 起始碼為 `0x000001`（影格內部的 Slice）或 `0x00000001`（一訊框的開頭，如 SPS/PPS/IDR）。

2. RTP 封裝格式（常見於網路即時串流如 WebRTC, RTSP）

- 直接將 NALU Header 與 Payload 打包進 RTP 封包。
- **不需要起始碼**，改由 RTP 標頭中的長度欄位或網路層協定來切分邊界。

---