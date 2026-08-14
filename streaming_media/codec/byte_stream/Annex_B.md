---
title: Annex B
source: https://gemini.google.com/app/64c330e1e2f995a0?hl=zh-TW
author:
published:
created: 2026-07-09
description: Gemini conversation with 4 messages
tags:
date:
process:
---
在 H.264 和 H.265 (HEVC) 的視訊編碼標準中， **Annex B** 是一種用於「位元流（Byte Stream）」的封裝格式。

簡單來說，當我們要把編碼好的視訊資料透過網路（例如 RTSP 串流）傳送，或是存進某些特定的檔案格式（如 MPEG-TS）時，解碼器需要知道從哪裡開始讀取下一個影像畫面。Annex B 就是為了解決這個問題而設計的。

## 核心設計：起始碼 (Start Code)

Annex B 最顯著的特徵，就是在每個 **NALU (Network Abstraction Layer Unit，網路抽象層單元)** 前面加上一個特定的 **起始碼（Start Code）** 。解碼器透過在連續的位元流中搜尋這個特殊訊號，來切分出一個個獨立的 NALU。

起始碼有兩種長度：

- **`0x00 00 00 01` (4 位元組)** ：通常用於關鍵的 NALU，例如一組資料流的開頭、SPS（序列參數集）、PPS（圖像參數集）以及關鍵影格（IDR 畫面）。
- **`0x00 00 01` (3 位元組)** ：通常用於非關鍵的 NALU，例如一般的 P 影格（前向預測畫面）或 B 影格（雙向預測畫面），用來節省頻寬。

下面的圖表清楚展示了資料從底層的原始資料位元組，一路包裝成 NALU，最後在 Annex B 格式中加上 Start Code 形成連續位元流的結構：

![[Pasted image 20260709112656.png]]

## 防止衝突：防偽位元組 (Emulation Prevention Byte)

既然 Annex B 是靠 `0x000001` 或 `0x00000001` 來辨識邊界，那萬一壓縮後的視訊影像資料（Payload）裡面恰好也出現了這串二進位資料怎麼辦？這樣解碼器不就會誤判嗎？

為了解決這個衝突，標準內建了 **防偽機制** （Emulation Prevention）。在編碼時，如果發現資料中出現以下連續位元組，就會強行塞入一個 **`0x03`** 作為識別：

- `0x00 00 00` $\rightarrow$ 變成 `0x00 00 03 00`
- `0x00 00 01` $\rightarrow$ 變成 `0x00 00 03 01`
- `0x00 00 02` $\rightarrow$ 變成 `0x00 00 03 02`
- `0x00 00 03` $\rightarrow$ 變成 `0x00 00 03 03`

這個被塞進去的 `0x03` 就叫做 **Emulation Prevention Byte** （如上圖中 RBSP 轉換為 EBSP 的過程）。解碼器在解析時，只要看到 `0x00 00 03` ，就會自動把 `0x03` 拿掉，還原成原始資料。

## Annex B 與 AVCC / HVCC 的對比

在實務開發上，你一定會遇到另外一種常見的格式，在 H.264 叫 **AVCC (或稱 MP4 格式)** ，在 H.265 叫 **HVCC** 。這兩者有本質上的不同：

| 特性 | Annex B 格式 | AVCC / HVCC 格式 |
| --- | --- | --- |
| **分割機制** | 使用 **起始碼** (`0x000001`) | 在 NALU 前面用固定長度（通常是 4 節點）來記錄 **NALU 長度** |
| **SPS / PPS 存放** | 直接穿插在視訊流中（每個 IDR 前面都有） | 獨立存放在檔案全域的 Header 中（例如 MP4 的 `avcC` box） |
| **防偽位元組 (`0x03`)** | **有** ，必須存在以防止跟起始碼衝突 | **有** ，雖然沒有起始碼，但為了保持標準一致性依然保留 |
| **常見應用場景** | RTSP/WebRTC 串流、MPEG-TS、`.h264` /`.h265` 裸流檔案 | MP4、MKV、MOV 等媒體封裝貨櫃檔案 |

對於寫 C/C++ 處理硬體解碼（例如使用 FFmpeg、Android MediaCodec 或 Windows Media Foundation）的工程師來說，正確判斷你的硬體解碼器輸入端需要的是 Annex B 還是 AVCC 長度標頭，是處理視訊解碼最常踩到的坑。如果解碼器吐出綠畫面或解析失敗，通常都是這兩種格式沒有轉換正確導致的。

## 一、 AVCC 轉 Annex B 格式（C/C++ 實作）

在 MP4 檔案中，影像資料是以 AVCC 格式儲存。每個 NALU 開頭沒有 `0x00000001` ，而是固定的 **4 節點（Byte）長度標頭** （儲存該 NALU 的純資料長度，大端序 Big-Endian）。

我們要將其轉換為 Annex B，核心邏輯就是： **讀取這 4 節點算出長度，將這 4 節點替換為 `0x00000001` 起始碼，然後指針跳過資料長度，繼續處理下一個 NALU。**

以下為不使用 C++ 標準函式庫（STL）的純 C 格式底層實作：

``` c
#include <stdint.h>
#include <stddef.h>

/**
 * @brief 將 AVCC (Length-prefixed) 緩衝區轉換為 Annex B (Start code) 格式
 * 
 * @param p_buffer      輸入與輸出的資料緩衝區（就地轉換 In-place conversion）
 * @param buffer_size   緩衝區的總總位元組長度
 * @return int          成功返回 0，失敗返回非 0 錯誤碼
 */
int convert_avcc_to_annexb(uint8_t* p_buffer, size_t buffer_size) {
    if (!p_buffer || buffer_size < 4) {
        return -1; 
    }

    size_t current_pos = 0;

    while (current_pos + 4 <= buffer_size) {
        // 1. 讀取 4 位元組的 AVCC 長度標頭 (Big-Endian 大端序)
        uint32_t nalu_length = ((uint32_t)p_buffer[current_pos]     << 24) |
                               ((uint32_t)p_buffer[current_pos + 1] << 16) |
                               ((uint32_t)p_buffer[current_pos + 2] << 8)  |
                               ((uint32_t)p_buffer[current_pos + 3]);

        // 2. 邊界檢查：防止毀損的資料導致記憶體越界訪問
        if (current_pos + 4 + nalu_length > buffer_size) {
            return -2; // 資料長度異常
        }

        // 3. 就地將 4 位元組長度標頭替換為 Annex B 4 位元組起始碼
        p_buffer[current_pos]     = 0x00;
        p_buffer[current_pos + 1] = 0x00;
        p_buffer[current_pos + 2] = 0x00;
        p_buffer[current_pos + 3] = 0x01;

        // 4. 移動指針到下一個 NALU 的開頭
        current_pos += 4 + nalu_length;
    }

    return 0;
}
```

> **注意：**
> 
> 1. 上述程式碼假設 AVCC 的長度標頭固定為 4 節點（這是 MP4 的絕對主流）。
> 2. MP4 檔案全域的 SPS 和 PPS 通常存放在 `avcC` Box 中（ `extradata` ），如果要把整份 MP4 轉成純 Annex B 裸流，必須先手動解析 `avcC` Box，把 SPS/PPS 加上起始碼寫在關鍵影格（IDR）的最前面。

## 二、 H.264 與 H.265 的 NALU Header 結構差異

當我們跳過起始碼後，緊接著的第一個（或前兩個）位元組就是 **NALU Header** 。解碼器就是透過它來判斷目前這個 NALU 到底是 SPS、PPS 還是關鍵影格。

H.264 和 H.265 在這裡的設計有很大的代差：H.264 只用 **1 位元組** ，而 H.265 因為支援更多層級與子層的視訊編碼，擴展到了 **2 位元組** 。

### 1\. H.264 NALU Header 結構 (1 Byte)

位元結構由高到低（Bit 7 到 Bit 0）：

``` bash
+---------------+---------------+-------------------------------+
| Forbidden (1) |  Ref_Idc (2)  |          Type (5)             |
+---------------+---------------+-------------------------------+
```

- **Forbidden Bit (1 bit)** ：禁止位元，必須為 `0` 。如果收到 `1` 表示網路傳輸有錯誤。
- **Nal\_Ref\_Idc (2 bits)** ：重要性指示。如果是 `00` 代表這個 NALU 不會被其他影格當作參考影格（例如 B 影格）；大於 `00` 則代表很重要（例如 SPS, PPS, IDR）。
- **Nal\_Unit\_Type (5 bits)** ：NALU 類型。

#### H.264 常用類型表：

| Type 值 (十進位) | 說明 | 核心用途 |
| --- | --- | --- |
| **5** | Coded Slice of an IDR Picture | **IDR 關鍵影格（I 影格）** |
| **1** | Coded Slice of a Non-IDR Picture | 非關鍵影格（通常是 P 影格或 B 影格） |
| **7** | Sequence Parameter Set (SPS) | 序列參數集（解析度、Profile 等配置） |
| **8** | Picture Parameter Set (PPS) | 圖像參數集（熵編碼模式、初始量化參數等） |

**C/C++ 判斷範例：**

``` C
uint8_t nalu_header = p_buffer[4]; // 假設前 4 節點是起始碼
uint8_t nalu_type = nalu_header & 0x1F; // 取低 5 位元 (0x1F = 00011111)

if (nalu_type == 5) {
    // 這是 H.264 關鍵影格
} else if (nalu_type == 7) {
    // 這是 H.264 SPS
}
```

### 2\. H.265 (HEVC) NALU Header 結構 (2 Bytes)

位元結構由高到低（共 16 bits）：

``` bash
+---------------+-----------------------+---------------+---------------+
| Forbidden (1) |        Type (6)       |   LayerId (6) |   Tid (3)     |
+---------------+-----------------------+---------------+---------------+
```

- **Forbidden Bit (1 bit)** ：一樣必須為 `0` 。
- **Nal\_Unit\_Type (6 bits)** ：NALU 類型（擴展到 6 位元，能容納更多新型影格）。
- **Nuh\_Layer\_Id (6 bits)** ：多層編碼多留的 ID，目前通常為 `0` 。
- **Nuh\_Temporal\_Id\_Plus1 (3 bits)** ：時域分層識別（用於可變影格率）。通常值為 `1` 。

#### H.265 常用類型表：

H.265 的關鍵影格分得非常細，但一般實務上，Type 落在 **16 到 23 之間** 的都屬於關鍵影格家族（RAP, Random Access Point）。

| Type 值 (十進位) | 說明 | 核心用途 |
| --- | --- | --- |
| **19** | IDR\_W\_RADL | **IDR 關鍵影格** （最常見的關鍵影格） |
| **20** | IDR\_N\_LP | **IDR 關鍵影格** （沒有前向參考） |
| **1** | TRAIL\_R | 一般前向參考影格（通常是 P 影格） |
| **32** | VPS\_NUT (VPS) | 視訊參數集（H.265 新增，高於 SPS） |
| **33** | SPS\_NUT (SPS) | 序列參數集 |
| **34** | PPS\_NUT (PPS) | 圖像參數集 |

**C/C++ 判斷範例：** H.265 的 Type 跨越了第一個位元組的後半段與第二個位元組。我們需要把第一個位元組往右移 1 位元，並遮蔽掉最高位的禁止位元：

``` C
uint8_t nalu_header_byte1 = p_buffer[4]; // 起始碼後的第一個位元組
uint8_t nalu_type = (nalu_header_byte1 >> 1) & 0x3F; // 右移 1 位元，取 6 位元 (0x3F = 00111111)

if (nalu_type == 19 || nalu_type == 20) {
    // 這是 H.265 關鍵影格
} else if (nalu_type == 32) {
    // 這是 H.265 特有的 VPS
} else if (nalu_type == 33) {
    // 這是 H.265 SPS
}
```

## 總結開發小撇步

1. **快速特徵識別：**
	- H.264 的 SPS 開頭通常是 `0x67` ，PPS 是 `0x68` ，IDR 關鍵影格是 `0x65` 或 `0x25` （因為包含了 Ref\_Idc 的位元）。
	- H.265 的 VPS 開頭通常是 `0x40` ，SPS 是 `0x42` ，PPS 是 `0x44` ，IDR 是 `0x26` 。
2. **記憶體操作安全：** 在處理非標準函式庫的底層 C 指針時，拿到 `nalu_length` 後 **務必** 先與 `buffer_size` 做減法或加法邊界校驗，串流傳輸常因網路丟包導致資料截斷，沒做校驗極易造成 Segmentation Fault。