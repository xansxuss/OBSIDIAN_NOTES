---
title: "Dragonwing IQ vs Q"
source: "https://chatgpt.com/c/6a7182ba-1724-83ee-b431-d4a10476894d"
author:
published:
created: 2026-08-04
description: "ChatGPT conversation with 2 messages"
tags:
  - "clippings"
---
Dragonwing IQ Family and Dragonwing Q Family比較

---

如果是做 **AI Edge、機器人、工業視覺、嵌入式 Linux** ，這兩個系列很容易搞混，因為都屬於 Qualcomm 的 **Dragonwing** 品牌，但定位其實完全不同。

| 項目 | Dragonwing IQ Family | Dragonwing Q Family |
| --- | --- | --- |
| 市場定位 | 工業 AI / Robotics / Vision | 一般 Industrial IoT / Enterprise |
| 主要特色 | 高 AI 算力 + Functional Safety + 即時控制 | 平衡 CPU、AI、Connectivity |
| AI Performance | 約 40~350 Dense TOPS（IQ8~IQ10） | 約 1~12 TOPS（Q6 等） |
| 即時 MCU | ✔ 內建 Real-time MCU | 通常沒有獨立即時 MCU |
| Safety | 支援 SIL、ECC、安全島等 | 一般 IoT 等級 |
| 溫度範圍 | \-40°C ~ 125°C | Industrial 等級，但安全功能較少 |
| 使用情境 | AMR、Robot、PLC、Machine Vision | Gateway、POS、Handheld、HMI、Edge Box |
| 軟體 | Yocto、Ubuntu、ROS、Industrial SDK | Yocto、Android、Ubuntu 等 |

資料來自 Qualcomm 官方產品說明。 [iq8 series](https://www.qualcomm.com/internet-of-things/products/iq8-series?utm_source=chatgpt.com),[iq9 series](https://www.qualcomm.com/internet-of-things/products/iq9-series?utm_source=chatgpt.com),[q6 series](https://www.qualcomm.com/internet-of-things/products/q6-series?utm_source=chatgpt.com)

---

## Dragonwing IQ Family

IQ = **Industrial Intelligence**

這系列可以視為 Qualcomm 專門為工業 AI 推出的旗艦平台。

例如：

- IQ8
- IQ9
- IQ10

主要特點：

### 1\. AI 很強

例如

| 型號 | AI |
| --- | --- |
| IQ8 | 40 TOPS |
| IQ9 | 100 TOPS |
| IQ10 | 350 Dense TOPS（700 Sparse TOPS） |

非常適合：

- YOLO
- Segment Anything
- LLM
- VLM
- 多 Camera AI

[iq8 series](https://www.qualcomm.com/internet-of-things/products/iq8-series?utm_source=chatgpt.com),[iq9 series](https://www.qualcomm.com/internet-of-things/products/iq9-series?utm_source=chatgpt.com),[iq10 series](https://www.qualcomm.com/internet-of-things/products/iq10-series?utm_source=chatgpt.com)

---

### 2\. 有 Real-time MCU

不像 Snapdragon 只有 Application Processor。

IQ 還內建：

- Real-time Core
- Safety Island
- CAN FD
- TSN Ethernet

因此可以同時：

```
Linux
    │
AI Inference

RT MCU
    │
Motor Control
Encoder
CAN
PLC
```

很適合：

- 機器人
- AGV
- AMR
- 工業控制

---

### 3\. Safety

IQ 系列加入大量工業需求：

- ECC Memory
- Functional Safety
- Error Detection
- Lockstep
- Safety Island

這是車用、工業自動化最在意的。

---

## Dragonwing Q Family

Q Family 可以想成：

> 工業版 Snapdragon。

例如：

- QCS6490
- QCS5430
- QCS8550
- Q6 Series

特色：

- AI 不錯
- CPU 不錯
- GPU 不錯
- Connectivity 很完整
- 功耗較低
- 成本較低

但：

沒有 IQ 那麼偏 Robotics。

---

## Q Family 比較像什麼？

比較適合：

- POS
- Kiosk
- Smart Camera
- Gateway
- Handheld
- Industrial Tablet
- Retail

例如：

```
Camera
    ↓
AI Detect
    ↓
Upload Cloud
```

這種就很適合。

---

## IQ 比較像什麼？

例如：

```
8 Cameras
LiDAR
Radar
IMU

        ↓

Fusion

        ↓

YOLO

        ↓

Path Planning

        ↓

Motor Control
```

這就是 IQ 的設計目標。

---

## Connectivity

兩者都有：

- PCIe
- USB
- Ethernet
- MIPI CSI
- DisplayPort
- LPDDR5
- UFS
- Wi-Fi
- Bluetooth

但 IQ 通常多：

- TSN Ethernet
- CAN FD
- Safety GPIO
- Real-time Peripheral

---

## Linux 支援

兩者都支援：

- Linux Yocto
- Ubuntu

IQ 另外更強調：

官方標示 IQ 系列可提供長期產品生命週期支援。 [iq8 series](https://www.qualcomm.com/internet-of-things/products/iq8-series?utm_source=chatgpt.com),[Qualcomm Dragonwing](https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_G_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf?utm_source=chatgpt.com)

---

## AI Vision 差異

若你在做：

則：

**IQ9 / IQ10** 能同時處理更多攝影機、更大的 AI 模型以及更高吞吐量，並針對工業環境最佳化； **Q Family** 則偏向中小型 Edge AI 與 IoT 應用。[iq9 series](https://www.qualcomm.com/internet-of-things/products/iq9-series?utm_source=chatgpt.com) [Qualcomm Dragonwing](https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_G_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf?utm_source=chatgpt.com)

---

## 一句話總結

- **Dragonwing Q Family** ：偏向通用 Industrial IoT 平台，重視成本、連線能力與平衡效能，適合閘道器、智慧終端、工業平板等。
- **Dragonwing IQ Family** ：偏向高階工業 AI 與 Robotics 平台，除了更高 AI 算力外，還加入即時控制、功能安全（Functional Safety）及工業級周邊，適合機器人、AMR、自主設備與機器視覺系統。

[^1]: [qualcomm.com](https://www.qualcomm.com/internet-of-things/products/iq8-series?utm_source=chatgpt.com)

[^2]: [qualcomm.com](https://www.qualcomm.com/internet-of-things/products/iq9-series?utm_source=chatgpt.com)