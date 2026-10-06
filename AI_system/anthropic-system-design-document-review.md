---
title: Anthropic 系統設計面試 - 文件審查推論系統
source: https://www.reddit.com/r/OfferEngineering/comments/1vz8yjv/anthropic_system_design_interview_document_review/
author: u/Aoki_zhang
published: 2026-08-27
created: 2026-08-28
description: Anthropic 系統設計面試經驗分享，聚焦文件審查推論系統的快取、批次與 GPU 排程設計
categorization: AI inference system design
tags:
  - anthropic
  - system-design
  - llm-inference
  - gpu-scheduling
  - interview-prep
  - batching
  - kv-cache
---

## 背景

來源貼文為 r/OfferEngineering（chillinterview.com 官方版）的面試經驗分享，真實性無法完全驗證，但題目本身的技術內容值得深入拆解與思考。核心情境是設計一個用於「文件審查」的 LLM 推論系統。

## 題目一：設計文件審查推論系統

面試官從一個開放式提示「設計文件審查的推論系統」出發，接著追問三個生產導向的子問題。

### 1. Cache hit rate（快取命中率）

文件審查場景常見「同一份文件被多次查詢」（例如：先摘要、再抓風險條款、再抓合約義務），因此 **prefix caching（前綴快取）** 是關鍵：

- 若同一份文件的 prompt 前綴相同，可快取其 KV cache，後續請求只需計算新增的 suffix（問題本身），大幅降低 prefill 成本。
- 命中率建議以 `命中 token 數 / 總輸入 token 數` 衡量，而非單純用請求數計算，因為部分前綴命中也有價值。
- 提升命中率的排程手段：**cache-aware routing（快取親和路由）**，將同一份文件的請求導向持有該文件 KV cache 的 worker，而非單純輪詢（round robin）。
- 權衡：cache affinity 會犧牲負載平衡彈性，容易讓熱門文件所在的 worker 過載，因此常見設計是「優先嘗試 cache-aware，若目標 worker 過載則 fallback 到負載平衡」。

### 2. Batching strategy（批次策略）

- 建議採用 **continuous batching（in-flight batching）**，而非傳統「等到湊滿固定大小才送出」的 static batching，避免長請求卡住整批短請求。
- 依照 SLA 分桶：互動式查詢與離線批次審查可分開佇列，各自設定不同的 batch timeout（互動流量可設 10-20ms，離線批次可拉長至 100ms+ 換取更高吞吐）。
- Batch size 上限建議依 GPU 記憶體（KV cache 占用）動態調整，而非寫死常數。

### 3. Load balancing before GPU workers（進入 GPU 前的負載平衡）

- 不能只看請求數，需依「預估運算成本」（輸入長度、預期輸出長度）做 **weighted load balancing**，因為文件長度差異可能極大。
- 搭配 worker 定期回報 queue 深度／GPU 記憶體使用率，路由層採 **least-loaded routing**。
- 需與 cache affinity 結合思考，形成「快取命中率、負載均衡、延遲」三者的多目標權衡問題，並無單一正解。

## 題目二：延伸問題 — 8 張 GPU 上同時服務大、小模型

### 限制條件

- 共 8 張 GPU 的 pool。
- 大模型：一個 batch 需要**全部 8 張 GPU**（推測為 tensor parallelism 切分需求，模型過大單卡無法容納）。
- 小模型：一個 batch 只需要 1 張 GPU。
- 兩種模型的單一 batch 推論延遲相同。

### 問題本質

大模型排程時會**獨佔**整個 GPU pool，此時無法同時服務小模型的 batch。這類似作業系統排程中「大任務獨佔資源 vs. 小任務吞吐」的經典問題（convoy effect 的變形），且排程自由度**只存在於時間軸，不存在於空間（GPU 數量）軸**——因為大模型的 GPU 數量需求無法動態調整。

### 可能設計方向

**方案一：時間切分（Time-slicing）**
- 固定時間 slot 輪替執行大、小模型 batch，依兩邊佇列到達率動態調整。
- 缺點：若無優先權機制，任一邊流量暴衝時會發生 starvation（任務餓死）。

**方案二：以佇列壓力做動態優先權**
- 分別追蹤兩條佇列的等待時間、佇列長度、到達率。
- 採用類似 **weighted fair queuing** 或 **aging 機制**，等待越久優先權越高，避免餓死任一方。
- 核心論點：大模型每次排程的機會成本較高（一次犧牲掉可執行 8 個小模型 batch 的資源），因此當大模型佇列已有請求等待時，應傾向優先清空，而非與小模型公平輪替，否則大模型平均延遲會被嚴重拉長。

**方案三：SLA-based 搶佔式排程**
- 若小模型有嚴格延遲 SLA，可設計為「批次間可搶佔」：小模型出現緊急請求時，延後下一個大模型 batch 的排程決策。
- 搶佔粒度只能落在 **batch 之間**，而非 batch 執行中，因為 GPU 運算一旦發射即無法中斷。

### 可延伸的理論深度

- 可用簡單成本函數形式化排程目標，例如 `Cost = Σ(等待時間 × 佇列優先權權重)`，排程器目標是每個時間點選擇使 Cost 增量最小的動作。
- 若已知兩種模型到達率（可用歷史流量估計為 Poisson process），可類比 **M/G/1 佇列理論**分析平均等待時間隨佇列長度的變化，並可引用 **Little's Law** 佐證設計背後的理論基礎。

## 待深化方向

- [ ] 用 Python 模擬不同排程策略（純輪替 vs. aging 優先權 vs. SLA 搶佔）在不同到達率下的平均延遲差異
- [ ] 對照 vLLM / TensorRT-LLM 實際的 continuous batching 與 prefix caching 實作機制
- [ ] 補充 tensor parallelism 與 pipeline parallelism 對「GPU 數量需求無法動態調整」這個限制的實際成因
