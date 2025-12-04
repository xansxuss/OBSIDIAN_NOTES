
## ★ ByteTrack 的核心觀念：

高分框 = 我很確定的觀測  
低分框 = 可能看起來爛，但仍然是世界的一部分

大多數 tracker（SORT、DeepSORT）在 tracking 的第一步就把低分框砍掉，  
ByteTrack 反其道而行：**保留所有分數 > 低門檻的框**，  
然後讓高分框先跟 active tracks 配對，  
配不到的 track 再用低分框補洞。

這個設計等於「給目標一個復活機會」。

---

## ★ ByteTrack 的完整流程（概念上）

我會用接近工程實作的方式描述，不會掉進學術文青模式。

### 1 — 偵測輸入：兩種門檻

你會準備兩種 score threshold：

- 高門檻（high score threshold）→ 高品質偵測
- 低門檻（low score threshold）→ 含雜訊的偵測

把偵測框分兩堆：  
high_det = score > high_thresh  
low_det = low_thresh < score < high_thresh

高分框是你真正信任的；  
低分框只是在你眼角偷瞄到的那種模糊人影。

---

### 2 — 用高分框做第一次匈牙利配對

把 active tracks 跟 high_det 做 IOU-based matching。

高分框能讓 ID 比較穩定，因為品質好、位置準。

配到的：更新 track  
配不到的：先擺著等第二輪

---

### 3 — 用低分框做第二次補洞配對

第二輪只針對「第一輪沒配到的舊 tracks」  
用 low_det 重新跑匈牙利（但會套更低的 IOU 門檻）。

你可以把它想成：  
「我不信任你，但我願意先聽你講幾句。」

低分框會讓舊 track 不至於因 occlusion、motion blur 而直接消失。

這就是 ByteTrack 最關鍵的創新點。

---

### 4 — 更新舊 track / 新增 track

凡是被任何一輪配對到的 track 全部更新（bbox + score）。  
高分更新 = 我很確定  
低分更新 = 先吊命

如果有高分框沒被配對 → 開新 track  
如果 track 長期沒匹配到 → remove（max_age）

---

### 5 — Output: 只輸出經過高分匹配的結果

即使 tracker 用了低分框的資訊來維持 ID，  
ByteTrack 只輸出「高分匹配」的追蹤結果，  
避免 trash-level 偵測造成一堆鬼 ID。

這是 ByteTrack "乾淨輸出" 的原因。

---

## ★ 為什麼 ByteTrack 如此穩？

穩定的主因不是花俏模型，而是簡單邏輯上的突破。

關鍵點是：

1. 多加入一階段配對 = 目標短暫 occlusion 不會失聯
2. 利用低分框延續軌跡，但不把它們當成最終輸出
3. 無需 ReID
4. 本質仍是 SORT，但用更聰明的資料流

## ★ ByteTrack vs SORT vs DeepSORT

SORT：  
極快、極簡，但遇到人群、遮擋就 GG。

DeepSORT：  
加了 ReID，但速度與複雜度都上去。

ByteTrack：  
靠「高低分框雙階段匹配」取代 ReID。  
效果 ≈ DeepSORT，速度 ≈ SORT。

它的效率非常適合嵌入式、Jetson、多人場景。

## ★ ByteTrack 整體資料流（口語動畫版）

1. YOLO 出一堆框（好壞混一起）
2. 先挑出高分框 → 跟 tracks 比，看誰是誰
3. 沒配到的 track 再拿低分框補洞續命
4. 維持 tracks（位置、score、age）
5. 輸出時只給高分的那些

整個方法像一個「現實主義的偵探」：  
證據高分 → 可靠  
證據低分 → 也可能是真的，不要太快丟掉

### ByteTrack 資料流動圖
![[ByteTrack 資料流動圖.png]]