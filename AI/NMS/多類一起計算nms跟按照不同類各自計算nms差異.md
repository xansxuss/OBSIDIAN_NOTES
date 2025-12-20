1. 兩種 NMS 方法的定義與差異
| 方法                     | 說明                        |
| ---------------------- | ------------------------- |
| **Class-wise NMS**     | 每個類別分開做一次 NMS（最常見）        |
| **Class-agnostic NMS** | 所有類別的框混在一起做一次 NMS，不管是狗還是貓 |

核心差異：是否允許不同類別的 box 相互抑制（suppression）
Class-wise NMS：

- 貓 vs 狗 的框互不干涉。
- 適合用在多類物件明顯不同的場景（如行人、車輛、交通號誌）。

Class-agnostic NMS：

- 不同類別的 box 可能互相壓制，例如「車 vs 卡車」可能只保留一個。
- 常用於只關心最顯著目標的任務（如 top-1 物件偵測、或單類輸出時）。
  
2. 實際例子對比
➤ Class-wise NMS：
dog_1: score=0.95
cat_1: score=0.93 (IoU with dog_1 = 0.7)
→ 結果：兩者都保留，因為不同類。

➤ Class-agnostic NMS：

dog_1: score=0.95
cat_1: score=0.93 (IoU = 0.7)
→ 結果：只保留 dog_1，因為 NMS 壓掉重疊的 cat_1。

3. 哪些情況下適合哪一種？

| 應用場景                    | 建議方法                       | 原因           |
| ----------------------- | -------------------------- | ------------ |
| 多類別偵測（COCO, Pascal VOC） | Class-wise NMS             | 可保留相同區域中不同物件 |
| 單類別模型（如人臉偵測）            | Class-agnostic NMS         | 不需要分類別壓制     |
| 模型中類別高度相似（如細分類）         | Class-agnostic + 類別加權後 NMS | 避免過多重複輸出     |
| 僅關心 top-k 物件（如攝影構圖）     | Class-agnostic NMS         | 關注最強信心框      |

4. 實務坑點與注意事項
⚠️ 多類別 + 多 label 問題（multi-label NMS）
若一個框有多個類別標籤（非 softmax，而是 sigmoid），則 class-wise NMS 會讓同一個 box 重複出現多次，需額外處理（如保留最大分數那個類別）。

⚠️ Class-agnostic NMS 當類別數多時會損失結果多樣性
可能一堆高分框都集中在某一類，其他類別的就被壓掉了。

延伸：Hybrid NMS
可以做類別加權（weighted IoU 或 weighted score）來達成更細膩的控制，例如：

pseudo
複製
編輯
adj_iou = IoU * similarity(c1, c2)  // 類似度高的類別互壓更嚴格
這對類別多、且有樹狀結構（如 COCO-stuff）時很有幫助。

🔧 實作建議（PyTorch 為例）
Torchvision 的 batched_nms 實際就是 Class-wise NMS：
python
複製
編輯
from torchvision.ops import batched_nms
keep = batched_nms(boxes, scores, labels, iou_threshold)
若要做 Class-agnostic NMS，只要全部 label 改為同一類別即可。

總結
| 項目            | Class-wise NMS | Class-agnostic NMS |
| ------------- | -------------- | ------------------ |
| 不同類別互壓？       | ❌ 否            | ✅ 是                |
| 可保留同區不同物件？    | ✅              | ❌                  |
| 多類場景表現        | ✅              | ⚠️ 可能不穩            |
| 單類偵測、top-k 場景 | ⚠️ 重複          | ✅ 簡單               |
| 易實作度          | ✅              | ✅（更簡單）             |
