在處理 Soft NMS 這種分數會動態變動的場景時，使用 **Max-Heap (最大堆積)** 是非常高效的選擇。傳統選擇排序在尋找最大值時需要 $O(N)$，而 Heap 只需要 $O(\log N)$。

以下是不使用 STL (`std::priority_queue`)，純 C++ 手作一個針對 `BBox` 設計的 Max-Heap 實作。

---

### 1. Heap 的核心機制

- **Heapify Up (向上調整)**：當新元素加入或分數增加時，將其與父節點比較並交換，直到滿足父節點大於子節點。
    
- **Heapify Down (向下調整)**：當取出最大值或分數減少時，將根節點與較大的子節點交換，確保堆積特性。
    
- **索引關係**：對於陣列中索引為 $i$ 的節點：
    
    - 父節點索引：$(i - 1) / 2$
        
    - 左子節點索引：$2i + 1$
        
    - 右子節點索引：$2i + 2$
        

---

### 2. C++ 實作：Max-Heap 輔助類別

```cpp
struct BBox {
    float x1, y1, x2, y2;
    float score;
};

// 交換函式
void swap_bbox(BBox& a, BBox& b) {
    BBox temp = a;
    a = b;
    b = temp;
}

// 向下調整，確保 Heap 特性
void heapify_down(BBox* heap, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && heap[left].score > heap[largest].score)
        largest = left;

    if (right < n && heap[right].score > heap[largest].score)
        largest = right;

    if (largest != i) {
        swap_bbox(heap[i], heap[largest]);
        heapify_down(heap, n, largest);
    }
}

// 建立初始堆積 O(N)
void build_heap(BBox* heap, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify_down(heap, n, i);
    }
}
```

### 3. 結合 Soft NMS 的優化思路

在 Soft NMS 中，因為後方候選框的分數會被調降，這會破壞 Heap 的特性。實務上有兩種做法：

Re-build Heap：每次分數變動後重新 build_heap（較耗時，但實作簡單）。

Lazy Update：將分數調降後的框留在 Heap 中，等到它被彈出（Pop）時再檢查其分數是否仍然有效。

以下示範如何在 Soft NMS 邏輯中運用 Heap 概念來選取最大值：

```cpp
void soft_nms_with_heap(BBox* boxes, int count, float sigma, float score_threshold) {
    // 1. 建立初始堆積
    build_heap(boxes, count);
    
    int active_count = count;

    for (int i = 0; i < count; ++i) {
        // 2. 取得當前最大值 (Heap 根節點)
        // 將根節點與目前的「尾端」交換，類似 Heap Sort 的過程
        swap_bbox(boxes[0], boxes[active_count - 1]);
        BBox M = boxes[active_count - 1]; // 這是目前分數最高的框
        active_count--;

        // 3. 重新整理剩餘的 Heap O(log N)
        heapify_down(boxes, active_count, 0);

        // 4. 更新其餘框的分數
        // 注意：這裡直接修改分數會破壞 Heap 特性
        // 在標準 Soft NMS 中，通常會在這裡重新建堆或使用更複雜的 Index Priority Queue
        for (int j = 0; j < active_count; ++j) {
            float iou = calculate_iou(M, boxes[j]);
            boxes[j].score *= expf(-(iou * iou) / sigma);
            
            // 如果分數低於門檻，將其移除（與目前的尾端交換並縮減 active_count）
            if (boxes[j].score < score_threshold) {
                swap_bbox(boxes[j], boxes[active_count - 1]);
                active_count--;
                j--;
            }
        }
        
        // 由於分數被大規模修改，必須重新 build_heap 確保下一輪取到的是真正的最大值
        build_heap(boxes, active_count);
        
        if (active_count <= 0) break;
    }
}
```

### 4. 實作視角
- 複雜度平衡：雖然 Heap 選最大值是 O(logN)，但 Soft NMS 的分數衰減是「全域性」的（每個剩下的框都要改分數），這導致每輪都要 O(N) 的更新。
- 不使用 STL 的好處：在編譯成給嵌入式系統（如 DSP、NPU 驅動）使用的二進位檔時，手寫的 Heap 不會引入龐大的模板庫，且記憶體配置完全可控。
