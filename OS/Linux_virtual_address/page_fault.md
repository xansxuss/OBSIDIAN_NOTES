### page fault（頁錯 / 頁面錯誤） 這個概念，這是作業系統記憶體管理中非常核心的一環。

🧠 一句話解釋

Page fault 是當 CPU 嘗試存取某個「目前不在實體記憶體中的虛擬頁」時，觸發的一個例外（exception）。
系統會藉此機會去把那個頁面載入（或報錯），再讓程式繼續執行。

🧩 運作機制流程（簡化示意）

``` bash
程式存取虛擬位址 → MMU 檢查頁表 → 發現頁不存在
       ↓
   觸發 Page Fault 例外
       ↓
OS 介入（Page Fault Handler）
       ↓
1️⃣ 判斷原因（頁在磁碟？權限錯誤？非法存取？）
2️⃣ 若頁在磁碟，從 swap / file 載入到 RAM
3️⃣ 更新頁表（page table）
4️⃣ 恢復程式執行
```

📂 Page Fault 的種類

| 類型                           | 說明                                     | 是否正常                                    |
| ---------------------------- | -------------------------------------- | --------------------------------------- |
| **Minor Page Fault（次要頁錯）**   | 頁不在進程的頁表中，但已在 RAM（例如共享頁被 unmap 後又 map） | ✅ 常見、快速                                 |
| **Major Page Fault（主要頁錯）**   | 頁不在 RAM，需要從磁碟載入                        | ⚠️ 比較慢                                  |
| **Invalid Page Fault（非法頁錯）** | 存取不存在或無權限的頁                            | ❌ Segmentation fault / Access violation |

💾 舉個實際例子（Linux）

假設你有個程式：

``` cpp
int main() {
    int *arr = (int*)malloc(1024 * 1024 * sizeof(int));
    arr[0] = 42;
    return 0;
}
```

這裡：

malloc() 只配置虛擬位址空間，沒有立刻分配實體 RAM。

當你第一次 arr[0] = 42; 時，
CPU 發現該頁未載入，產生 page fault。

OS 分配實體頁 → 更新頁表 → 程式繼續執行。

✅ 所以這是「正常 page fault」，是延遲配置（lazy allocation） 的一部分。

🧮 在 Linux 下觀察 page fault

``` bash
cat /proc/<pid>/stat | awk '{print "minor:", $10, "major:", $12}'
```

或用 time：

``` bash
/usr/bin/time -v ./your_program
```

你會看到：

``` bash
Minor (reclaim) page faults: 123
Major (requiring I/O) page faults: 2
```

與 CUDA / Unified Memory 的關係

如果你最近在玩 CUDA Unified Memory（cudaMallocManaged），那就更 relevant 了。
因為 Unified Memory 就是靠「page fault-driven migration」實現 CPU/GPU 記憶體一致性。

➡️ 當 GPU 嘗試存取一個目前只在 host 端的頁：

會觸發 GPU page fault
driver 會將那個頁 migrate 到 GPU 記憶體
更新 page table
之後該頁屬於 GPU domain
這就是「Lazy Memory Migration」的底層實作之一。

🧨 延伸閱讀關鍵詞

Demand paging
Page table & TLB
Copy-on-write
NUMA page migration
CUDA Unified Memory Fault Handling