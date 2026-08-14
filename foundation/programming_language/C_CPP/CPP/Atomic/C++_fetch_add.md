---
title: "C++_fetch_add"
source: "https://gemini.google.com/app/44e9885cbb14cdf5?hl=zh-TW"
author:
published:
created: 2026-08-14
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**You**

C++ fetch\_add

---

**Gemini**

在 C++ 中， `std::atomic::fetch_add` 是用來對原子變數（atomic variable）進行 **原子加法運算** 的成員函數。它能夠確保在多執行緒（multithreading）環境下，讀取變數值、加上指定的數值，並寫回記憶體的整個過程是不可分割的（atomic），不會受到其他執行緒的干擾，從而避免資料競爭（data race）。

## 一、 函數語法與參數

`fetch_add` 是 `std::atomic<T>` 類別模板的方法，其常見的宣告形式如下（以整數型別為例）：

```
T fetch_add( T arg, std::memory_order order = std::memory_order_seq_cst ) noexcept;
```

### 參數說明

1. **`arg`** ：要加到原子變數上的數值（型別通常為 `T` 或對應的整數偏移量）。
2. **`order`** ：記憶體順序（Memory Order），用來控制執行緒之間的記憶體同步與指令重排序。預設為 `std::memory_order_seq_cst` （循序一致性）。

### 回傳值

- 回傳該原子變數 **在相加之前的原本數值** （Old Value）。這點非常重要，它與直接使用 `+=` 運算子不同（ `+=` 回傳的是相加後的最新值）。

## 二、 運作邏輯與功用

1. **原子性（Atomicity）：** 整個「讀取 $\rightarrow$ 相加 $\rightarrow$ 寫入」的動作在硬體層面（透過 CPU 的特殊指令，如 x86 的 `LOCK XADD` ）被保證為單一不可中斷的操作。
2. **返回值特性：** 如果你需要知道操作前的值來做後續判斷（例如實作無鎖資料結構、計數器或票號分配）， `fetch_add` 非常好用。
3. **與 `operator+=` 的差異：**
	- `atomic_var.fetch_add(1);` $\rightarrow$ 回傳 **加之前** 的值。
		- `atomic_var += 1;`（或 `++atomic_var` ） $\rightarrow$ 回傳 **加之後** 的值。

## 三、 程式碼範例與詳細說明

以下範例示範如何在多執行緒環境中使用 `std::atomic::fetch_add` 來實作安全執行緒的計數器：

```
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>

// 宣告一個全域的原子計數器，初始值為 0
std::atomic<int> global_counter(0);

void worker_task(int id) {
    // 每個執行緒將計數器加 3
    // fetch_add 會回傳加法「之前」的舊值
    int old_value = global_counter.fetch_add(3, std::memory_order_relaxed);
    
    std::cout << "Thread " << id 
              << ": 舊值 = " << old_value 
              << ", 當前最新值 = " << global_counter.load(std::memory_order_relaxed) 
              << "\n";
}

int main() {
    const int num_threads = 4;
    std::vector<std::thread> threads;

    // 建立多個執行緒同時對同一個原子變數進行 fetch_add
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_task, i);
    }

    // 等待所有執行緒結束
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "最終計數器結果: " << global_counter.load() << "\n";
    return 0;
}
```

### 邏輯與功用解析：

1. **`std::atomic<int> global_counter(0);`**：建立一個安全的原子整數，不需使用傳統的 `std::mutex` 就能安全地跨執行緒修改。
2. **`global_counter.fetch_add(3, std::memory_order_relaxed);`**：
	- 執行緒安全地將 `global_counter` 的值加上 `3` 。
		- 由於多個執行緒同時執行，硬體會排定先後順序，絕對不會發生遺失更新（lost update）的問題。
		- 變數 `old_value` 記錄了該執行緒下手修改那一瞬間之前的數值。
3. **`std::memory_order_relaxed`** ：在不需嚴格同步其他記憶體變數的簡單計數器場景下，使用放寬的記憶體順序可以獲得最佳的執行效能。