---
title: "__atomic_store_n"
source: "https://gemini.google.com/app/b24b053a3f95757d"
author:
published:
created: 2026-06-26
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---

在 C/C++ 核心處理多執行緒（Multi-threading）與並行（Concurrency）程式設計時， `__atomic_store_n` 是一個非常底層且重要的 **GCC 內建函式 (Built-in Function)** 。它被用來執行 **原子性的寫入操作** （Atomic Store），確保某個變數在被寫入時，不會被其他執行緒看到「寫到一半」的半殘狀態。

不使用 C++ 標準函式庫（如 `<atomic>` ），直接使用 GCC 內建的 `__atomic` 系列函式是非常道地且有效控制硬體行為的做法。

## 語法與參數

```
void __atomic_store_n (type *ptr, type val, int memorder)
```

- **`ptr`** ：指向目標記憶體位置的指標（該記憶體必須是符合對齊要求的基礎型態，如 `int`, `long`, 指標等）。
- **`val`** ：準備寫入該記憶體位置的值。
- **`memorder`** ： **記憶體順序（Memory Order）** 。這是最關鍵的參數，用來告訴編譯器與 CPU 可以進行何種程度的指令重排（Instruction Reordering）。

## 支援的記憶體順序 (Memory Order)

對寫入操作（Store）而言，並非所有的記憶體順序都適用。你只能使用以下種類：

| 記憶體順序 | 說明 |
| --- | --- |
| **`__ATOMIC_RELAXED`** | **最寬鬆的順序** 。僅保證此操作本身是原子的，但不保證周圍其他程式碼的執行順序。編譯器與 CPU 可以自由重排指令。 |
| **`__ATOMIC_RELEASE`** | **釋放語義（常用）** 。確保在此操作 *之前* 的所有讀寫指令，絕不會被重排到此操作之後。常用於同步訊號（如釋放鎖、通知資料已準備好）。 |
| **`__ATOMIC_SEQ_CST`** | **順序一致性（預設/最嚴格）** 。保證所有執行緒看到的所有 `SEQ_CST` 操作都有一個全域一致的順序。效能開銷最大。 |

> ⚠️ **注意** ： `__atomic_store_n` **不能** 使用 `__ATOMIC_CONSUME`, `__ATOMIC_ACQUIRE`, 或 `__ATOMIC_ACQ_REL` ，因為這些是針對「讀取（Load）」或「讀寫（Read-Modify-Write）」操作設計的。

## 實戰範例

以下是一個不依賴 C++ 標準函式庫，單純使用 C 語法與 GCC 內建函式實現的「生產者-消費者」簡單旗標（Flag）同步範例：

``` C
// 假設這是在某個標頭檔或全域範圍
int shared_data = 0;
int ready_flag = 0; // 當作同步用的旗標

// 執行緒 A：生產者
void producer() {
    shared_data = 42; // 準備資料
    
    // 使用 RELEASE 確保 shared_data 的寫入
    // 絕對不會被重排到 ready_flag 之後！
    __atomic_store_n(&ready_flag, 1, __ATOMIC_RELEASE);
}

// 執行緒 B：消費者
void consumer() {
    // 這裡使用 __atomic_load_n（搭配 ACQUIRE）來配對
    while (__atomic_load_n(&ready_flag, __ATOMIC_ACQUIRE) == 0) {
        // 等待旗標變為 1
    }
    
    // 此時 ready_flag 為 1，且因為 ACQUIRE/RELEASE 的配對，
    // 保證能正確讀取到 shared_data 的最新值 42
    int my_data = shared_data; 
}
```

## 為什麼不用 \_\_sync\_lock\_test\_and\_set？

在舊的 code base 看過 `__sync` 系列的內建函式（例如 `__sync_synchronize` ）。

- `__sync` 系列是舊版的 GCC 擴充（Legacy BLTINS），它強制使用最嚴格的 Full Barrier（等同於 `__ATOMIC_SEQ_CST` ），無法微調效能。
- `__atomic` 系列是為了支援 C11 / C++11 的記憶體模型而開發的新標準， **強烈建議全面使用 `__atomic` 系列** 來取代舊的 `__sync` 系列，以獲得更好的效能優化空間。
