`std::atomic<bool>` 是 C++ 標準庫中提供的一種「原子型別（atomic type）」專門用來表示 **布林值（`bool`）** 的原子操作版本。它位於標頭檔 `<atomic>` 中。

---

### 🔹用途說明

在多執行緒（multithreading）環境下，若多個執行緒同時讀寫同一個變數，會造成**資料競爭（data race）**。  
使用 `std::atomic<bool>` 可以保證對這個布林變數的**讀取與寫入操作都是原子的（atomic operation）**，也就是說這些操作不會被中斷或交錯。

---

### 🔹基本語法

``` cpp
#include <atomic> 
#include <thread> 
#include <iostream>  
td::atomic<bool> flag(false);  void worker() {     // 等待直到 flag 為 true     
while (!flag.load(std::memory_order_acquire)) {         // busy wait     
}     std::cout << "Worker thread started!\n"; 
}  int main() {     
    std::thread t(worker);      // 模擬一些工作     
    std::this_thread::sleep_for(std::chrono::seconds(1));      // 將 flag 設為 true，喚醒 worker     
    flag.store(true, std::memory_order_release);      
    t.join(); 
    }
```

---

### 🔹常用成員函式

| 函式 | 說明 |
|---- | ---- |
| `store(bool desired, memory_order order = memory_order_seq_cst)` | 將布林值寫入原子變數。 |
| `load(memory_order order = memory_order_seq_cst)` | 讀取當前值。 |
| `exchange(bool desired, memory_order order = memory_order_seq_cst)` | 將變數設為 `desired`，並回傳舊值。 |
| `compare_exchange_weak(expected, desired)` / `compare_exchange_strong(expected, desired)` | 比較並交換，用於實作無鎖（lock-free）演算法。 |

---

### 🔹特點

- 原子性：保證操作不可被中斷。
- 執行緒安全：多執行緒可安全地同時操作。
- 效能：通常比使用 mutex（互斥鎖）更快。
- 限制：僅適合非常簡單的共享狀態（如 flag），不適用於複雜資料結構。

---

### 🔹簡單例子

``` cpp
std::atomic<bool> ready = false;  // Thread A 
ready.store(true);  // Thread B 
if (ready.load()) {     
    // do something 
    }
```
