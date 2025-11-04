`std::atomic<bool>` 是 C++11 起引入的原子類型之一，用來在**多執行緒環境中安全地讀寫布林值**。  
它的特點是「操作不可分割（atomic）」──保證在任意時刻只有一個執行緒能改動這個值，避免 race condition。

---

### 🧩 一、基本概念

```cpp
#include <atomic> 
#include <iostream> 
#include <thread>  
std::atomic<bool> ready(false);  
void worker(int id) {     
while (!ready.load(std::memory_order_acquire)) {         // 忙等，直到主執行緒設定 ready = true     
}     
std::cout << "Thread " << id << " 開始工作\n"; }  
int main() {     
std::thread t1(worker, 1);     
std::thread t2(worker, 2);      
std::this_thread::sleep_for(std::chrono::seconds(1));     
ready.store(true, std::memory_order_release);      
t1.join();     
t2.join(); 
}
```

🔍 **解釋：**

- `std::atomic<bool> ready(false);`  
    宣告一個原子布林值，初始化為 `false`。
- `ready.store(true)`  
    設定值為 `true`（寫入操作是原子的）。
- `ready.load()`  
    讀取值（讀取操作也是原子的）。

---

### ⚙️ 二、主要操作函式

| 函式                                   | 功能                        | 範例                                |
| ------------------------------------ | ------------------------- | --------------------------------- |
| `.store(value)`                      | 寫入                        | `flag.store(true);`               |
| `.load()`                            | 讀取                        | `if (flag.load()) {}`             |
| `.exchange(value)`                   | 原子交換（回傳舊值）                | `bool old = flag.exchange(true);` |
| `.compare_exchange_strong(exp, val)` | CAS 強版本（compare-and-swap） |                                   |
| `.compare_exchange_weak(exp, val)`   | CAS 弱版本（可能失敗要重試）          |                                   |

---

### 🧠 三、記憶體順序（memory order）

預設的 `memory_order_seq_cst` 是最安全但也最慢的。  
常見模式：

| 模式                     | 意義                     |
| ---------------------- | ---------------------- |
| `memory_order_relaxed` | 不保證順序，只確保原子性           |
| `memory_order_acquire` | 確保之後的操作不會重排到前面         |
| `memory_order_release` | 確保之前的操作不會重排到後面         |
| `memory_order_acq_rel` | 同時具備 acquire + release |
| `memory_order_seq_cst` | 全域順序一致（最嚴格）            |

📌 常見用法：

``` cpp
// 寫入方（release） 
flag.store(true, std::memory_order_release);  
// 讀取方（acquire） 
while (!flag.load(std::memory_order_acquire));
```


---

### 🚀 四、典型應用場景

1. **多執行緒的啟動信號（start flag）**
    
    ``` cpp
    std::atomic<bool> ready(false);
    ```
    主線設定，子線等待。
    
2. **停止旗標（stop flag）**
    
    ``` cpp
    std::atomic<bool> stop(false); while (!stop.load()) { do_something(); }
    ```
    
1. **無鎖（lock-free）結構中的狀態位**
    
    - 避免 mutex 的開銷。
    - 適合輕量旗標、state 管理。

---

### 🧮 五、注意事項

- ✅ 原子操作 ≠ 無需同步一切。它僅確保「這個變數」是安全的，不代表整個邏輯安全。
- ⚠️ 不能直接用於條件變數 wait（需要轉換或包裝）。
- 🚫 不可複製與賦值（copy/assign operator 被 delete）。
    

---

### 🧩 範例：compare_exchange

``` cpp
std::atomic<bool> flag(false); 
bool expected = false;  // 只有當 flag == false 時才設成 true 
if (flag.compare_exchange_strong(expected, true)) {     
std::cout << "成功設為 true\n"; 
} 
else {     
std::cout << "flag 原本不是 false（是 true）\n"; 
}
```