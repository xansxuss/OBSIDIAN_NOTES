在多執行緒程式設計（Multi-threading）中，`pthread_cond_t` 是 POSIX Threads（pthreads）函式庫提供的 **條件變數（Condition Variable）** 型別。它通常與互斥鎖（`pthread_mutex_t`）搭配使用，用來解決執行緒間的同步問題，特別是當某個執行緒需要等待特定條件成立時。

### 核心概念與機制

條件變數的本質是：**讓執行緒在不滿足條件時進入休眠（釋放 CPU），直到另一個執行緒更新了狀態並發出信號通知。**

---

### 主要操作函式

在 C/C++ 中，主要的操作介面如下：

1. **初始化**
    
    - 靜態：`pthread_cond_t cond = PTHREAD_COND_INITIALIZER;`
        
    - 動態：`pthread_cond_init(&cond, NULL);`
        
2. **等待條件 (Wait)**
    
    - `pthread_cond_wait(&cond, &mutex);`
        
    - **重要：** 此函式在呼叫時會自動釋放 `mutex` 並阻塞；當被喚醒時，會重新取得 `mutex` 後才回傳。
        
3. **發送訊號 (Signal/Broadcast)**
    
    - `pthread_cond_signal(&cond);`：喚醒 **一個** 正在等待該條件的執行緒。
        
    - `pthread_cond_broadcast(&cond);`：喚醒 **所有** 正在等待該條件的執行緒。
        
4. **銷毀**
    
    - `pthread_cond_destroy(&cond);`
        

---

### 標準使用模式 (Code Pattern)

使用條件變數時，為了避免「虛假喚醒」（Spurious Wakeup），必須在 `while` 迴圈中檢查條件。
##### 1. 等待者 (Consumer/Wait)

``` c
pthread_mutex_lock(&mutex);
while (condition == false) {  // 必須用 while 而非 if
    pthread_cond_wait(&cond, &mutex);
}
// 執行臨界區代碼
pthread_mutex_unlock(&mutex);
```

##### 2. 通知者 (Producer/Signal)

``` c
pthread_mutex_lock(&mutex);
condition = true;             // 更新狀態
pthread_cond_signal(&cond);   // 發送訊號
pthread_mutex_unlock(&mutex);
```

<h3 style="color: red;">
為什麼一定要配合 Mutex?
</h3>

1. **競爭條件 (Race Condition)**：如果沒有 Mutex，在執行緒檢查 `condition` 為 false 到準備呼叫 `wait` 之間，若另一個執行緒剛好發出了 `signal`，這個 `signal` 就會遺失，導致等待者永遠睡死。
    
2. **原子性**：`pthread_cond_wait` 內部必須原子地完成「釋放鎖 + 進入休眠」，這需要 Mutex 的保護。
    

### 與 `std::condition_variable` 的差異

既然您偏好不使用標準函式庫（STL），`pthread_cond_t` 是在 Linux/Unix 系統下進行底層開發的最佳選擇。它比 C++11 的 `std::condition_variable` 更接近系統核心，且不會引入額外的實作開銷，非常適合 AI 框架底層或高效能運算（HPC）的場景。

---

### 注意事項

- **死結 (Deadlock)**：確保 `wait` 之前已經正確 lock 了對應的 mutex。
    
- **資源洩漏**：動態初始化的 `pthread_cond_t` 在不再使用時必須呼叫 `destroy`。
    
- **記憶體對齊**：在自定義的並發資料結構中，注意 `pthread_cond_t` 的對齊，避免 Cache False Sharing。