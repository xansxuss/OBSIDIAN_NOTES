在 C++ 中進行執行緒（Thread）管理時，若要遵循不使用標準函式庫（STL）的原則，通常需要直接呼叫作業系統底層的 API（如 Windows 的 Win32 API 或 Linux 的 POSIX Threads / pthread）。

以下針對 C++ 執行緒管理的關鍵觀念與底層實作邏輯進行說明：

## 1. 核心管理機制
執行緒管理主要包含四個生命週期階段：建立 (Creation)、同步 (Synchronization)、互斥 (Exclusion) 與 結束 (Termination)。

#### 建立與生命週期
在不使用 std::thread 的情況下，您需要根據平台進行封裝：

- Linux/Unix (POSIX): 使用 `pthread_create()`。

- Windows: 使用 `CreateThread()` 或 `beginthreadex()`。

開發建議：可以自行封裝一個底層的 Thread 類別，利用 RAII (Resource Acquisition Is Initialization) 模式來管理資源，確保執行緒在物件析構時能正確 join 或 detach。

#### 執行緒同步與互斥
為了防止 Race Condition (競態條件)，必須使用同步原語：

- Mutex (互斥鎖)： 確保同一時間只有一個執行緒能存取特定資源。底層對應 `pthread_mutex_t` 或 Windows 的 `CRITICAL_SECTION`。

- Condition Variable (條件變數)： 用於執行緒間的發信（Signaling），讓執行緒在特定條件未滿足時進入休眠，避免 Busy Waiting (忙碌等待)。

## 2. 記憶體管理與快取一致性
在高效能的 C++ 開發中，執行緒管理不只是「開多個執行緒」，更要考慮硬體架構：

- False Sharing (偽共享)： 當兩個執行緒頻繁修改位於同一個 Cache Line 的不同變數時，會導致效能大幅下降。開發時應使用 alignas 關鍵字來確保資料對齊。

- Memory Barriers (記憶體屏障)： 在不使用 std::atomic 的情況下，必須手動處理編譯器與 CPU 的指令重排（Instruction Reordering），確保多執行緒下的資料可見性。

## 3. 執行緒池 (Thread Pool) 的設計
頻繁地建立與銷毀執行緒會產生極大的系統開銷。實務上會預先建立一組執行緒並維護一個任務佇列（Task Queue）：

- Worker Threads: 固定數量的執行緒，持續從佇列中取出任務。

- Task Queue: 儲存待處理的函式指標或任務物件。

- Scheduler: 負責將任務分配給閒置的執行緒。