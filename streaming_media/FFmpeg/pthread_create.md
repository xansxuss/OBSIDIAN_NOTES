在 C 語言的並行程式設計（Concurrent Programming）中，pthread_create() 是 POSIX Threads 標準中最重要的函式，用於在目前的行程（Process）中建立一個新的執行緒（Thread）。

由於您偏好 C/C++ 且儘量不使用標準函式庫，以下針對 pthread_create() 的機制與實作細節進行說明。

### 1. 函式原型
``` C
#include <pthread.h>

int pthread_create(pthread_t *thread, 
                   const pthread_attr_t *attr,
                   void *(*start_routine) (void *), 
                   void *arg);
```

### 2. 參數解析

- **`thread` (輸出參數)**：指向 `pthread_t` 型別的指標。函式成功執行後，會將新執行緒的識別碼（ID）存入此位址。
    
- **`attr`**：指向執行緒屬性結構的指標。若傳入 `NULL`，則使用系統預設屬性（如：Joinable 狀態、預設 Stack 大小）。
    
- **`start_routine`**：這是**執行緒進入點**（Entry Point）。它是一個函式指標，接收一個 `void *` 參數並回傳 `void *`。
    
- **`arg`**：傳遞給 `start_routine` 的單一引數。若需要傳遞多個數值，通常會封裝進一個 `struct` 並傳遞其位址。
    

### 3. 記憶體與架構視角

當 `pthread_create()` 被呼叫時，作業系統的核心（Kernel）會分配獨立的 **Stack** 給該執行緒，但它會與主執行緒共享 **Heap**、**全域變數**、**檔案描述符（File Descriptors）** 以及 **代碼段（Code Segment）**。

---

### 4. 程式碼範例 (C 語言)

這裡展示一個簡單的建立流程：

```C
#include <pthread.h>
#include <unistd.h>

// 執行緒執行的函式
void* thread_job(void* arg) {
    long val = (long)arg;
    // 執行任務...
    return (void*)(val * 2);
}

int main() {
    pthread_t tid;
    long input = 42;

    // 建立執行緒
    if (pthread_create(&tid, NULL, thread_job, (void*)input) != 0) {
        return 1; // 建立失敗
    }

    void* retval;
    // 等待執行緒結束並取得回傳值
    pthread_join(tid, &retval);

    return 0;
}
```

### 5. 工程師筆記：C++ 與指標安全

在 C++ 環境下使用 `pthread_create()` 時，有幾點需要特別注意：

1. **成員函式問題**：`start_routine` 必須是靜態成員函式（`static`）或全域函式。普通的成員函式隱含 `this` 指標，簽章與 `void* (*)(void*)` 不相符。
    
2. **型別轉換**：在傳遞 `arg` 時，通常需要透過 `static_cast<void*>` 轉換。在讀取回傳值時，則需確保記憶體生命週期正確（避免回傳 Thread Local 的 Stack 變數位址）。
    
3. **錯誤處理**：與許多 C 函式不同，`pthread` 系列函式**不會**設置 `errno`，而是直接回傳錯誤代碼（成功則回傳 0）。