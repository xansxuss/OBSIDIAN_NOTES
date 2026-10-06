---
title: "pthread_self()"
source: "https://gemini.google.com/app/ace98e99153108de"
author:
published:
created: 2026-08-21
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
`pthread_self()` 是 POSIX 執行緒（POSIX Threads, 簡稱 pthread）API 中的一個函式，主要功能是 **取得呼叫該函式的執行緒（Thread）自身的 ID** 。

**語法與標頭檔**

需要引入標頭檔 `<pthread.h>` ：

``` C++
#include <pthread.h>

pthread_t pthread_self(void);
```

- **傳回值** ：傳回呼叫此函式的執行緒識別碼，型態為 `pthread_t` 。
- **參數** ：無。

**核心邏輯與功用說明**

1. **識別執行緒身分** ： 在多執行緒程式中，主執行緒（Main thread）或建立出來的子執行緒經常需要知道「我是誰」。例如，在日誌（Logging）系統中記錄是哪一個執行緒在執行任務，或是執行緒要存取 Thread-Local Data 時，可以透過 `pthread_self()` 取得自己的 ID。
2. **與其他執行緒 ID 比較** ： POSIX 標準中， `pthread_t` 型態不一定是整數（在某些平台上可能是結構或指標），因此 **不能直接用 `==` 運算子比較兩個 `pthread_t`** 。如果需要比較兩個執行緒是否相同，必須搭配 `pthread_equal()` 函式：
	``` C++
	pthread_t self_id = pthread_self();
	if (pthread_equal(self_id, target_thread_id)) {
	    // 兩者為同一執行緒
	}
	```
3. **執行緒資源回收與分離** ： 若執行緒希望自己在結束時自動釋放資源，而不需由其他執行緒呼叫 `pthread_join()` 來回收，可以透過 `pthread_self()` 結合 `pthread_detach()` 實現「Self-detaching」：
	``` C++
	pthread_detach(pthread_self());
	```

**範例程式碼**

以下展示如何呼叫 `pthread_self()` 取得自身的執行緒 ID 並分離該執行緒（ **完全採用 C 語言標準語法與 POSIX API，不使用 C++ 標準函式庫** ）：

``` C++
#include <stdio.h>
#include <pthread.h>

void* thread_routine(void* arg) {
    // 邏輯 1：取得自身的執行緒 ID
    pthread_t my_id = pthread_self();
    
    // 邏輯 2：將自身設為 detached 狀態，結束時自動回收資源，不需主執行緒 join
    pthread_detach(my_id);

    // 邏輯 3：印出訊息（轉成 unsigned long 僅供 debug 展示）
    printf("[Child Thread] My thread ID is: %lu\n", (unsigned long)my_id);

    return NULL;
}

int main() {
    pthread_t thread;

    // 建立子執行緒
    if (pthread_create(&thread, NULL, thread_routine, NULL) != 0) {
        perror("Failed to create thread");
        return 1;
    }

    pthread_t main_id = pthread_self();
    printf("[Main Thread] My thread ID is: %lu\n", (unsigned long)main_id);

    // 給子執行緒一點時間執行
    for (volatile int i = 0; i < 100000000; ++i);

    return 0;
}
```
