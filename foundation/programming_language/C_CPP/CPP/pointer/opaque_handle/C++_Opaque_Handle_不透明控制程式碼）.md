---
title: C++ Opaque Handle（不透明控制程式碼）
source:
author:
published:
created: 2026-09-21
description: 用不完整型別指標或整數 ID 隱藏物件內部結構，達成封裝、ABI 穩定與降低編譯相依的介面設計手法。
categorization: foundation/programming_language/C_CPP/CPP/pointer
tags:
  - C++
  - opaque-handle
  - 封裝
  - ABI穩定
  - 介面設計
  - extern-C
  - 不完整型別
---

## 核心概念

**Opaque handle**：使用者只拿到一個代表物件的「代號」（指標或整數），看不到內部結構，只能透過函式介面操作。

設計動機：

1. **封裝**：使用者無法直接修改內部成員，不會破壞不變條件（invariant）。
2. **ABI 穩定**：內部結構改動時，使用者程式不必重新編譯，參見 [[ABI穩定性]]。
3. **縮短編譯時間、隔離相依**：標頭檔不需 include 內部用到的其他標頭。

## 三種實作方式

| 方式 | 型別 | 型別安全 | 備註 |
|---|---|---|---|
| 前置宣告的不完整型別指標 | `struct Foo*` | 有 | **最推薦**，編譯器會擋錯型別 |
| `void*` | `void*` | 無 | 任何指標都能傳入，容易誤用 |
| 整數 ID / index | `int` / `uint32_t` | 弱 | 可加 generation 防舊代號誤用 |

## 方式一：不完整型別指標（最常用）

### 標頭檔 `int_stack.h`（使用者看得到）

```cpp
#ifndef INT_STACK_H
#define INT_STACK_H

#ifdef __cplusplus
extern "C" {          // 使用 C 連結方式，C 與 C++ 都能呼叫
#endif

// 只前置宣告、不定義內容 → 不完整型別
// 使用者只能持有指標，無法 new、sizeof 或存取成員
typedef struct IntStack IntStack;

IntStack* int_stack_create(int capacity);          // 失敗回傳 nullptr
void      int_stack_destroy(IntStack* s);
int       int_stack_push(IntStack* s, int value);  // 成功 0，失敗 -1
int       int_stack_pop(IntStack* s, int* out);    // 成功 0，空的 -1
int       int_stack_size(const IntStack* s);

#ifdef __cplusplus
}
#endif
#endif
```

### 實作檔 `int_stack.cpp`（內部細節只在這裡）

```cpp
#include "int_stack.h"

struct IntStack {       // 真正的定義只出現在 .cpp
    int* data;
    int  size;
    int  capacity;
};

IntStack* int_stack_create(int capacity)
{
    if (capacity <= 0) return nullptr;
    IntStack* s = new IntStack;
    s->data     = new int[capacity];
    s->size     = 0;
    s->capacity = capacity;
    return s;
}

void int_stack_destroy(IntStack* s)
{
    if (!s) return;     // 容許 nullptr，行為與 free(nullptr) 一致
    delete[] s->data;
    delete s;
}

int int_stack_push(IntStack* s, int value)
{
    if (!s || s->size >= s->capacity) return -1;
    s->data[s->size++] = value;
    return 0;
}

int int_stack_pop(IntStack* s, int* out)
{
    if (!s || !out || s->size == 0) return -1;
    *out = s->data[--s->size];
    return 0;
}
```

### 使用端

```cpp
IntStack* s = int_stack_create(4);
int_stack_push(s, 10);
// s->size = 999;   ← 編譯錯誤：不完整型別無法解參考
// IntStack x;      ← 編譯錯誤：無法建立不完整型別的物件
int_stack_destroy(s);   // 誰 create，誰 destroy
```

### 邏輯重點

- `typedef struct IntStack IntStack;` 只告訴編譯器「有這個型別」，使用端只能操作指標。
- **所有權規則**：由庫配置與釋放，使用者必須成對呼叫 `create` / `destroy`。
- **錯誤處理**：用回傳值表示，不丟例外，因為例外不可跨越 [[extern C]] 邊界。

## 方式二：整數 Handle（含 generation 防呆）

高 16 bit 為 generation（世代），低 16 bit 為 index（陣列位置）。

```cpp
typedef unsigned int Handle;
#define HANDLE_INVALID 0u

struct Slot {
    int          value;
    unsigned int gen;    // 此位置目前的世代
    bool         used;
};

static Slot g_slots[16]; // 固定大小的物件池

Handle handle_create(int value)
{
    for (unsigned int i = 0; i < 16; ++i) {
        if (!g_slots[i].used) {
            g_slots[i].used  = true;
            g_slots[i].value = value;
            g_slots[i].gen  += 1;                     // 每次重用，世代 +1
            return (g_slots[i].gen << 16) | (i + 1);  // index+1，讓 0 保留為無效
        }
    }
    return HANDLE_INVALID;
}

static Slot* handle_lookup(Handle h)
{
    if (h == HANDLE_INVALID) return nullptr;
    unsigned int idx = (h & 0xFFFFu) - 1;
    unsigned int gen = h >> 16;
    if (idx >= 16) return nullptr;
    if (!g_slots[idx].used || g_slots[idx].gen != gen) return nullptr;
    return &g_slots[idx];
}

void handle_destroy(Handle h)
{
    Slot* s = handle_lookup(h);
    if (s) s->used = false;
}
```

### 邏輯重點

- 整數本身沒有意義，無法從中推得任何位址。
- 位置被重用後，舊 handle 的 generation 對不上，`handle_lookup` 回傳 `nullptr`，可避免 [[Use-After-Free]]。

## 與 Pimpl 的差別

- **Opaque handle**：C 風格介面，自由函式加不完整型別指標，可跨語言。
- **[[Pimpl]]**：C++ 類別內部持有 `Impl*`，使用者仍用類別語法，僅限 C++。
- 兩者的隔離概念相同，差別在介面形式。

## 常見陷阱

1. **忘記 destroy**：記憶體洩漏，文件需明確標示所有權。
2. **重複 destroy（double free）**：destroy 後把自己的指標設為 `nullptr`。
3. **`void*` 失去型別檢查**：能用不完整型別就不要用 `void*`。
4. **執行緒安全**：handle 只是代號，內部狀態是否安全需另行規範。
5. **例外跨越 C 介面**：在 `extern "C"` 函式內 catch 全部例外，轉成錯誤碼。
6. **`new` 失敗**：預設丟 `std::bad_alloc`；不使用標準函式庫時需另訂策略，例如 `-fno-exceptions` 或自訂配置器。

## 實務例子

- FFmpeg 的 `SwsContext`、`AVDictionary`：標頭只宣告 `struct SwsContext;`，透過 `sws_getContext()` 取得、`sws_freeContext()` 釋放。