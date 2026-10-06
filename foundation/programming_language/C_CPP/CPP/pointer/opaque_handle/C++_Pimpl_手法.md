---
title: C++ Pimpl 手法
source: 
author: 
published: 
created: 2026-09-21
description: 將類別的私有實作藏進只在 .cpp 定義的 Impl 結構，以縮短編譯時間、隱藏第三方相依並穩定 ABI。
categorization: foundation/programming_language/C_CPP/CPP/class
tags:
  - cpp
  - pimpl
  - design-pattern
  - compilation-firewall
  - abi
  - rule-of-five
  - 不使用標準函式庫
---

# C++ Pimpl 手法（Pointer to Implementation）

## 核心概念

把類別的私有成員與實作細節搬到只在 `.cpp` 定義的 `Impl` 結構，公開標頭檔只留一根指向 `Impl` 的指標。

解決的三個問題:

1. **縮短編譯時間**:修改 `Impl` 不會讓所有 `#include` 該標頭檔的檔案重新編譯，稱為「編譯防火牆」。
2. **隱藏第三方相依**:[[FFmpeg]]、[[CUDA]] 等標頭檔只在 `.cpp` 引入，使用者不需設定 include path。
3. **穩定 [[ABI]]**:不論 `Impl` 增加多少成員，公開類別的 `sizeof` 都只是一根指標，對 `.so` / `.dll` 特別重要。

## 範例程式碼（不使用標準函式庫、不使用 namespace）

### Decoder.h（公開介面）

```cpp
#ifndef DECODER_H
#define DECODER_H

class Decoder {
public:
    Decoder();
    ~Decoder();  // 只宣告，定義放在 .cpp

    // 禁止複製:避免兩個物件指向同一個 Impl 造成 double delete
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // 允許移動:轉移 Impl 的所有權
    Decoder(Decoder&& other);
    Decoder& operator=(Decoder&& other);

    int  Open(const char* path);
    void Close();
    int  GetWidth() const;

private:
    struct Impl;   // 前向宣告，此處為不完整型別
    Impl* impl_;   // 標頭檔只看得到這根指標
};

#endif
```

### Decoder.cpp（實作細節）

```cpp
#include "Decoder.h"
// 重型或第三方標頭檔只放這裡
// #include <libavformat/avformat.h>

// 在 .cpp 內才給出 Impl 的完整定義
struct Decoder::Impl {
    int  width;
    int  height;
    bool opened;
    Impl() : width(0), height(0), opened(false) {}
};

Decoder::Decoder() : impl_(new Impl()) {}

// 解構函式必須在這裡定義，因為只有這裡看得到 Impl 的完整定義
// delete 不完整型別的指標不會呼叫解構函式，屬於未定義行為
Decoder::~Decoder() {
    delete impl_;   // delete nullptr 是安全的
}

// 移動建構:接手對方的指標，再把對方清空
Decoder::Decoder(Decoder&& other) : impl_(other.impl_) {
    other.impl_ = nullptr;
}

// 移動賦值:先釋放自己的資源，再接手對方的
Decoder& Decoder::operator=(Decoder&& other) {
    if (this != &other) {   // 防止自我賦值
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

int Decoder::Open(const char* path) {
    (void)path;
    impl_->opened = true;
    impl_->width  = 1920;
    impl_->height = 1080;
    return 0;
}

void Decoder::Close()          { impl_->opened = false; }
int  Decoder::GetWidth() const { return impl_->width; }
```

### main.cpp（使用端）

```cpp
#include "Decoder.h"

int main() {
    Decoder dec;                                // 只依賴 Decoder.h
    dec.Open("test.mp4");
    int w = dec.GetWidth();

    Decoder dec2(static_cast<Decoder&&>(dec));  // 不用 std::move 的移動寫法
    (void)w;
    return 0;
}
```

## 邏輯說明

| 步驟 | 作用 |
|---|---|
| `struct Impl;` 前向宣告 | 編譯器只需知道型別存在，指標大小固定，不需要完整定義 |
| `Impl* impl_` | 成員只剩一根指標，標頭檔與實作徹底解耦 |
| 建構函式 `new Impl()` | 在堆積（heap）配置實作物件 |
| 解構函式放 `.cpp` | 確保 `delete` 時 `Impl` 已是完整型別 |
| 禁止複製、實作移動 | 遵守 [[Rule_of_Five]]，避免 double delete |

## 注意事項與缺點

1. **解構函式必須定義在 `.cpp`**:若改用 `unique_ptr<Impl>`，解構函式被編譯器在標頭檔隱含產生時，會因 `Impl` 是不完整型別而編譯失敗。
2. **多一次間接存取與堆積配置**:不適合效能極端敏感、大量小物件的熱路徑。
3. **`const` 不會傳遞**:`const` 成員函式中指標本身是 const，但 `Impl` 仍可被修改，編譯器不會擋。
4. **[[Rule_of_Five]] 要自己顧**:解構、複製建構、複製賦值、移動建構、移動賦值缺一都容易出 bug。
5. **不適合 `inline` 與 template**:價值來自實作在 `.cpp`，成員函式不能寫在標頭檔。

## 替代方案比較

| 方式 | 優點 | 缺點 |
|---|---|---|
| **Pimpl** | 保留一般類別用法（可放堆疊、可移動） | 需額外 `new`、樣板程式碼多 |
| **抽象介面 + 工廠函式** | 可多型、可替換實作（方便 mock 測試） | 只能拿指標，有虛擬函式呼叫成本 |
| **Opaque handle**（C 風格） | ABI 最乾淨，C 也能呼叫 | 沒有 [[RAII]]，需手動 `Create` / `Destroy` |

**選擇原則**:要給 C 或其他語言使用、或需要 mock 測試，選抽象介面或 opaque handle；只是想隱藏重型標頭檔、縮短編譯時間，選 Pimpl。

## 英文用語

- **Pimpl**:pointer to implementation 的縮寫，又稱 "Cheshire Cat idiom" 或 "compilation firewall"。
- **incomplete type**:不完整型別。
- **forward declaration**:前向宣告。

## 相關筆記

- [[C++ class 設計]]
- [[FFmpeg]]
- [[Rule_of_Five]]
- [[RAII]]