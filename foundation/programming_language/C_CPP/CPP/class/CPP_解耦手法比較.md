---
title: C++ 隱藏實作與降低耦合手法比較
source: 
author: 
published: 
created: 2026-09-21
description: 比較 Forward Declaration、Pimpl、抽象介面加工廠函式、Opaque Handle 四種 C++ 隔離實作手法的原理、優缺點與選用時機。
categorization: foundation/programming_language/C_CPP/CPP/class
tags:
  - CPP
  - Pimpl
  - Forward-Declaration
  - 抽象介面
  - 工廠模式
  - Opaque-Handle
  - ABI
  - 編譯防火牆
  - 設計模式
---

# C++ 隱藏實作與降低耦合手法比較

## 定位

- **Forward Declaration 是語法工具**，其餘三者是完整的「隔離實作」設計手法。
- 後三者內部通常也會用到前置宣告（例如 `struct Impl;`、`typedef struct Handle Handle;`）。
- 相關概念：Incomplete Type、編譯防火牆、ABI 穩定性、RAII

## 1. Forward Declaration（前置宣告）

**做法**：只宣告型別名稱，不引入完整定義。

```cpp
// decoder.h
class Frame;                      // 前置宣告：告訴編譯器「有這個型別」，但不知道內容

class Decoder {
public:
    bool Decode(const Frame* in, Frame* out); // OK：指標/參照不需要知道型別大小
private:
    Frame* last_;                 // OK：指標大小固定
    // Frame member_;             // 錯誤：incomplete type，不知道要配置多少空間
};
```

- **原理**：處理指標或參照時，編譯器只需知道大小等於指標大小，不需要型別內部，因此可省掉 `#include "frame.h"`。
- **限制**：只能用於指標與參照；不能以值宣告成員、不能呼叫成員函式、不能繼承。
- **注意**：不會隱藏自己 class 的 private 成員，別人仍看得到 `last_`。

## 2. Pimpl（Pointer to Implementation）

**做法**：public class 只留一個指向實作結構的指標，實作結構的定義放在 `.cpp`。

```cpp
// widget.h（使用者只看到這個，完全不含第三方標頭）
class Widget {
public:
    Widget();
    ~Widget();                              // 必須在 .cpp 實作
    Widget(const Widget&) = delete;         // 禁止拷貝，避免兩個物件共用同一個 impl_ 而 double free
    Widget& operator=(const Widget&) = delete;

    int Read(unsigned char* buf, int size);
private:
    struct Impl;                            // 前置宣告巢狀結構
    Impl* impl_;                            // 唯一的資料成員
};

// widget.cpp（實作細節全部藏在這裡）
#include "widget.h"
#include "third_party.h"                    // 第三方標頭只出現在 .cpp

struct Widget::Impl {                       // 在這裡才給出完整定義
    ThirdPartyCtx* ctx;
    int            counter;
};

Widget::Widget() : impl_(new Impl()) {
    impl_->ctx = 0;
    impl_->counter = 0;
}

Widget::~Widget() { delete impl_; }         // 此處 Impl 已是完整型別，delete 才合法

int Widget::Read(unsigned char* buf, int size) {
    impl_->counter++;
    // ... 呼叫 ThirdPartyCtx 的 API
    return size;
}
```

- **效果**：`sizeof(Widget)` 永遠等於一個指標；修改 `Impl` 欄位或更換第三方函式庫，使用者程式碼不需重新編譯（編譯防火牆）。
- **陷阱**：解構函式一定要放 `.cpp`，否則在標頭 `delete impl_` 會遇到 incomplete type。
- **陷阱**：必須禁止拷貝（或自行實作深拷貝），否則 double free。

## 3. 抽象介面 + 工廠函式

**做法**：標頭只公開純虛擬函式介面與一個建立函式，實作類別完全不出現在標頭。

```cpp
// idecoder.h
class IDecoder {
public:
    virtual ~IDecoder() {}
    virtual int  Decode(const unsigned char* data, int size) = 0;  // 純虛擬函式
    virtual void Destroy() = 0;   // 由建立它的模組負責釋放，避免跨 DLL 的 new/delete 不一致
};

enum DecoderType { DEC_SOFTWARE, DEC_HARDWARE };
IDecoder* CreateDecoder(DecoderType type);   // 工廠函式：依參數決定回傳哪個實作

// decoder_impl.cpp
#include "idecoder.h"

class SwDecoder : public IDecoder {
public:
    int  Decode(const unsigned char* data, int size) { return size; }
    void Destroy() { delete this; }
};

class HwDecoder : public IDecoder {
public:
    int  Decode(const unsigned char* data, int size) { return size; }
    void Destroy() { delete this; }
};

IDecoder* CreateDecoder(DecoderType type) {
    if (type == DEC_HARDWARE) return new HwDecoder();
    return new SwDecoder();
}
```

- **效果**：呼叫端只持有 `IDecoder*`，透過 vtable 動態分派。
- **最大價值**：執行期可切換多種實作（軟體／硬體解碼），且方便寫 mock 做單元測試。
- **代價**：物件只能在 heap 建立，每次呼叫多一次虛擬函式間接跳轉。

## 4. Opaque Handle（不透明控制代碼）

**做法**：C 風格 API，型別只宣告不定義，使用者只拿得到指標。

```cpp
// decoder_api.h
#ifdef __cplusplus
extern "C" {                                 // 關閉 name mangling，讓 C / Python ctypes 都能連結
#endif

typedef struct DecoderHandle DecoderHandle;  // 只宣告、永不在標頭定義

DecoderHandle* decoder_create(int codec);
int            decoder_decode(DecoderHandle* h, const unsigned char* data, int size);
void           decoder_destroy(DecoderHandle* h);

#ifdef __cplusplus
}
#endif

// decoder_api.cpp
#include "decoder_api.h"

struct DecoderHandle {                       // 內部可以是任何 C++ 東西
    int codec;
    int frames;
};

DecoderHandle* decoder_create(int codec) {
    DecoderHandle* h = new DecoderHandle();
    h->codec  = codec;
    h->frames = 0;
    return h;
}

int decoder_decode(DecoderHandle* h, const unsigned char* data, int size) {
    if (!h) return -1;                       // C API 沒有例外，錯誤用回傳碼表達
    h->frames++;
    return size;
}

void decoder_destroy(DecoderHandle* h) { delete h; }
```

- **效果**：介面只用 C 基本型別與不透明指標，ABI 最穩定，是跨語言（Python ctypes、Rust、C）的標準做法。
- **代價**：無 RAII、無型別安全，需手動 `destroy`，錯誤只能靠回傳碼。

## 綜合比較

| 面向 | Forward Declaration | Pimpl | 抽象介面 + 工廠 | Opaque Handle |
|---|---|---|---|---|
| 本質 | 語法工具 | 設計模式 | 設計模式 | C 風格 API 慣例 |
| 隱藏 private 成員 | 否 | 是 | 是 | 是 |
| 減少編譯依賴 | 部分 | 是 | 是 | 是 |
| 執行期多型／換實作 | 否 | 否 | 是 | 否（需在內部自己做） |
| 執行期成本 | 無 | 一次額外 heap 配置與指標間接存取 | 虛擬函式呼叫（vtable） | 指標間接存取，成本小 |
| ABI 穩定性 | 無幫助 | 良好（sizeof 固定） | 尚可，vtable 佈局依編譯器而異 | 最佳（純 C ABI） |
| 可在堆疊上建立物件 | 是 | 是 | 否 | 否 |
| 好寫 mock／測試 | 否 | 較難 | 容易 | 較難（需換連結） |
| 跨語言呼叫（C／Python） | 否 | 否 | 困難 | 容易 |
| RAII／型別安全 | 完整 | 完整 | 完整（需 `Destroy` 或 virtual dtor） | 無，需手動 `destroy` |
| 實作複雜度 | 極低 | 低 | 中 | 中（需包一層 C 封裝） |

## 選用建議

- **只想少 include、加快編譯** → Forward Declaration，成本最低。
- **隱藏第三方函式庫（如 FFmpeg、CUDA）、保持二進位相容，且只有單一實作** → Pimpl。
- **需要多個可替換的後端，或要做單元測試** → 抽象介面 + 工廠。
- **給 C 程式、Python（`ctypes`）呼叫，或做成穩定的 `.so` / `.dll`** → Opaque Handle。

**常見組合**：對外用 Opaque Handle 的 C API，內部用抽象介面管理多種後端，各後端類別再用 Pimpl 包住第三方函式庫。

## 備註

- 實務上 Pimpl 與工廠回傳的指標常改用 `std::unique_ptr` 自動釋放；本筆記為配合不使用標準函式庫的原則，改用原生 `new` / `delete`，需特別留意解構與 `Destroy` 的責任歸屬。
