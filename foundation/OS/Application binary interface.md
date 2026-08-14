---
title: Application binary interface
source: https://claude.ai/chat/fb6dbc10-8bd7-428a-9130-aa70139f927b
author:
published:
created: 2026-06-15
description: test_memry_yolov8n.cpp
tags:
  - clippings
---
## ABI（應用程式二進位介面）

ABI 是程式編譯後，在 **二進位層級** 上定義程式元件之間如何互動的規範。

### 核心概念

ABI 規範了：

- **函式呼叫慣例（Calling Convention）** — 參數如何傳遞（暫存器 or 堆疊）、回傳值放哪裡、誰負責清理堆疊
- **資料型別的記憶體佈局** — 結構體的對齊方式（alignment）、填充（padding）、大小
- **名稱修飾（Name Mangling）** — C++ 為支援函式重載，會將函式名稱編碼，例如 `void foo(int)` → `_Z3fooi`
- **系統呼叫介面** — 如何從 userspace 進入 kernel
- **動態連結規則** — 共享函式庫（.so /.dll）如何在執行時期被載入與解析

---

### ABI vs API 比較

|       | API      | ABI            |
| ----- | -------- | -------------- |
| 層級    | 原始碼層     | 二進位層           |
| 對象    | 開發者      | 編譯器 / 連結器 / OS |
| 例子    | 函式簽名、標頭檔 | 呼叫慣例、記憶體對齊     |
| 相容性破壞 | 改函式名稱或參數 | 改結構體大小、欄位順序    |

---

### 具體範例

**結構體對齊（Struct Padding）**

```c
struct Foo {
    char  a;   // 1 byte
    // 3 bytes padding
    int   b;   // 4 bytes → 必須對齊到 4-byte 邊界
};
// sizeof(Foo) = 8，不是 5
```

**x86-64 System V 呼叫慣例（Linux/macOS）**

```
前 6 個整數參數 → RDI, RSI, RDX, RCX, R8, R9
回傳值          → RAX
浮點參數        → XMM0–XMM7
```

---

### 常見 ABI 破壞情境

在 C++ 共享函式庫開發中，這些改動會 **破壞 ABI 相容性** ，需要重新編譯所有依賴方：

- 在 class 中間插入新的成員變數
- 改變虛擬函式表（vtable）的順序
- 更改函式的預設參數
- 改變 `enum` 的底層型別大小

---

### 實際影響

- **Linux 套件管理** — 函式庫升級若破壞 ABI，所有依賴套件都需重新編譯，這就是為何 soname（如 `libfoo.so.2` ）會隨主版號變動
- **Python C 擴充** — 不同 CPython 版本的 `.pyd` /`.so` 通常無法混用
- **Kernel / glibc** — Linux 刻意維持 stable syscall ABI，確保舊的二進位在新核心上仍可執行

簡單來說，API 是你寫程式時遵守的契約，ABI 則是 **編譯器和 OS 在機器碼層面遵守的契約** 。

---

## ABI 版本不相容問題

### 為什麼會有版本差異？

當函式庫演進時，若改動影響了二進位層級的結構，就會產生新的 ABI 版本。常見原因：

- 結構體新增/移除欄位
- 虛擬函式表順序改變
- 函式簽名改變（參數型別、數量）
- 編譯器版本升級（對齊規則、最佳化方式不同）
- 平台/OS 改變

---

### Linux 的處理機制：soname

```
libfoo.so        → 開發用符號連結
libfoo.so.2      → soname（ABI 主版號）
libfoo.so.2.3.1  → 實際檔案（完整版號）
```

```bash
# 查看函式庫的 soname
readelf -d libfoo.so | grep SONAME
# 輸出：(SONAME) Library soname: [libfoo.so.2]
```

- **主版號改變** → ABI 不相容，需重新編譯依賴方
- **次版號改變** → 通常向後相容（只新增介面）

---

### 版本衝突的實際症狀

```bash
# 執行時找不到正確版本
./app: error while loading shared libraries:
  libfoo.so.1: cannot open shared object file

# C++ name mangling 不符
undefined symbol: _ZN3Foo3barEi

# 結構體大小不一致（最難除錯）
Segmentation fault  ← 讀寫到錯誤的記憶體位置
```

---

### 版本共存機制

#### Linux：同時安裝多版本

```bash
/usr/lib/libfoo.so.1.2.0   # 舊版
/usr/lib/libfoo.so.2.0.0   # 新版

# 不同程式各自連結到需要的版本
app_old → libfoo.so.1
app_new → libfoo.so.2
```

#### Symbol Versioning（符號版本控制）

```c
// 在同一個 .so 內同時提供舊版與新版符號
__asm__(".symver foo_v1, foo@LIB_1.0");
__asm__(".symver foo_v2, foo@@LIB_2.0");  // @@ 表示預設版本
```

```bash
# 查看函式庫的符號版本
nm -D libfoo.so | grep foo
objdump -T libfoo.so
```

---

### C++ 特有問題：Name Mangling 差異

不同編譯器（甚至同一編譯器的不同版本）對同一函式產生不同符號名：

```cpp
void Foo::bar(int x) {}
```

| 編譯器                       | Mangled Name        |
| ------------------------- | ------------------- |
| GCC / Clang (Itanium ABI) | `_ZN3Foo3barEi`     |
| MSVC                      | `?bar@Foo@@QAEXH@Z` |
| GCC 古早版本                  | 又是另一套               |

**解法：用 `extern "C"` 跨 C++ 邊界**

```cpp
// 匯出時關閉 name mangling
extern "C" {
    void foo_bar(Foo* self, int x);
}
```

---

### 偵測與避免 ABI 破壞的工具

```bash
# abidiff：比較兩個版本的 ABI 差異
abidiff libfoo_old.so libfoo_new.so

# abi-compliance-checker：產生詳細相容性報告
abi-compliance-checker -lib foo \
    -old old.xml -new new.xml

# nm / objdump：手動查看符號
nm -D libfoo.so
objdump -T libfoo.so
```

---

### 設計上如何維持 ABI 穩定？

```cpp
// ❌ 直接暴露結構體（改動就破壞 ABI）
struct Config {
    int width;
    int height;
    // 日後新增欄位 → ABI 破壞
};

// ✅ Pimpl 模式（指標隱藏實作細節）
class Config {
public:
    int width();
    int height();
private:
    struct Impl;          // 只有指標，大小永遠是 8 bytes
    Impl* pImpl;
};

// ✅ 用不透明指標（Opaque Pointer）作為 C API
typedef struct Config_ Config;  // 外部看不到內部結構
Config* config_create();
int     config_get_width(Config*);
void    config_destroy(Config*);
```

---

### 總結

```
ABI 版本不同的核心問題：
編譯時期的假設  ≠  執行時期的實際佈局
        ↓
記憶體讀寫位置錯誤 → Crash 或資料損毀
```

最安全的跨版本邊界方式： **用 C ABI（ `extern "C"` + 不透明指標）** 作為公開介面，把 C++ 的複雜性封在函式庫內部。