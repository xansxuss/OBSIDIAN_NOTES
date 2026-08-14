# SDK 標頭檔閱讀教學

閱讀 SDK 的標頭檔（header file，`.h` / `.hpp`）是使用任何 C/C++ 函式庫最基本、也最重要的能力。官方文件經常過時或寫得不清楚，但標頭檔就是「真相」——它直接告訴你有哪些函式、型別、常數可以用，以及它們的使用規則。以下用一套固定的閱讀步驟，帶你逐層拆解一份標頭檔。

---

## 一、看整體結構之前，先看檔頭三件套

打開一份標頭檔，先別急著看函式，先確認這三個東西：

### 1. Include Guard（防止重複引入）

```c
#ifndef MYSDK_CORE_H
#define MYSDK_CORE_H

// ... 內容 ...

#endif // MYSDK_CORE_H
```

或是更現代的寫法：

```c
#pragma once
```

這只是防止同一個標頭檔被重複展開造成重複定義，跟功能無關，可以直接跳過。

### 2. `extern "C"`

```cpp
#ifdef __cplusplus
extern "C" {
#endif

// C 風格的函式宣告

#ifdef __cplusplus
}
#endif
```

看到這個，代表這個 SDK 的核心是用 C 撰寫、但同時想讓 C++ 程式呼叫。原因是 C++ 編譯器預設會對函式名稱做「名稱修飾（name mangling）」，`extern "C"` 就是告訴編譯器：這些符號請用 C 的規則命名，不要修飾，這樣才能跟編譯好的函式庫（`.lib` / `.a` / `.so` / `.dll`）連結成功。

**這是重要訊號**：只要看到大量 `extern "C"`，就代表這是 C 風格 ABI（Application Binary Interface）的 SDK，介面設計會偏向「不透明指標（opaque pointer）+ 函式」，而不是 C++ 的類別繼承。

### 3. 匯出巨集（export macro）

Windows 上常見這種寫法：

```c
#ifdef MYSDK_EXPORTS
    #define MYSDK_API __declspec(dllexport)
#else
    #define MYSDK_API __declspec(dllimport)
#endif

MYSDK_API int MySdk_Init(void);
```

看到 `MYSDK_API`、`XXX_EXPORT`、`XXX_API` 這類巨集，先不用管它的內容，把它當「透明」的東西過濾掉即可——它只是在告訴編譯器這個符號要不要匯出成 DLL/SO 的公開介面，不影響你怎麼呼叫這個函式。

---

## 二、抓出「型別區」

在看任何函式之前，先把檔案裡的型別定義掃過一遍，心裡建個索引：

```c
typedef struct MySdkContext MySdkContext;   // 不透明結構（opaque struct）
typedef void* MySdkHandle;                  // 控制代碼（handle）
typedef int (*MySdkCallback)(int code, void* userData); // 回呼函式指標
```

三種常見型態：

| 型態 | 特徵 | 意義 |
|---|---|---|
| 不透明結構 | 只有 `typedef struct Foo Foo;`，找不到成員定義 | 你不該去猜它的記憶體配置，只能透過 SDK 提供的函式操作它 |
| Handle（控制代碼） | 通常是 `void*` 或整數型別 | 代表某個由 SDK 內部管理的資源，你只負責拿著它傳來傳去 |
| 回呼函式指標 | `typedef 回傳型別 (*名稱)(參數列);` | SDK 會在某個時機呼叫你註冊的函式，要注意執行緒是誰呼叫的 |

如果是 C++ 風格的 SDK，還會看到：

```cpp
enum class MySdkResult : int32_t {
    Ok = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
};
```

用 `enum class` 而非傳統 `enum`，代表作者在意型別安全，避免跟其他列舉值混用比較。

---

## 三、逐一拆解函式宣告

拿到一個陌生函式宣告時，用固定順序拆解，例如：

```c
MYSDK_API MySdkResult MySdk_CreateContext(
    const MySdkConfig* config,
    MySdkContext** outContext
);
```

拆解步驟：

1. **回傳型別**：`MySdkResult`——通常代表這函式會失敗，回傳值就是錯誤碼，要檢查。
2. **函式名稱**：`MySdk_CreateContext`——多數 C 風格 SDK 會用「模組前綴_動詞」命名，方便一眼看出這是哪個子系統的函式。
3. **參數逐一看**：
   - `const MySdkConfig* config`：`const` 代表這是唯讀的輸入參數，函式不會修改它指向的內容；`*` 代表傳指標而非傳值，通常是為了避免複製大型結構。
   - `MySdkContext** outContext`：雙重指標，代表「輸出參數」——呼叫端傳入一個 `MySdkContext*` 變數的地址，函式內部負責配置好物件、把指標寫回去。看到 `**` 且變數名前綴是 `out`，八九不離十就是這個模式。

**判斷輸入/輸出參數的小技巧**：
- 沒有 `const` 的指標多半是輸出用（或輸入輸出皆可，要看註解）。
- 命名有 `out`、`result`、`ret` 前綴或後綴的，通常是輸出參數。
- 傳「值」（不是指標也不是參考）的參數，一定是輸入。

---

## 四、C++ 風格 SDK 的額外重點

如果 SDK 是給 C++ 用的（沒有 `extern "C"`），要多留意這些：

### 1. 類別介面（Interface Class）

```cpp
class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual bool Initialize(int width, int height) = 0;
    virtual void DrawFrame() = 0;
};
```

- 前綴 `I` 通常代表「介面（interface）」，這種類別只有純虛擬函式（pure virtual function），你不會直接建構它，而是透過 SDK 提供的工廠函式拿到實作。
- **虛擬解構子**一定要注意有沒有寫——沒寫的話，透過基底類別指標刪除衍生類別物件會有未定義行為。

### 2. RAII 包裝類別

```cpp
class MySdkSession {
public:
    MySdkSession();
    ~MySdkSession();
    MySdkSession(const MySdkSession&) = delete;
    MySdkSession& operator=(const MySdkSession&) = delete;
private:
    void* impl_;
};
```

- 看到建構子/解構子成對出現，代表資源會自動管理。
- 看到 `= delete` 的複製建構子/指定運算子，代表這個類別**禁止複製**，只能用移動語意（move）或用指標/參考傳遞——這通常暗示內部握有獨佔資源（例如檔案控制程式碼、GPU 資源）。

### 3. 樣板（template）與 SFINAE 系列

如果看到 `enable_if`、`is_same`、`decltype` 這類東西大量出現，代表這段是編譯期的介面約束，先不要慌，可以先把樣板參數當黑箱看，專注在「這個函式最終接受什麼型別、回傳什麼」，等需要深入客製化時再回頭研究樣板機制。

---

## 五、善用文件註解（Doxygen 風格）

品質好的 SDK 標頭檔會用 Doxygen 註解說明用途：

```c
/**
 * @brief 建立一個新的算繪內容（rendering context）。
 * @param config 初始化設定，呼叫端擁有其所有權。
 * @param[out] outContext 成功時會寫入新建立的內容控制代碼。
 * @return MySdkResult_Ok 表示成功，其餘為錯誤碼。
 * @note 此函式非執行緒安全（thread-safe），需在主執行緒呼叫。
 */
```

閱讀順序建議：**先看 `@note` / `@warning`**（執行緒安全、生命週期、所有權規則這類最容易踩雷的資訊），再看 `@param` / `@return` 補齊細節。

---

## 六、實用小工具

- **IDE 的「跳到定義」／「找出所有參照」**：比自己肉眼搜尋快很多，尤其是巨集展開後的實際型別。
- **`ctags` / `clangd`**：在純文字編輯器（如 Vim）也能做到符號跳轉。
- **Doxygen 產生 HTML 文件**：如果 SDK 附了 Doxyfile，可以直接產生一份帶有類別繼承圖、呼叫關係的網頁文件，比純讀標頭檔輕鬆許多。
- **`grep`／`ripgrep` 找巨集定義**：遇到看不懂的巨集，直接搜尋它的 `#define`，往往就在同一個標頭檔或相鄰的 `*_export.h`、`*_config.h` 檔案裡。

---

## 七、一份標頭檔的完整閱讀順序（總結）

1. 看 include guard、`extern "C"`、匯出巨集——先過濾掉這些「雜訊」。
2. 掃過所有型別定義，建立心智索引（哪些是不透明結構、哪些是列舉、哪些是回呼函式）。
3. 找出模組的「入口函式」（通常是 `XXX_Init` / `XXX_Create` 之類），從這裡開始追生命週期：建立 → 使用 → 釋放。
4. 逐一拆解你需要的函式簽名：回傳型別、參數的 `const`／指標階層／輸入輸出方向。
5. 讀文件註解，特別留意執行緒安全與所有權（ownership）相關的警告。
6. 有疑問時，用 IDE 跳到定義，或搜尋是否有對應的 `.c`／範例程式碼可以參照。

熟練這套流程後，即使遇到完全沒有文件的 SDK，也能單靠標頭檔本身推敲出正確的使用方式。
