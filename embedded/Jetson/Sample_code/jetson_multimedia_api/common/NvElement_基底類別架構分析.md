---
title: NvElement 基底類別架構分析
source:
author:
published:
created: 2026-08-31
description: 拆解 jetson_multimedia_api 中 NvElement 基底類別的繼承與組合設計
categorization: edge_AI/jetson_multimedia_api
tags:
  - jetson
  - v4l2
  - cpp
  - inheritance
  - video-decode
---

## 背景

拆解 [[jetson-video-decode]] 範例程式時，先從 `jetson_multimedia_api` 的共通基底類別 `NvElement` 開始研究。此類別是所有 V4L2 與非 V4L2 元件的共同根基底類別，提供錯誤狀態追蹤與 profiling（效能量測）介面。

## NvElement.h 重點拆解

### Include 項目
- `<iostream>`：`printProfilingStats` 需要 `std::ostream`。
- `<sys/time.h>`、`<stdint.h>`、`<string.h>`：給衍生類別預先準備的常用型別/函式，此檔案本身未直接使用。
- `"NvElementProfiler.h"`：必要 include，因為成員變數 `profiler` 型別為 `NvElementProfiler`，且用到其巢狀型別 `NvElementProfilerData`、`ProfilerField`。

### Public 介面

```cpp
virtual int isInError() { return is_in_error; }
virtual ~NvElement() {}
void getProfilingData(NvElementProfiler::NvElementProfilerData &data);
void printProfilingStats(std::ostream &out_stream = std::cout);
virtual void enableProfiling();
bool isProfilingEnabled();
```

- `isInError()`：virtual，回傳錯誤旗標，衍生類別可覆寫加入額外檢查。
- **virtual 解構子**：必要寫法，因為框架預期以 `NvElement*` 指標刪除衍生類別物件，若非 virtual 會造成衍生類別資源未釋放。
- `getProfilingData`：用參考傳遞（`&data`）避免複製整個資料結構，屬於效能考量。
- `printProfilingStats`：帶預設參數 `std::ostream &out_stream = std::cout`，可彈性導向檔案或字串串流。
- `enableProfiling`：virtual，允許衍生類別覆寫做額外初始化。

### Protected 區塊：抽象基底類別的設計手法

```cpp
NvElement(const char *name, NvElementProfiler::ProfilerField = NvElementProfiler::PROFILER_FIELD_NONE);
int is_in_error;
const char *comp_name;
NvElementProfiler profiler;
NvElement(const NvElement& that);
void operator=(NvElement const&);
```

- **建構子放在 protected**：外部不能直接 `new NvElement(...)`，只能被衍生類別呼叫 → 沒有 pure virtual function，但達到類似抽象類別的效果。
- `comp_name` 為 `const char*`：不擁有記憶體所有權，只存指標，呼叫端需保證字串生命週期。
- `profiler` 以「值」方式持有（組合 composition），生命週期綁定在 `NvElement` 物件上。
- 複製建構子與指定運算子**只宣告不定義**：C++11 前禁止複製的舊手法（連結階段找不到符號會失敗），因為類別內有裸指標成員，淺層複製會有風險。現代寫法應改用 `= delete`。

## NvElement.cpp 重點拆解

```cpp
void NvElement::getProfilingData(NvElementProfiler::NvElementProfilerData &data)
{
    profiler.getProfilerData(data);
}
```
單純委派給成員物件 `profiler`，本身不做運算 → 組合優於繼承的體現。

```cpp
void NvElement::printProfilingStats(std::ostream &out_stream)
{
    out_stream << "----------- Element = " << comp_name << " -----------" << std::endl;
    profiler.printProfilerData(out_stream);
    out_stream << "-------------------------------------" << std::endl;
}
```
注意：定義處不可重複寫預設參數 `= std::cout`（C++ 規則：預設參數只能在宣告處出現一次）。

```cpp
NvElement::NvElement(const char *name, NvElementProfiler::ProfilerField fields)
    :profiler(fields)
{
    is_in_error = 0;
    if (!name)
        is_in_error = 1;
    this->comp_name = name;
}
```
- 用**初始化列表**建構 `profiler`（必要，因無適用的預設建構子）。
- `!name` 檢查傳入指標是否為 `nullptr`，若是則設定錯誤旗標。

## 整體架構概念

```
NvElement（共通基底：錯誤狀態、profiling 介面）
   │
   ├── V4L2 相關元件（如 NvV4l2Element）
   │      └── NvVideoDecoder、NvVideoEncoder 等具體元件
   │
   └── 非 V4L2 元件（如 EGL/DRM 顯示輸出元件）
```

設計手法重點：
1. **共通行為往上抽**：錯誤狀態、profiling 放在最上層基底類別。
2. **protected 建構子 + virtual 解構子**：預期使用情境為 `NvElement *elem = new NvV4l2Element(...);`。
3. **組合優於繼承**：profiler 用持有物件的方式加入，避免污染繼承階層。
4. **virtual 方法預留多型擴充點**：`isInError()`、`enableProfiling()` 供下層元件覆寫。

## 實作參考程度的判斷

- 若專案直接使用 NVIDIA 提供的 library（`NvVideoDecoder` 等）：**必須**遵循此繼承架構，因為這些類別已綁定在此基底之上。
- 若目標是**自行重新設計精簡版解碼包裝器**（尤其偏好不使用 C++ 標準函式庫時）：
  - 可保留概念：共通錯誤狀態、profiling 掛勾點、組合式效能量測、protected 建構子防呆。
  - 可捨棄/改寫：`<iostream>` 依賴（改用自訂 log 或 C API）、`const char*` 生命週期風險（改用固定 char array 或 enum）、舊式禁止複製寫法（改用 `= delete`）。

## 待辦／延伸方向

- 下一步可拆解 `NvElementProfiler` 的時間量測實作。
- 或往下拆 `NvV4l2Element` / `NvVideoDecoder` 的繼承實作方式。
