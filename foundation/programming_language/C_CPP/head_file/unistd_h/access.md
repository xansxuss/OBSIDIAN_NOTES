---
title: access
source: https://claude.ai/chat/1c141b29-5005-49a3-93ef-e3cb0f472923
author:
published:
created: 2026-08-28
description:
tags:
---

# `access()` — unistd.h

## 宣告

```c
/* Test for access to NAME using the real UID and real GID. */
extern int access (const char *__name, int __type) __THROW __nonnull ((1));
```

## 說明

用來檢查目前執行程式的**真實使用者（real UID）**與**真實群組（real GID）**
是否有權限存取指定檔案，而非使用「有效使用者（effective UID）」來檢查。

在有 setuid/setgid 的程式中特別重要：可以讓程式檢查「原本呼叫者」
是否真的有權限，而不是用提升後的權限去檢查。

## 參數

| 參數      | 說明 |
|-----------|------|
| `__name`  | 要檢查的檔案路徑（`__nonnull ((1))` 表示不可為 `NULL`） |
| `__type`  | 要檢查的權限類型，可用 `|` 組合下列巨集 |

### `__type` 可用巨集

| 巨集     | 意義       |
|----------|------------|
| `F_OK`   | 檔案是否存在 |
| `R_OK`   | 是否可讀   |
| `W_OK`   | 是否可寫   |
| `X_OK`   | 是否可執行 |

## 回傳值

- 成功（有權限）：回傳 `0`
- 失敗：回傳 `-1`，並設定 `errno`（例如 `ENOENT`、`EACCES`）

## 補充 / 陷阱

- `__THROW`：C++ 用巨集，表示此函式不會丟出例外（因為是 C 函式）。
- **TOCTOU（Time-Of-Check-To-Time-Of-Use）風險**：
  `access()` 檢查完後才去開檔案，中間可能發生 race condition。
  在需要安全性的場合，建議直接嘗試 `open()` 並檢查回傳值，
  而不是先 `access()` 再 `open()`。

## 分類

`C_CPP/head_file/` — unistd.h 函式說明
