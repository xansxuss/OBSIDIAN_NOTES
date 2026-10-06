---
title: access
source: https://claude.ai/chat/1c141b29-5005-49a3-93ef-e3cb0f472923
author:
published:
created: 2026-08-28
description:
tags:
categorization:
---

# `access()` 範例 — 檢查 nvdec 裝置節點是否存在

## 情境

Jetson 平台上 nvdec 解碼裝置節點可能有兩種名稱，先用 `access()` 確認
哪一個存在，再決定要 open 哪個裝置。

```cpp
#include <cstdio>
#include <cerrno>
#include <unistd.h>

#define DECODER_DEV     "/dev/nvhost-nvdec"
#define DECODER_DEV_ALT "/dev/v4l2-nvdec"

int main()
{
    if (access(DECODER_DEV, F_OK) == 0)
    {
        printf("[OK] %s 存在\n", DECODER_DEV);
    }
    else
    {
        printf("[FAIL] %s 不存在 (errno=%d)\n", DECODER_DEV, errno);
    }

    if (access(DECODER_DEV_ALT, F_OK) == 0)
    {
        printf("[OK] %s 存在\n", DECODER_DEV_ALT);
    }
    else
    {
        printf("[FAIL] %s 不存在 (errno=%d)\n", DECODER_DEV_ALT, errno);
    }

    return 0;
}
```

## 編譯

```bash
g++ -std=c++17 -Wall -Wextra test_decoder_dev.cpp -o test_decoder_dev
```

## 執行

```bash
./test_decoder_dev
```

## 補充

- `F_OK` 只檢查檔案／裝置節點是否存在，不檢查讀寫權限。
- 若要同時確認讀寫權限（nvdec 裝置通常屬於 `video` 群組），
  改用 `R_OK | W_OK`：

```cpp
if (access(DECODER_DEV, R_OK | W_OK) == 0)
```

- 此範例只用 `<cstdio>` 的 `printf`，未使用 `<iostream>`，
  盡量避開 C++ 標準函式庫（`access()` 本身屬於 POSIX/C 函式）。

## 分類

`head_file/unistd_h/` — access() 實務範例
