---
title: "fcntl-linux"
source: "https://gemini.google.com/app/e0e7d044a09a20f0?hl=zh-TW"
author:
published:
created: 2026-08-24
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
這份原始碼是 GNU C Library (glibc) 中的內部標頭檔 `<bits/fcntl-linux.h>` 。主要功能為針對 Linux 作業系統核心，定義檔案控制（File Control）、檔案開啟旗標（Open Flags）、檔案鎖（File Locking）以及 Linux 特有的系統呼叫（System Calls）相關巨集與常數。

以下為此標頭檔的完整詳細解析：

**1\. 架構與保護機制**

- **防止直接引用** ：檔案開頭使用 `#ifndef _FCNTL_H` 判斷，若使用者直接 `#include <bits/fcntl-linux.h>` 會觸發預處理編譯錯誤。使用者必須統一引用 `<fcntl.h>` 。
	H+ 1
- **架構共用與覆蓋機制** ：包含不同 Linux 架構（如 x86, ARM, MIPS 等）共用的巨集。透過 `#ifndef` 的保護，特定架構可以在包含此檔案前先行定義特定數值（例如不同的 bitmask）。
	H+ 1

**2\. 核心功能區塊解析**

**開啟與狀態旗標（Open / File Status Flags）** 定義傳入 `open()` 或 `fcntl()` 的存取模式與行為旗標：

- **存取模式 mask** ： `O_ACCMODE` (0003) 用於遮罩提取 `O_RDONLY` (00), `O_WRONLY` (01), `O_RDWR` (02)。
	H
- **POSIX / 基礎旗標** ： `O_CREAT` (0100), `O_EXCL` (0200), `O_TRUNC` (01000), `O_APPEND` (02000), `O_NONBLOCK` (04000) 等。
	H
- **特化與效能旗標** ：
	- `O_DIRECT` ：繞過 Linux Page Cache，進行直接 I/O。
		H
		- `O_CLOEXEC` ：開啟檔案時直接設定 `FD_CLOEXEC` 旗標，防止多執行緒環境下 `fork` + `exec` 產生的 Race Condition。
		H
		- `O_NOATIME` ：讀取時不安裝/更新檔案的 last access time（atime），適用於讀取密集型系統。
		H
		- `O_PATH` ：僅取得檔案路徑的描述子（File Descriptor），不進行實際檔案開啟操作。
		H
		- `O_TMPFILE` ：建立匿名暫存檔（配合 `O_DIRECTORY` ），檔案在關閉時自動銷毀，防範暫存檔預測攻擊。
		H

**檔案鎖機制（File Locking）**

- **POSIX Record Locks** ： `F_GETLK`, `F_SETLK`, `F_SETLKW` 。程式碼中包含針對 32 位元與 64 位元檔案偏移量（ `__USE_FILE_OFFSET64` ）的相容性轉換處理，如自動轉接至 `F_GETLK64` 等。
	H+ 1
- **OFD Locks (Open File Description Locks)** ：定義 `F_OFD_GETLK` (36), `F_OFD_SETLK` (37), `F_OFD_SETLKW` (38)。與傳統 POSIX process-associated 鎖不同，OFD 鎖綁定於 Open File Description（可跨 `fork` / `CLONE_FILES` 共享），解決了多執行緒庫（Multi-threading libraries）調用 `close()` 意外釋放鎖的問題。
	H+ 1
- **BSD `flock` 支援** ：定義 `LOCK_SH`, `LOCK_EX`, `LOCK_NB`, `LOCK_UN` 。
	H

**`fcntl()` 操作指令（Commands）**

- **描述子操作** ： `F_DUPFD` (0), `F_GETFD` (1), `F_SETFD` (2), `F_GETFL` (3), `F_SETFL` (4), `F_DUPFD_CLOEXEC` (1030)。
	H
- **非同步 I/O 與 Owner 管理** ： `F_SETOWN`, `F_GETOWN`, `F_SETOWN_EX`, `F_GETOWN_EX` 。配對的 `struct f_owner_ex` 可以精確定義接收 `SIGIO` 的目標為 Thread (TID)、Process (PID) 或 Process Group (PGRP)。
	H+ 1
- **高級 Linux 特性指令** ：
	- **Pipe 控制** ： `F_SETPIPE_SZ` / `F_GETPIPE_SZ` 用於動態調整 Pipe 緩衝區大小。
		H
		- **Memfd Seals** ： `F_ADD_SEALS` / `F_GET_SEALS` 配合 `F_SEAL_SHRINK`, `F_SEAL_GROW`, `F_SEAL_WRITE` 等旗標，限制 shared memory 檔案的變更操作。
		H
		- **I/O 寫入生命週期提示** ： `F_GET_RW_HINT` / `F_SET_RW_HINT` （包含 `RWH_WRITE_LIFE_SHORT`, `LONG` 等），通知底層 NVMe / SSD 快閃記憶體最佳化 Write Amplification。
		H

**預先讀取與 I/O 建議（POSIX Advise & Directory Notification）**

- **`posix_fadvise` 參數** ： `POSIX_FADV_NORMAL`, `POSIX_FADV_RANDOM`, `POSIX_FADV_SEQUENTIAL`, `POSIX_FADV_WILLNEED`, `POSIX_FADV_DONTNEED` 。
	H
- **目錄變更通知（dnotify）** ： `DN_ACCESS`, `DN_MODIFY`, `DN_CREATE`, `DN_DELETE` 等（現多已被 `inotify` / `fanotify` 取代）。
	H

**3\. Linux 專屬 API 函式宣告**

標頭檔尾端包含多個 Linux 特有的零拷貝（Zero-Copy）與高效能系統呼叫介面：

| 函式名稱 | 主要用途與特點 |
| --- | --- |
| `readahead()` | 提示 Linux 核心預先將指定的檔案區域讀入 Page Cache。  H |
| `sync_file_range()` | 提供細粒度（Fine-grained）的檔案區域同步（Sync），可控制是否等待 dirty pages 寫入。  H |
| `vmsplice()` | 將使用者空間記憶體（ `struct iovec` ）直接 Splice 進入 Pipe 緩衝區。  H |
| `splice()` | 在兩個檔案描述子之間（其中一個需為 Pipe）直接移動資料，全程在核心空間完成，無需 Copy 到使用者空間。  H |
| `tee()` | 在兩個 Pipe 之間複製資料（零拷貝）。  H |
| `fallocate()` / `fallocate64()` | 預先分配（或 Hole punch 釋放）檔案磁碟空間。  H |
| `name_to_handle_at()` / `open_by_handle_at()` | 允許透過 File Handle 而非路徑字串來開啟檔案（常用於用戶態檔案伺服器如 NFS）。  H |