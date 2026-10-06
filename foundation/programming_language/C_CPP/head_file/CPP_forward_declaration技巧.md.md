---
title: C++ 前向宣告（Forward Declaration）技巧
source: 
author: 
published: 
created: 2026-09-21
description: 整理 C++ 前向宣告的規則與四種常用技巧，用來減少 #include、打破循環相依並隱藏實作細節。
categorization: foundation/programming_language/C_CPP/CPP/class
tags:
  - CPP
  - forward-declaration
  - incomplete-type
  - Pimpl
  - circular-dependency
  - 標頭檔
---

# C++ 前向宣告（Forward Declaration）技巧

## 核心概念

前向宣告只告訴編譯器「這個型別存在」，不提供完整定義，此時該型別稱為**不完整型別（incomplete type）**。

```cpp
class Foo;   // 只宣告，Foo 是 incomplete type
```

主要用途：
1. **減少 `#include`**：縮短編譯時間，避免改一個標頭檔就重編一大片。
2. **打破循環相依（circular dependency）**：A 需要 B、B 也需要 A。
3. **隱藏實作細節**：Pimpl、不透明控制程式碼（opaque handle）。

## 不完整型別的限制

| 可以 | 不行 |
|---|---|
| 宣告 `Foo*`、`Foo&` 的變數或成員 | 宣告 `Foo` 物件成員（不知道大小） |
| 函式宣告中使用 `Foo`（參數、回傳值） | 呼叫成員函式、存取成員 |
| 傳遞、比較、賦值 `Foo*` | 對 `Foo*` 做指標運算（需 `sizeof`） |
| `typedef` / `using` 別名 | 繼承、`sizeof(Foo)`、`new Foo`、`delete` 指標 |

> 判斷原則：編譯器需要**大小**或**成員**時，就必須有完整定義。

## 技巧一：打破循環相依

```cpp
// graph.h
#ifndef GRAPH_H
#define GRAPH_H

class Node;                 // 前向宣告，不 include node.h

class Graph {
public:
    Graph();
    ~Graph();
    void AddNode(Node* n);  // 只用指標，不需 Node 完整定義
    int  Count() const;
private:
    Node** m_nodes;         // 指標陣列，大小已知
    int    m_count;
    int    m_cap;
};

#endif
```

```cpp
// node.h
#ifndef NODE_H
#define NODE_H

class Graph;                // 前向宣告，不 include graph.h

class Node {
public:
    Node(Graph* owner, int id);
    Graph* Owner() const { return m_owner; }  // 只回傳指標
    int    Id() const    { return m_id; }
private:
    Graph* m_owner;
    int    m_id;
};

#endif
```

```cpp
// graph.cpp
#include "graph.h"
#include "node.h"           // 在 .cpp 才 include 完整定義

Graph::Graph() : m_nodes(nullptr), m_count(0), m_cap(0) {}
Graph::~Graph() { delete[] m_nodes; }   // 只釋放指標陣列，Node 所有權另行決定

void Graph::AddNode(Node* n) {
    if (m_count == m_cap) {             // 容量不足就擴充
        int newCap = (m_cap == 0) ? 4 : m_cap * 2;
        Node** p = new Node*[newCap];
        for (int i = 0; i < m_count; ++i) p[i] = m_nodes[i];
        delete[] m_nodes;
        m_nodes = p;
        m_cap = newCap;
    }
    m_nodes[m_count++] = n;
}

int Graph::Count() const { return m_count; }
```

**邏輯說明**
- 兩個 `.h` 互不 include，循環相依消失。
- 需要完整型別的操作（例如 `n->Id()`）放在 `.cpp`，由 `.cpp` include 完整定義。
- 原則：**標頭檔盡量前向宣告，`.cpp` 才 include**。

## 技巧二：Pimpl（隱藏實作）

```cpp
// decoder.h
#ifndef DECODER_H
#define DECODER_H

class DecoderImpl;          // 前向宣告，實作細節完全不外露

class Decoder {
public:
    Decoder();
    ~Decoder();             // 必須宣告，並在 .cpp 定義
    Decoder(const Decoder&) = delete;             // 禁止複製，避免重複釋放
    Decoder& operator=(const Decoder&) = delete;
    bool Open(const char* path);
private:
    DecoderImpl* m_impl;    // 只存指標
};

#endif
```

```cpp
// decoder.cpp
#include "decoder.h"

class DecoderImpl {         // 完整定義只在 .cpp
public:
    int m_fd;
    // 可放任何第三方標頭的型別，使用者看不到
};

Decoder::Decoder() : m_impl(new DecoderImpl()) { m_impl->m_fd = -1; }
Decoder::~Decoder() { delete m_impl; }   // 此處已是完整型別，可安全 delete

bool Decoder::Open(const char* path) {
    (void)path;
    return m_impl->m_fd >= 0;
}
```

**陷阱**：若 `~Decoder()` 寫成 inline 或由編譯器隱式產生，`delete m_impl` 會在只有前向宣告時編譯，屬於未定義行為（解構函式不會被呼叫），編譯器通常只給警告。**解構函式務必定義在 `.cpp`。**

## 技巧三：C 函式庫的不透明型別

適用於 [[FFmpeg]]、[[Jetson]] 等 C API 的封裝：

```cpp
// 自己的標頭檔，不想 include 整個 libavformat
struct AVFormatContext;     // C 風格 struct 也能前向宣告

class Demuxer {
public:
    Demuxer();
    ~Demuxer();
private:
    AVFormatContext* m_fmt; // 只存指標
};
```

- 前向宣告的關鍵字（`struct`）要與原始定義一致，否則 MSVC 可能警告。
- 若原型別是 typedef 出來的匿名型別，**不能**前向宣告，只能 include 原標頭檔。

## 技巧四：enum 與 template

```cpp
enum class PixelFormat : int;   // C++11 起，指定底層型別的 enum 可前向宣告
void SetFormat(PixelFormat f);

template <typename T> class Buffer;   // 模板前向宣告
Buffer<int>* MakeBuffer();            // 使用時仍只是指標
```

- 沒指定底層型別的一般 `enum` **不能**前向宣告（不知道大小）。
- 必須寫成 `enum class X : 底層型別;`，且與完整定義的底層型別一致。

## 注意事項

1. 不要前向宣告標準函式庫型別（未定義行為）；不使用標準函式庫時不受影響。
2. `class Foo;` 與 `struct Foo;` 雖可互換，但混用會讓部分編譯器警告，建議統一。
3. 標頭檔只放前向宣告，`.cpp` 才 include；成員為指標或參考時不需完整定義。
4. **繼承與值成員一定要完整定義**，這是 `incomplete type` 錯誤最常見的來源。
5. inline 函式與模板若本體需要完整型別，要把本體移到 `.cpp` 或 include 完整定義。
6. 取捨：少了 include 就少了自動檢查，改 class 名稱時要同步更新所有前向宣告。

## 英文用語

- forward declaration：前向宣告
- incomplete type / complete type：不完整型別 / 完整型別
- circular dependency：循環相依（物件之間的參考才用 cyclic reference）
- opaque type / opaque handle：不透明型別 / 不透明控制程式碼

## 相關筆記

- [[C++_Pimpl_手法]]