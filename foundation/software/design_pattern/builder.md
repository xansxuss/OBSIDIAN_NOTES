---
title: Builder（建造者模式）
source: 
author: 
published: 
created: 2026-09-10
description: 把複雜物件的建構過程拆成一系列具名步驟，解決 telescoping constructor 問題
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - cpp
---

## 動機

當物件建構參數多、部分選填、步驟需照順序執行，或同一套步驟能產生不同表示時，硬塞所有參數到建構子會出現 telescoping constructor 問題（一堆重載建構子疊參數）。Builder 把建構步驟拆成一系列具名方法，讓呼叫端一步步組裝。

## 基本結構：Fluent Interface

```cpp
class PizzaBuilder {
public:
    PizzaBuilder& size(int s) { pizza_.setSize(s); return *this; }
    PizzaBuilder& cheese() { pizza_.setCheese(true); return *this; }
    Pizza build() { return pizza_; }
private:
    Pizza pizza_;
};

Pizza p = PizzaBuilder().size(12).cheese().build();
```

每個方法回傳 `*this`，讓呼叫可以一路串下去，稱為 **Fluent Interface（流暢介面）**。

## 進階：Director 分離「步驟」與「順序」

GoF 原始定義還有 **Director** 角色，負責定義「用哪個順序呼叫 Builder 的方法」：

```cpp
class HouseBuilder {
public:
    virtual void buildFoundation() = 0;
    virtual void buildStructure() = 0;
    virtual void buildRoof() = 0;
    virtual ~HouseBuilder() {}
};

class ConstructionDirector {
public:
    void construct(HouseBuilder& builder) {
        builder.buildFoundation();
        builder.buildStructure();
        builder.buildRoof();
    }
};
```

「先地基、再結構、再屋頂」這個順序知識只存在 `Director` 裡一份，換不同的具體 `HouseBuilder`（木屋/磚屋）順序不變，只有每步驟的實作不同；順序若要調整只需改 `Director`。

## 與工廠模式的差異

| 項目 | Factory | Builder |
|---|---|---|
| 關心的問題 | 建立哪一種物件 | 如何一步步組裝複雜物件 |
| 呼叫方式 | 一次呼叫拿到完整物件 | 分多次呼叫累積狀態，最後 `build()` |
| 適用時機 | 有多種型別選擇 | 結構複雜、參數多、可選欄位多 |

## 延伸閱讀

- [[factory-method]]
- [[design-patterns-overview]]
