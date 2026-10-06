---
title: Visitor（訪問者模式）
source: 
author: 
published: 
created: 2026-09-10
description: 對穩定的類別階層新增各式操作而不修改既有類別，核心機制為雙重分派
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - double-dispatch
---

## 動機

有一個穩定的類別階層（一組固定的具體類別），但需要不斷新增各式操作，且不想每次新增操作就修改每個類別本身。把「操作」抽出成獨立的 Visitor 類別，每個具體類別提供 `accept(visitor)`，把自己交給 visitor 處理，讓 visitor 依實際型別執行對應邏輯。

## 基本結構（雙重分派 Double Dispatch）

```cpp
class Circle; class Square;

class ShapeVisitor {
public:
    virtual void visit(Circle& c) = 0;
    virtual void visit(Square& s) = 0;
    virtual ~ShapeVisitor() {}
};

class Shape { public: virtual void accept(ShapeVisitor& visitor) = 0; virtual ~Shape() {} };

class Circle : public Shape {
public:
    void accept(ShapeVisitor& visitor) override { visitor.visit(*this); }
};

class AreaVisitor : public ShapeVisitor {
public:
    void visit(Circle& c) override { totalArea_ += 3.14159 * c.radius() * c.radius(); }
    void visit(Square& s) override { totalArea_ += s.side() * s.side(); }
private:
    double totalArea_ = 0.0;
};
```

新增操作（例如 `DrawVisitor`）完全不需要改動 `Circle`、`Square`。

## 為什麼需要雙重分派

若 `Shape* shape = &circle;` 只呼叫 `visitor.visit(*shape)`，因為 `*shape` 的**靜態型別**是 `Shape&`，而 `ShapeVisitor` 沒有 `visit(Shape&)` 重載，會編譯錯誤或呼叫錯版本。

正確做法是透過 `shape->accept(visitor)`：
1. 第一次分派（虛擬函式 `accept`）依 `shape` **動態型別**決定執行 `Circle::accept` 還是 `Square::accept`
2. 在 `Circle::accept` 內部，`*this` 的靜態型別已確定是 `Circle&`，因此 `visitor.visit(*this)` 在編譯期就能正確解析到 `visit(Circle&)`

這種「先靠虛擬函式決定物件實際型別，再靠函式重載決定呼叫哪個 visit」的兩階段機制稱為**雙重分派（Double Dispatch）**，比一般虛擬函式呼叫（只做一次分派）多做了一次。

## 取捨

| 好處 | 壞處 |
|---|---|
| 新增操作完全不用修改既有 `Shape` 子類別 | 新增一種新的 `Shape` 子類別需修改 `ShapeVisitor` 介面，所有既有 Visitor 都要跟著改 |

與 [[abstract-factory]] 的取捨方向相反：Visitor 適合「類別階層穩定、操作經常變動」的情境；若類別階層本身常新增子類別，Visitor 反而難維護。

## 延伸閱讀

- [[abstract-factory]]
- [[design-patterns-overview]]
