---
title: 靜態工廠方法與建構子
source: 
author: 
published: 
created: 2026-09-10
description: 比較 C++ 建構子與靜態工廠方法的差異、優缺點與使用時機
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - cpp
  - oop
---

## 概念

靜態工廠方法（Static Factory Method）是用類別內的 `static` 成員函式取代（或補充）建構子來建立物件，函式名稱可自由命名，藉此表達「用什麼方式建立」的語意。

## 建構子的限制

- 名稱固定跟類別同名，無法表達建立方式的語意
- 不能依參數決定回傳不同子類別（建構子只能回傳自己這個型別）
- 每次呼叫必定建立新物件，無法快取或重用

## 範例：具名建立

```cpp
class Point {
public:
    static Point fromCartesian(int x, int y) { return Point(x, y); }
    static Point fromPolar(double r, double theta) {
        int x = static_cast<int>(r * cos_approx(theta));
        int y = static_cast<int>(r * sin_approx(theta));
        return Point(x, y);
    }
private:
    Point(int x, int y) : x_(x), y_(y) {}  // 建構子設為 private，強迫外部只能透過工廠方法建立
    int x_, y_;
    static double cos_approx(double theta);
    static double sin_approx(double theta);
};
```

## 優點

| 項目 | 說明 |
|---|---|
| 具名建立 | 語意比同參數型別的多個建構子更清楚，避免歧義 |
| 可回傳子類別 | 回傳基底類別指標/參考時，內部可依條件建立不同衍生類別 |
| 可控制物件產生 | 可做快取（Flyweight）、單例限制、物件池 |
| 搭配私有建構子 | 強制外部只能透過固定入口建立物件 |

## 與動態建立的關係

C++ 沒有正式的「動態建構子」語法，通常指靜態工廠方法搭配多型：回傳基底類別指標，實際 `new` 出哪個衍生類別在執行期才決定，這是 [[factory-method]] 的核心行為。

## 延伸閱讀

- [[factory-method]]：多型工廠方法，執行期決定建立哪個具體型別
- [[abstract-factory]]：一整組工廠，建立一整族相關聯的產品
- [[design-patterns-overview]]
