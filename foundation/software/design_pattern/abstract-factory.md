---
title: 抽象工廠（Abstract Factory）
source: 
author: 
published: 
created: 2026-09-10
description: 用一整組工廠建立一整族相關聯的產品，確保同一家族的產品互相搭配一致
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - cpp
---

## 動機

當系統需要支援多種「產品家族」，且同一家族內的產品必須一起被建立、互相搭配時，用抽象工廠保證一致性（例如 Windows 風格的按鈕不能配上 Mac 風格的視窗）。

## 結構（四個角色）

1. 抽象產品：如 `Button`、`Checkbox`
2. 具體產品：各家族下的實作，如 `WindowsButton`、`MacButton`
3. 抽象工廠：定義一組建立方法
4. 具體工廠：實作抽象工廠，建立同一家族的一整組具體產品

```cpp
class Button { public: virtual void draw() = 0; virtual ~Button() {} };
class Checkbox { public: virtual void draw() = 0; virtual ~Checkbox() {} };

class GUIFactory {
public:
    virtual Button* createButton() = 0;
    virtual Checkbox* createCheckbox() = 0;
    virtual ~GUIFactory() {}
};

class WindowsFactory : public GUIFactory {
public:
    Button* createButton() override { return new WindowsButton(); }
    Checkbox* createCheckbox() override { return new WindowsCheckbox(); }
};
```

呼叫端只透過 `GUIFactory*` 操作，換掉傳入的具體工廠，整組 UI 風格就一起換掉，不會出現混搭。

## 與工廠方法的關係

抽象工廠內部的每個 `createButton()`、`createCheckbox()` 本質上都是 [[factory-method]]。可以說：抽象工廠 = 把多個工廠方法打包成一個介面，統一管理一整族產品的建立。

| 項目 | 工廠方法 | 抽象工廠 |
|---|---|---|
| 建立的產品數量 | 一種產品的不同變化 | 多種產品，組成一個家族 |
| 新增產品種類 | 新增子類別即可 | 需修改抽象工廠介面，較麻煩 |
| 新增家族 | 不適用 | 新增一個具體工廠即可，很容易 |
| 一致性保證 | 無 | 有 |

取捨：新增家族容易，但新增產品種類麻煩（需改動 `GUIFactory` 介面，所有具體工廠跟著改）。

## 常見應用場景

- 跨平台 UI 套件
- 資料庫存取層：不同資料庫各自的 Connection/Command/Transaction 一起搭配
- 遊戲開發：不同主題的怪物工廠

## 記憶體管理

解構子需宣告為 `virtual`，誰呼叫 `createXxx()` 就要負責對應的 `delete`。

## 延伸閱讀

- [[factory-method]]
- [[static-factory-method]]
- [[design-patterns-overview]]
