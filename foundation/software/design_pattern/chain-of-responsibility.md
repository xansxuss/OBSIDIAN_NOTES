---
title: Chain of Responsibility（責任鏈模式）
source: 
author: 
published: 
created: 2026-09-10
description: 把多個處理者串成一條鏈，請求沿鏈傳遞直到有人處理，發送者不需知道最終處理者是誰
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
---

## 動機

一個請求可能被多個物件中的其中一個處理，但呼叫端不需要知道究竟是哪一個。把處理者串成一條鏈，請求沿鏈傳遞直到有人處理為止。典型場景：分層例外處理、GUI 事件冒泡、請假簽核流程（依權限決定誰能核准）。

## 基本結構

```cpp
class Approver {
public:
    void setNext(Approver* next) { next_ = next; }
    void handleRequest(int amount) {
        if (canApprove(amount)) doApprove(amount);
        else if (next_) next_->handleRequest(amount);   // 自己處理不了，往下一個丟
    }
    virtual ~Approver() {}
protected:
    virtual bool canApprove(int amount) = 0;
    virtual void doApprove(int amount) = 0;
private:
    Approver* next_ = nullptr;
};

class TeamLead : public Approver { protected: bool canApprove(int amount) override { return amount <= 1000; } void doApprove(int amount) override; };
class Manager  : public Approver { protected: bool canApprove(int amount) override { return amount <= 10000; } void doApprove(int amount) override; };
class Director : public Approver { protected: bool canApprove(int amount) override { return true; } void doApprove(int amount) override; };
```

呼叫端把三者用 `setNext` 串起來，`lead.handleRequest(amount)` 會依金額沿鏈往上傳，直到有人能核准。

## 運作邏輯重點

- 每個 `Approver` 只知道自己能不能處理、以及下一個是誰，完全不知道整條鏈有多長、最終誰會處理
- 新增一層審核只需建立新類別並調整串接順序，不用修改既有類別
- 要注意鏈的**尾端處理**：若整條鏈都沒人能處理，目前寫法會靜默地什麼都不做，實務上通常需在鏈尾加明確的預設行為（拋例外、記錄 log），避免請求被默默吃掉

## 與 Command 的搭配

常搭配 [[command]]：把「請求」包裝成 Command 物件在鏈上傳遞，而不是傳單一數值，讓鏈上每個節點取得更完整的請求上下文，也方便之後記錄或 undo。

## 延伸閱讀

- [[command]]
- [[design-patterns-overview]]
