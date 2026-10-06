---
title: Template Method（樣板方法）
source: 
author: 
published: 
created: 2026-09-10
description: 基底類別固定演算法流程順序，讓子類別只實作或覆寫其中特定步驟
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - inheritance
---

## 動機

多個子類別共享同一套演算法骨架，但骨架中某幾個步驟的細節因子類別而異時，把整個骨架寫在基底類別的一個不可覆寫方法裡，把會變動的步驟宣告成虛擬函式讓子類別各自實作。核心：「流程順序」由基底類別控制，「每個步驟怎麼做」由子類別決定。

## 基本結構

```cpp
class DataProcessor {
public:
    void process() {                    // 樣板方法：固定流程順序
        readData();
        parseData();
        if (needsValidation()) validateData();  // hook（勾點）
        saveResult();
    }
    virtual ~DataProcessor() {}
protected:
    virtual void readData() = 0;
    virtual void parseData() = 0;
    virtual void saveResult() = 0;
    virtual bool needsValidation() { return true; }  // 有預設實作的勾點
    virtual void validateData() {}
};

class JSONProcessor : public DataProcessor {
protected:
    void readData() override;
    void parseData() override;
    void saveResult() override;
    bool needsValidation() override { return false; }  // 跳過驗證步驟
};
```

## 運作邏輯重點

- `process()` 本身不希望被子類別覆寫，因為「流程順序」是這個模式故意固定住不讓子類別更動的部分——與其他模式不同：其他模式通常整組行為可抽換，這裡只開放「步驟內容」，不開放「步驟順序」
- `needsValidation()` 這種有預設實作、子類別可選擇覆寫的虛擬函式稱為 **hook（勾點）**，用來微調流程分支
- 好萊塢原則（Hollywood Principle）：「Don't call us, we'll call you」——子類別不主動呼叫基底類別方法驅動流程，而是基底類別的 `process()` 在適當時機主動呼叫子類別實作的步驟

## 與 Strategy 的差異

Template Method 用繼承（is-a），多個子類別共用同一套骨架；[[strategy]] 用組合（has-a），整個演算法整包抽換，Context 與演算法無繼承關係。一般設計原則建議優先用組合而非繼承，但若多個變化版本高度共用同一套固定流程、只有少數步驟不同，Template Method 仍是直接的做法。

## 延伸閱讀

- [[strategy]]
- [[design-patterns-overview]]
