---
title: 多型工廠方法（Factory Method）
source: 
author: 
published: 
created: 2026-09-10
description: 用共同介面包裝多種具體實作，執行期決定實際建立哪個具體型別
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - cpp
  - polymorphism
---

## 概念

用一個共同介面（基底類別）包裝多種具體實作，透過工廠方法在執行期依條件決定實際建立哪一個具體型別，呼叫端不需要知道底層是哪個 class。三個角色：抽象產品、具體產品、工廠方法。

## 範例

```cpp
class Logger {
public:
    virtual void write(const char* msg) = 0;
    virtual ~Logger() {}
};

class ConsoleLogger : public Logger { public: void write(const char* msg) override; };
class FileLogger : public Logger {
public:
    FileLogger(const char* path) : path_(path) {}
    void write(const char* msg) override;
private:
    const char* path_;
};

enum class LoggerType { Console, File };
Logger* createLogger(LoggerType type, const char* filePath = nullptr) {
    switch (type) {
        case LoggerType::Console: return new ConsoleLogger();
        case LoggerType::File:    return new FileLogger(filePath);
    }
    return nullptr;
}
```

## 為什麼能「動態」：兩層意義

| 層面 | 發生時機 | 機制 |
|---|---|---|
| 建立哪個型別 | 執行期，依參數決定 | 工廠方法內的判斷邏輯 |
| 呼叫哪個函式 | 執行期，依物件實際型別決定 | C++ 虛擬函式機制（vtable） |

指標的**靜態型別**（例如 `Logger*`）與**動態型別**（實際指向的物件）不同，呼叫虛擬函式時透過物件的 vtable 指標查表分派，因此「編譯期只知道基底類別」也能在「執行期正確呼叫到衍生類別的實作」。

## 進階：Factory Method Pattern（GoF）

把「建立邏輯」也做成可覆寫的虛擬函式，讓子類別自行決定要建立什麼，而不是集中寫在一個 switch 裡：

```cpp
class Dialog {
public:
    void render() { Button* btn = createButton(); btn->draw(); delete btn; }
protected:
    virtual Button* createButton() = 0;
};
class WindowsDialog : public Dialog {
protected: Button* createButton() override { return new WindowsButton(); }
};
```

新增變化時只需新增子類別，不用修改集中的 switch，符合開放封閉原則。

## 記憶體管理注意事項

- 基底類別解構子必須是 `virtual`，否則透過基底類別指標 `delete` 衍生類別物件時資源不會正確釋放
- 誰呼叫 `createXxx()` 建立物件，誰就要負責 `delete`

## 延伸閱讀

- [[static-factory-method]]：一般（非多型）的靜態工廠方法
- [[abstract-factory]]：把多個工廠方法打包成一個介面，管理一整族產品
- [[strategy]]：寫法相似，但關注點是「換行為」而非「建立物件」
- [[design-patterns-overview]]
