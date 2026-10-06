---
title: Mediator（中介者模式）
source: 
author: 
published: 
created: 2026-09-10
description: 引入中介者物件讓多個物件只跟中介者溝通，降低物件間多對多的直接耦合
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
---

## 動機

一群物件互相直接呼叫對方（多對多關係），物件數量一多依賴關係會變得非常複雜（義大利麵條式耦合）。引入一個中介者物件，讓所有物件只跟中介者溝通，不直接互相呼叫。典型場景：聊天室、GUI 表單元件間的連動。

## 基本結構

```cpp
class Colleague;
class Mediator { public: virtual void notify(Colleague* sender, const char* event) = 0; virtual ~Mediator() {} };

class Colleague {
public:
    Colleague(Mediator* mediator) : mediator_(mediator) {}
protected:
    Mediator* mediator_;
};

class Checkbox : public Colleague {
public:
    Checkbox(Mediator* mediator) : Colleague(mediator) {}
    void check() { checked_ = true; mediator_->notify(this, "checked"); }
private:
    bool checked_ = false;
};

class TextBox : public Colleague {
public:
    TextBox(Mediator* mediator) : Colleague(mediator) {}
    void setEnabled(bool enabled) { enabled_ = enabled; }
private:
    bool enabled_ = true;
};

class FormMediator : public Mediator {
public:
    void notify(Colleague* sender, const char* event) override {
        if (sender == checkbox_) textBox_->setEnabled(false);  // 業務規則集中寫在這裡
    }
private:
    Checkbox* checkbox_ = nullptr;
    TextBox* textBox_ = nullptr;
};
```

## 運作邏輯重點

- `Checkbox` 跟 `TextBox` 完全不認識彼此，都只持有一個 `Mediator*`——原本 N 個物件兩兩互相呼叫（N 平方等級的關係數量）被收斂成各自只跟 1 個中介者溝通（線性關係數量）
- 業務邏輯規則集中寫在中介者裡，變更時只需改一個地方，但中介者本身容易隨同事數量增加而變得肥大、難維護，這是常見的取捨

## 與 Observer 的關係

Mediator 內部很常用 [[observer]] 的機制來實作「同事通知中介者」這一段，差異在於關注點：Observer 關心「一對多的通知」本身；Mediator 關心「用第三方物件降低多個物件間的耦合」，通知只是達成目的的手段之一。

## 延伸閱讀

- [[observer]]
- [[design-patterns-overview]]
