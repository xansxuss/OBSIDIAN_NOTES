---
title: Observer（觀察者模式）
source: 
author: 
published: 
created: 2026-09-10
description: 主題狀態改變時自動通知多個觀察者，形成一對多的解耦通知關係
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
---

## 動機

當一個物件（Subject / 主題）的狀態改變時，需要通知多個其他物件（Observer / 觀察者），但不希望兩者之間寫死緊密依賴。典型場景：GUI 事件通知、資料模型同步、事件系統。核心是「一對多依賴：狀態改變時所有依賴者自動收到通知並更新」。

## 基本結構

三個角色：抽象 Observer（`update()` 介面）、抽象 Subject（維護 Observer 清單、訂閱/取消訂閱/通知）、具體 Subject/Observer。

```cpp
class Observer { public: virtual void onNotify(int newValue) = 0; virtual ~Observer() {} };

class Subject {
public:
    void attach(Observer* obs) { /* 加入清單 */ }
    void detach(Observer* obs) { /* 從清單移除 */ }
protected:
    void notifyAll(int newValue) {
        for (int i = 0; i < observerCount_; ++i) observers_[i]->onNotify(newValue);
    }
private:
    Observer* observers_[16];
    int observerCount_ = 0;
};

class TemperatureSensor : public Subject {
public:
    void setTemperature(int t) { temperature_ = t; notifyAll(temperature_); }
private:
    int temperature_ = 0;
};
```

## 運作邏輯重點

- `Subject` 完全不知道具體 Observer 是哪個類別，只認得抽象介面——新增新 Observer 不需修改 Subject 任何一行
- 通知方向永遠是 Subject → Observer 單向，Observer 不應反過來呼叫 Subject 改狀態，否則容易互相呼叫產生無窮迴圈
- `attach`/`detach` 讓訂閱關係可在執行期動態調整

## 常見陷阱

- **生命週期管理**：Observer 被刪除卻忘記 `detach()`，Subject 留著懸空指標（dangling pointer），下次通知時存取已釋放記憶體
- **通知順序不保證語意**：多個 Observer 之間若有依賴關係，順序寫在清單裡並不直觀，容易埋下難察覺的 bug

## 延伸閱讀

- [[mediator]]：常搭配 Observer 機制實作「同事通知中介者」，但關注點是降低多物件間耦合而非通知本身
- [[design-patterns-overview]]
