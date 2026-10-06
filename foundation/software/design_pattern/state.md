---
title: State（狀態模式）
source: 
author: 
published: 
created: 2026-09-10
description: 把每種狀態實作成獨立類別，物件把行為委派給目前狀態物件，避免大量狀態判斷
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - state-machine
---

## 動機

當物件行為隨內部狀態不同而改變，且狀態間有明確轉換規則時，用一堆 `if(state==X)...else if...` 會讓程式碼難以維護。State 模式把每種狀態各自實作成一個類別，物件本身只持有目前狀態物件，並把行為委派給它。典型場景：訂單狀態機、TCP 連線狀態、遊戲角色狀態。

## 基本結構

```cpp
class TrafficLight;

class LightState { public: virtual void handle(TrafficLight& light) = 0; virtual ~LightState() {} };

class TrafficLight {
public:
    TrafficLight(LightState* initial) : state_(initial) {}
    void setState(LightState* newState) { state_ = newState; }
    void next() { state_->handle(*this); }
private:
    LightState* state_;
};

class RedLight : public LightState { public: void handle(TrafficLight& light) override; };
class GreenLight : public LightState { public: void handle(TrafficLight& light) override; };
class YellowLight : public LightState { public: void handle(TrafficLight& light) override; };

static RedLight redState; static GreenLight greenState; static YellowLight yellowState;
void RedLight::handle(TrafficLight& light) { light.setState(&greenState); }
void GreenLight::handle(TrafficLight& light) { light.setState(&yellowState); }
void YellowLight::handle(TrafficLight& light) { light.setState(&redState); }
```

## 運作邏輯重點

- `TrafficLight::next()` 完全不知道目前是哪個狀態，只呼叫 `state_->handle(*this)`——「該轉換到哪個狀態」的知識被下放到各具體狀態類別自己身上
- 新增一種狀態只需新增一個類別、調整相鄰狀態的轉換邏輯，不用改集中的巨大 switch
- 狀態物件若無成員變數，可共用同一個靜態實例（如 `redState`），不需每次轉換都重新配置記憶體

## 與 Strategy 的差異（結構幾乎一樣，容易混淆）

| 項目 | [[strategy]] | State |
|---|---|---|
| 誰決定換成哪一個 | 呼叫端主動指定 | 狀態物件自己決定下一個狀態 |
| 各實作間關係 | 通常互相獨立，無順序概念 | 通常有明確轉換規則，形成狀態機 |
| Context 是否知道有別的選項 | 知道有多種策略可選 | 通常不知道有哪些狀態、更不知道轉換規則 |

## 延伸閱讀

- [[strategy]]
- [[design-patterns-overview]]
