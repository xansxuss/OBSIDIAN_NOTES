---
title: Strategy（策略模式）
source: 
author: 
published: 
created: 2026-09-10
description: 把演算法抽出成可替換物件，讓使用者在執行期動態切換做法而不需大量 if/switch
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - dependency-injection
---

## 動機

同一件事有多種做法，希望在執行期動態切換，且避免用大量 `if/else` 或 `switch` 把每種做法寫在同一函式裡（新增做法就要改這個函式，違反開放封閉原則）。核心：把演算法本身抽出成獨立、可替換的物件，使用它的類別（Context）只透過共同介面呼叫。

## 基本結構

```cpp
class SortStrategy { public: virtual void sort(int* arr, int size) = 0; virtual ~SortStrategy() {} };

class BubbleSort : public SortStrategy { public: void sort(int* arr, int size) override; };
class InsertionSort : public SortStrategy { public: void sort(int* arr, int size) override; };

class Sorter {
public:
    void setStrategy(SortStrategy* strategy) { strategy_ = strategy; }
    void sortArray(int* arr, int size) { if (strategy_) strategy_->sort(arr, size); }
private:
    SortStrategy* strategy_ = nullptr;
};
```

呼叫端可在執行期用 `setStrategy` 換掉演算法，`Sorter` 本身完全不變。

## 運作邏輯重點

- `Sorter`（Context）只認得 `SortStrategy*` 抽象介面，實際跑哪種演算法由外部注入——這是**依賴注入（Dependency Injection）**的具體實踐
- 新增新演算法只需新增一個子類別，`Sorter` 不用修改，符合開放封閉原則
- 跟 if/else 硬寫的差異：if/else 把「選擇」跟「實作」混在同一函式；Strategy 把兩者拆成獨立關注點

## 與其他模式比較

| 模式 | 關心的問題 | 抽換的是什麼 |
|---|---|---|
| [[factory-method]] | 建立哪一種物件 | 物件的建立過程 |
| Strategy | 執行哪一種演算法 | 物件建立好後某個行為的實作方式 |
| [[observer]] | 狀態改變時通知誰 | 一對多的通知關係 |

Strategy 常與 Factory 搭配：用 Factory 建立要用的 Strategy 物件，兩者各自負責不同關注點。

### 與 State 的相似與差異

寫法上與 [[state]] 幾乎一樣（都用虛擬函式抽換行為），差異在於：Strategy 由呼叫端主動指定要用哪個策略，各實作間通常獨立無順序；State 由狀態物件自己決定下一個狀態，通常有明確轉換規則形成狀態機。

## 延伸閱讀

- [[factory-method]]
- [[state]]
- [[design-patterns-overview]]
