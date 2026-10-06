---
title: C++ 繼承（inheritance）技巧整理
source: 
author: 
published: 
created: 2026-08-28
description: 整理 C++ 繼承的存取層級、virtual function 開銷、菱形繼承、CRTP 與 composition 取捨
categorization: programming_language/C_CPP/CPP
tags:
  - cpp
  - inheritance
  - virtual-function
  - CRTP
  - 效能最佳化
  - 物件導向設計
---

## 1. 繼承存取層級：public / protected / private

```cpp
class Base {
public:
    int a;
protected:
    int b;
private:
    int c;
};

class Derived : public Base {
    // a 仍是 public，b 仍是 protected，c 完全看不到
};

class Derived2 : private Base {
    // a、b 都變成 private，等於「用 Base 實作，但外部看不出繼承關係」
};
```

- `public` 繼承：代表「是一種（is-a）」關係，是多型設計的基礎，最常用。
- `private` 繼承：代表「用...來實作」，實務少見，通常改用 composition 更清楚。
- `protected` 繼承：幾乎用不到，除非多層繼承架構需要中間層存取權限。

## 2. virtual function 與多型的底層代價

```cpp
class Shape {
public:
    virtual float area() const = 0;   // 純虛擬函式，Shape 是抽象類別
    virtual ~Shape() {}                 // 虛擬解構子
};

class Circle : public Shape {
    float r;
public:
    Circle(float r_) : r(r_) {}
    float area() const override { return 3.14159f * r * r; }
};
```

- 類別內只要有一個 `virtual` 函式，編譯器就會加一個隱藏指標（vptr）指向 vtable。
- 呼叫 `shape_ptr->area()` 時透過 vptr 查表，執行期才決定呼叫哪個版本 → 動態多型。
- **代價**：額外指標記憶體（64-bit 約 8 bytes）、間接跳轉、無法 inline、對分支預測不友善。
- 若多型呼叫發生在即時處理的熱路徑（hot path）上，開銷會被放大 → 與 [[jetson-video-decode]] 的即時影像處理場景相關，需留意。

## 3. virtual 解構子：不加會出事

```cpp
Shape* s = new Circle(5.0f);
delete s;   // Shape 解構子若非 virtual，只會呼叫 Shape::~Shape()
            // Circle 的解構子不會被呼叫，資源可能沒釋放
```

規則：只要類別打算被繼承、且可能透過基底類別指標刪除物件，解構子就必須是 `virtual`，即使目前沒有需要清理的資源也一樣，因為無法保證未來衍生類別不會有。

## 4. `override` 與 `final`

```cpp
class Base {
public:
    virtual void update(int x) {}
};

class Derived : public Base {
public:
    void update(float x) override {}
    // 編譯錯誤！簽名不符（int vs float），這是「隱藏」而非「覆寫」
};

class Final : public Base {
public:
    void update(int x) final {}  // 禁止再被進一步覆寫
};
```

沒有 `override` 時，簽名寫錯不會報錯，只會默默變成一個不相干的新函式，多型行為悄悄壞掉。每次覆寫都加 `override`，讓編譯器幫忙做二次檢查。

## 5. 多重繼承與菱形繼承（diamond problem）

```cpp
class Animal { public: int id; };
class Bird : public Animal {};
class Mammal : public Animal {};

class Bat : public Bird, public Mammal {};
// Bat 內有兩份 Animal，bat.id 會產生歧義，需寫成 bat.Bird::id
```

解法：virtual inheritance

```cpp
class Animal { public: int id; };
class Bird : virtual public Animal {};
class Mammal : virtual public Animal {};
class Bat : public Bird, public Mammal {};
// 現在只有一份 Animal，bat.id 不再有歧義
```

代價：物件記憶體佈局變複雜（需額外指標定位共用基底子物件），建構順序也較難直覺理解。能用 composition 或純抽象介面取代多重繼承時，優先取代。

## 6. CRTP：靜態多型，無 vtable 開銷

適合效能敏感場景，用編譯期多型取代執行期多型：

```cpp
template<typename Derived>
class ShapeBase {
public:
    float area() const {
        // static_cast 把自己 downcast 成衍生類別，呼叫其成員函式
        // 全部在編譯期決定，無 vptr、無間接跳轉，可被 inline
        return static_cast<const Derived*>(this)->area_impl();
    }
};

class Circle : public ShapeBase<Circle> {
    float r;
public:
    Circle(float r_) : r(r_) {}
    float area_impl() const { return 3.14159f * r * r; }
};
```

- `Circle` 繼承自「以自己為模板參數的 `ShapeBase<Circle>`」，故稱「詭異的遞迴模板樣式」。
- 缺點：`ShapeBase<Circle>`、`ShapeBase<Square>` 其實是不同型別，失去用同一基底指標處理不同衍生類別的能力（失去真正的執行期多型）。
- 適合已在編譯期就知道具體型別的情境，例如針對特定攝影機型號、特定推論後端寫的固定 pipeline，與 [[license-plate-recognition]] 專案中固定硬體 pipeline 的設計方向相符。

## 7. Composition over inheritance

能用「持有」解決的問題，不用「繼承」：

```cpp
// 不好：語意上 Robot 並不是一種 Logger
class Robot : public Logger { ... };

// 好：Robot「擁有」一個 Logger
class Robot {
    Logger logger;
public:
    void move() { logger.log("moving"); }
};
```

繼承代表強烈的 is-a 語意耦合，一旦用錯，後續牽扯多型與 vtable 佈局的重構會很痛苦。只有真的需要「透過基底類別指標統一操作不同衍生類別」（真正需要多型）時才選繼承；純粹想重用程式碼，composition 幾乎永遠更安全。

## 待思考

- 若要做模組化設計（例如不同推論後端、不同前處理策略），可評估：熱路徑上是否適合改用 CRTP 取代 virtual function，以換取效能。
