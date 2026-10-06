---
title: C++ 多載 (Overloading) 與多型 (Polymorphism) 差異
source: 
author: 
published: 
created: 2026-09-07
description: 比較 C++ 中函式多載、執行期多型、覆寫三者的機制與差異
categorization: foundation/programming_language/C_CPP/class
tags:
  - cpp
  - overloading
  - polymorphism
  - virtual-function
  - oop
---

## 多載 (Function Overloading)

**定義**：同一作用域內允許多個「同名函式」，只要**參數列表**（型別、數量、順序）不同即可。編譯器在**編譯期**依呼叫時的引數決定呼叫哪一版本。

```cpp
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

int main() {
    add(1, 2);        // 呼叫 (int, int)
    add(1.5, 2.5);     // 呼叫 (double, double)
    add(1, 2, 3);      // 呼叫三參數版本
    return 0;
}
```

### 關鍵特性
- 屬於**靜態綁定 (Static Binding)**，又稱編譯期多型
- 只比對「函式簽章 (signature)」，執行期無額外判斷，效能無額外開銷
- 只看參數列表，**回傳型別不同不能構成多載**
- 與繼承、虛擬函式無關，純函式也可以多載

---

## 執行期多型 (Runtime Polymorphism)

**定義**：透過**繼承**與**虛擬函式 (virtual function)**，讓基底類別的指標或參考在**執行期**依實際指向的物件型別，動態決定呼叫哪個函式實作。

```cpp
class Animal {
public:
    virtual void speak() const {
        // 讓子類別覆寫
    }
    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    void speak() const override {
        // Dog 的實作
    }
};

class Cat : public Animal {
public:
    void speak() const override {
        // Cat 的實作
    }
};

void makeSound(const Animal& a) {
    a.speak();  // 執行期才決定呼叫 Dog::speak 或 Cat::speak
}

int main() {
    Dog d;
    Cat c;
    makeSound(d);  // 呼叫 Dog::speak
    makeSound(c);  // 呼叫 Cat::speak
    return 0;
}
```

### 關鍵特性
- 屬於**動態綁定 (Dynamic Binding)**，又稱執行期多型
- 底層機制為**虛擬函式表 (vtable)**：含虛擬函式的 class 會有一張表，物件內部藏著指向該表的指標 `vptr`，呼叫虛擬函式時透過此指標查表決定實際執行的函式
- 因多一次查表動作，效能略有開銷（相較一般函式呼叫）
- 必須搭配繼承關係，且函式須宣告為 `virtual`

---

## 覆寫 (Overriding) 

**Overriding** 是實現執行期多型的**手段**：衍生類別重新定義基底類別中已宣告為 `virtual` 的函式，且**函式簽章必須完全相同**（參數列表、`const` 修飾皆須一致），否則會變成「隱藏 (hiding)」而非覆寫。

```cpp
class Base {
public:
    virtual void func(int x) const {}
};

class Derived : public Base {
public:
    void func(int x) const override {}  // 正確覆寫，簽章完全相同
    // void func(double x) const {}     // 這是多載/隱藏，不是覆寫！
};
```

- `override` 關鍵字（C++11 起）讓編譯器檢查簽章是否真的對應到基底類別的虛擬函式，避免打錯字或簽章不符卻沒發現
- Overriding 發生在**繼承鏈**上，是「同簽章、不同實作」；Overloading 發生在**同一作用域**，是「同名字、不同簽章」

---

## 核心差異對照表

| 項目 | 多載 (Overloading) | 覆寫 (Overriding) | 多型 (Polymorphism / 執行期) |
|---|---|---|---|
| 決定時機 | 編譯期 | 編譯期宣告，執行期呼叫 | 執行期 |
| 判斷依據 | 參數列表不同 | 衍生類別重新實作同簽章函式 | 物件的實際型別 |
| 是否需要繼承 | 不需要 | 需要 | 需要 |
| 是否需要 virtual | 不需要 | 需要 | 需要 |
| 函式簽章 | 必須不同 | 必須完全相同 | （依賴 Overriding） |
| 效能開銷 | 無 | 無（宣告本身無開銷） | 有（vtable 查表） |

---

## 一句話總結

- **多載**：同名字、參數不同，編譯器在編譯期就幫你挑好版本
- **覆寫**：衍生類別對基底類別的 `virtual` 函式重新實作，簽章完全相同
- **多型**：靠覆寫達成，物件實際型別在執行期才決定呼叫哪個實作
