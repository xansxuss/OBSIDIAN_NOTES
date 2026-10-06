---
title: C++ 封裝（Encapsulation）
source: 
author: 
published: 
created: 2026-08-28
description: 說明 C++ 封裝的核心概念、資料隱藏與公開介面設計，並以 BankAccount 類別為例示範。
categorization: programming_language/C_CPP/CPP/class
tags:
  - cpp
  - oop
  - encapsulation
  - class-design
---

## 核心概念

封裝是物件導向程式設計的核心概念之一，指的是把資料（成員變數）與操作資料的方法（成員函式）包裝在同一個類別裡，並限制外部直接存取內部資料，只能透過類別提供的公開介面來操作。

1. **資料隱藏**：把成員變數設為 `private`，外部無法直接讀寫。
2. **公開介面**：透過 `public` 的成員函式（getter/setter 或其他行為函式）讓外部間接存取或操作資料。
3. **不變條件（invariant）維護**：因為外部無法繞過類別直接改資料，類別可以在設定資料時做檢查，確保物件內部狀態永遠合法。

## 範例程式碼

```cpp
class BankAccount {
private:
    // 私有成員變數，外部無法直接存取
    long balance;   // 單位：分（避免浮點數誤差）
    int accountId;

public:
    // 建構子：初始化時就檢查資料合法性
    BankAccount(int id, long initBalance) : accountId(id) {
        balance = (initBalance >= 0) ? initBalance : 0;
    }

    // 公開介面：讀取餘額（getter），const 表示不會修改物件狀態
    long getBalance() const {
        return balance;
    }

    // 公開介面：存款，只允許存入正數
    void deposit(long amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    // 公開介面：提款，檢查餘額是否足夠，避免變成負數
    bool withdraw(long amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
};
```

### 邏輯說明

- `balance` 設為 `private`，外部程式碼不可能直接寫出破壞資料合法性的操作（如強制設為負數）。
- 所有對 `balance` 的修改都必須經過 `deposit()` 或 `withdraw()`，函式內部有邏輯把關，確保物件狀態永遠一致。
- `getBalance()` 是唯一的讀取管道，且標記為 `const`，語意上告訴呼叫者這個操作不會改變帳戶狀態。

## 為什麼這樣做比較好

若把 `balance` 設成 `public`，外部可任意賦值，繞過所有邏輯檢查，物件容易進入不合法狀態（例如餘額變負數）。封裝把「如何維持資料合法」的責任集中在類別內部，外部只需呼叫函式、不需知道實作細節，降低模組間耦合度；未來要修改內部實作（例如把 `balance` 改用其他資料結構儲存），只要公開介面行為不變，外部程式碼完全不用修改。