---
title: Singleton（單例模式）
source: 
author: 
published: 
created: 2026-09-10
description: 確保類別在程式執行期間只有一個實例，並附上執行緒安全的實作方式與爭議
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - cpp
  - thread-safety
---

## 動機

確保一個類別在整個程式執行期間只有一個實例，並提供全域可存取的入口。常見用途：設定檔管理員、日誌系統、硬體資源管理。

## 標準實作：Meyers' Singleton（C++11 起執行緒安全）

```cpp
class ConfigManager {
public:
    static ConfigManager& instance() {
        static ConfigManager inst;   // local static 變數
        return inst;
    }
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
private:
    ConfigManager() : value_(0) {}
    int value_;
};
```

### 為什麼這是目前標準做法

1. local static 變數的初始化是 lazy 的，第一次執行到 `instance()` 才建構，不用時不佔空間，也避開「靜態初始化順序問題（Static Initialization Order Fiasco）」——多個全域/靜態物件因不同編譯單元初始化順序未定義而互相依賴出錯
2. C++11 標準保證 local static 初始化在多執行緒下安全（只有一個執行緒真的執行初始化，其他等待），不需手動加鎖
3. 建構子設為 private + 刪除複製建構子/指派運算子，封死從外部再造出第二個實例的路徑

## 常見錯誤寫法

```cpp
class BadSingleton {
public:
    static BadSingleton* instance() {
        if (ptr_ == nullptr) { ptr_ = new BadSingleton(); }  // 多執行緒下可能同時通過判斷
        return ptr_;
    }
private:
    static BadSingleton* ptr_;
};
```

兩個執行緒同時第一次呼叫時都可能通過 `ptr_ == nullptr` 檢查，導致 `new` 被呼叫兩次，產生兩個實例，這種「先檢查再動作」模式天生有 race condition。

## 爭議

- **全域狀態**：本質是包裝過的全域變數，任何地方都能直接存取修改，容易產生隱藏耦合
- **難以測試**：全域唯一實例難以替換成 mock 版本做單元測試
- **隱藏依賴**：函式內部呼叫 `instance()`，從函式簽章看不出依賴這個全域狀態，跟依賴注入（Dependency Injection）原則相衝突

實務建議：只是需要「全域可存取的單一資源」時，優先當成一般物件建立、透過參數（依賴注入）傳給需要的地方；只有明確需要限制建立次數且能接受上述缺點時才用 Singleton。

## 延伸閱讀

- [[design-patterns-overview]]
