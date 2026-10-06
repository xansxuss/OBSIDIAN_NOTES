---
title: 設計模式總覽與比較
source: 
author: 
published: 
created: 2026-09-10
description: GoF 建立型與行為型設計模式的核心問題總覽、彼此差異與選用時機比較
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - creational-pattern
  - behavioral-pattern
  - cpp
  - overview
---

## 建立型模式（Creational Patterns）

| 模式 | 核心問題 |
|---|---|
| [[static-factory-method]] | 用具名靜態方法取代建構子，表達建立方式的語意 |
| [[factory-method]] | 執行期決定實際建立哪個具體型別（多型） |
| [[abstract-factory]] | 建立一整族相關聯的產品，確保同家族產品搭配一致 |
| [[builder]] | 把複雜物件的建構過程拆成一系列具名步驟 |
| [[singleton]] | 確保類別在程式執行期間只有一個實例 |

### 建立型模式關係

- [[static-factory-method]] 是 [[factory-method]] 的基礎，差別在於後者強調「回傳基底類別指標、執行期決定具體子類別」
- [[abstract-factory]] 本質上是把多個 [[factory-method]] 打包成一個介面，管理一整族產品
- [[builder]] 關心「怎麼一步步組裝」，跟 Factory 系列關心「建立哪一種」是不同維度的問題，兩者可以並存

## 行為型模式（Behavioral Patterns）

| 模式 | 核心問題 |
|---|---|
| [[observer]] | 狀態改變時，如何通知多個相依物件 |
| [[strategy]] | 同一件事有多種做法，如何在執行期抽換 |
| [[command]] | 如何把一次請求包裝成物件，支援延遲執行、記錄、undo |
| [[state]] | 行為隨內部狀態改變，如何避免大量 if/switch |
| [[template-method]] | 多個子類別共享同一套流程骨架，只有部分步驟不同 |
| [[chain-of-responsibility]] | 請求可能被多個處理者之一處理，如何解耦發送者與處理者 |
| [[mediator]] | 多個物件互相溝通，如何避免網狀依賴 |
| [[memento]] | 如何在不破壞封裝的前提下保存/還原物件狀態 |
| [[visitor]] | 類別階層穩定、但要不斷新增新操作，如何避免修改既有類別 |
| [[iterator]] | 如何走訪容器內容，而不暴露內部資料結構 |

## 常被混淆的模式對照

| 對照組 | 關鍵差異 |
|---|---|
| [[strategy]] vs [[state]] | Strategy 由呼叫端主動選擇策略，各實作間通常無順序關係；State 由狀態物件自己決定下一個狀態，形成狀態機 |
| [[strategy]] vs [[template-method]] | Strategy 用組合（has-a）整包抽換演算法；Template Method 用繼承（is-a）固定流程順序，只開放部分步驟 |
| [[command]] vs [[strategy]] | 寫法都是把行為包成物件；Command 關心「一次操作」能否被記錄/復原，Strategy 關心「選哪種做法」 |
| [[factory-method]] vs [[abstract-factory]] | 前者建立一種產品的不同變化；後者建立多種產品組成的一整個家族，保證彼此搭配一致 |
| [[mediator]] vs [[observer]] | Mediator 常用 Observer 機制實作通知，但 Mediator 關心「降低多物件間耦合」，Observer 關心「一對多通知」本身 |
| [[command]] vs [[chain-of-responsibility]] | 兩者常搭配：把請求包成 Command 物件，在責任鏈上傳遞，取得更完整的請求上下文 |
| [[memento]] vs [[command]] | 兩者都能支援 undo；Memento 靠「狀態快照」還原，Command 靠「記錄操作並反向執行」還原 |

## 選用時機速查

- 建立物件語意不清、參數型別重複 → [[static-factory-method]] / [[factory-method]]
- 需要保證一整組物件互相搭配 → [[abstract-factory]]
- 物件建構步驟多、可選欄位多 → [[builder]]
- 全域唯一資源（謹慎使用） → [[singleton]]
- 一個狀態改變要通知多方 → [[observer]]
- 演算法要能動態抽換 → [[strategy]]
- 操作要能排隊、記錄、復原 → [[command]]
- 行為隨狀態轉換 → [[state]]
- 多個子類別共用固定流程、少數步驟不同 → [[template-method]]
- 請求要依序嘗試多個處理者 → [[chain-of-responsibility]]
- 多個物件互相溝通產生網狀耦合 → [[mediator]]
- 需要保存/還原狀態但不破壞封裝 → [[memento]]
- 類別階層穩定但操作常變動 → [[visitor]]
- 走訪容器但不暴露內部結構 → [[iterator]]

## 延伸閱讀

- [[static-factory-method]]
- [[factory-method]]
- [[abstract-factory]]
- [[builder]]
- [[singleton]]
- [[observer]]
- [[strategy]]
- [[command]]
- [[state]]
- [[template-method]]
- [[chain-of-responsibility]]
- [[mediator]]
- [[memento]]
- [[visitor]]
- [[iterator]]
