---
title: Memento（備忘錄模式）
source: 
author: 
published: 
created: 2026-09-10
description: 在不破壞封裝性的前提下儲存並還原物件的內部狀態，支援 undo 功能
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - undo-redo
  - encapsulation
---

## 動機

需要儲存物件內部狀態以便還原（undo），但不想破壞封裝性（不想把私有成員整個公開給外部）。讓物件自己產生一個「備忘錄」物件存進當下狀態，備忘錄對外是不透明黑盒子，只有原物件自己看得懂內容，其他人只能保管與還原，不能窺探或修改。

## 基本結構

三個角色：Originator（狀態擁有者）、Memento（備忘錄）、Caretaker（負責保管但不能看內容）。

```cpp
class TextMemento {
public:
    friend class TextEditor;   // 只允許 TextEditor 存取內部內容
private:
    TextMemento(const char* content) : content_(content) {}
    const char* content_;
};

class TextEditor {
public:
    void setContent(const char* content) { content_ = content; }
    TextMemento save() { return TextMemento(content_); }
    void restore(const TextMemento& memento) { content_ = memento.content_; }
private:
    const char* content_ = "";
};

class History {
public:
    void push(const TextMemento& m) { mementos_[count_++] = m; }
    TextMemento pop() { return mementos_[--count_]; }
private:
    TextMemento mementos_[32] = {TextMemento("")};
    int count_ = 0;
};
```

## 運作邏輯重點

- `History`（Caretaker）持有一堆 `TextMemento`，但因為 `TextMemento` 的建構子與成員都是 `private`，只對 `TextEditor` 開了 `friend`，所以 `History` 只能保管與傳遞，無法讀取或竄改內容——這是在「支援 undo」與「不破壞封裝」之間取得平衡的方式
- 只有 `TextEditor`（Originator）知道要存哪些資料進 `TextMemento`、以及怎麼還原，這個知識不外洩給 `Caretaker` 或呼叫端

## 實務取捨

若狀態很大（例如整份文件），每次 `save()` 完整複製會很耗記憶體，實務上常見最佳化是只存差異（diff）而非完整快照，但會讓 `restore` 邏輯變複雜，需依情境權衡。

## 延伸閱讀

- [[command]]：另一種支援 undo 的方式，但著重「操作」本身而非「狀態快照」
- [[design-patterns-overview]]
