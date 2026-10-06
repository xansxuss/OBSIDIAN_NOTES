---
title: Command（命令模式）
source: 
author: 
published: 
created: 2026-09-10
description: 把一次請求包裝成物件，讓觸發者與執行者解耦，支援延遲執行、排隊與 undo/redo
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
  - undo-redo
---

## 動機

把一個請求本身包裝成物件，讓你能用參數傳遞請求、把請求排入佇列、記錄請求（做 undo/redo），或徹底解耦「觸發者」跟「執行者」——按鈕不需要知道按下去要做什麼事，只需持有一個 Command 物件並呼叫 `execute()`。

## 基本結構

角色：抽象 Command（`execute()` / `undo()`）、接收者（實際知道怎麼做事的物件）、具體 Command（把一次操作包成物件）、呼叫者（持有 Command，但不知內部做什麼）。

```cpp
class Command { public: virtual void execute() = 0; virtual void undo() = 0; virtual ~Command() {} };

class TextDocument {
public:
    void insertText(const char* text, int pos);
    void deleteText(int pos, int length);
};

class InsertTextCommand : public Command {
public:
    InsertTextCommand(TextDocument* doc, const char* text, int pos)
        : doc_(doc), text_(text), pos_(pos) {}
    void execute() override { doc_->insertText(text_, pos_); }
    void undo() override { doc_->deleteText(pos_, length(text_)); }
private:
    TextDocument* doc_; const char* text_; int pos_;
};

class CommandHistory {
public:
    void executeCommand(Command* cmd) { cmd->execute(); history_[historyCount_++] = cmd; }
    void undoLast() { history_[--historyCount_]->undo(); }
private:
    Command* history_[32]; int historyCount_ = 0;
};
```

## 運作邏輯重點

- 呼叫者只認得 `Command` 抽象介面，不需知道具體邏輯，只負責儲存並在需要時呼叫 `execute()`/`undo()`
- 每個 Command 把「要做什麼」跟「對誰做」都封裝在自己內部，因此可以被到處傳遞、暫存、排入佇列，執行時機被延後且可被記錄
- Undo 機制可行是因為每個 Command 除了知道怎麼「做」，也知道怎麼「復原」，這個對稱性需要在設計時一併考慮

## 與 Strategy 的差異

寫法都是把行為包成物件，但 Strategy 關心「同一件事有多種做法，選一種」，通常不記錄歷史；Command 關心「一次操作」本身，重點在能被儲存、排隊、復原。

## 延伸閱讀

- [[strategy]]
- [[chain-of-responsibility]]：常搭配 Command，把請求包裝成 Command 在鏈上傳遞
- [[design-patterns-overview]]
