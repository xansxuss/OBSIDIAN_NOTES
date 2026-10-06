---
title: Iterator（迭代器模式）
source: 
author: 
published: 
created: 2026-09-10
description: 提供統一介面循序存取容器元素，不暴露容器內部資料結構
categorization: foundation/software/design_pattern
tags:
  - design-pattern
  - behavioral-pattern
  - cpp
---

## 動機

提供一種方式循序存取聚合物件（容器）內的元素，而不暴露該聚合物件的內部資料結構（陣列、鏈結串列或樹）。呼叫端只透過統一介面（`hasNext()`/`next()`）走訪，不需知道內部怎麼存。

## 基本結構

```cpp
class Iterator { public: virtual bool hasNext() = 0; virtual int next() = 0; virtual ~Iterator() {} };

class IntArray {
public:
    IntArray(int capacity) : capacity_(capacity), size_(0) { data_ = new int[capacity]; }
    ~IntArray() { delete[] data_; }
    void add(int value) { if (size_ < capacity_) data_[size_++] = value; }

    class ArrayIterator : public Iterator {
    public:
        ArrayIterator(const IntArray& arr) : arr_(arr), index_(0) {}
        bool hasNext() override { return index_ < arr_.size_; }
        int next() override { return arr_.data_[index_++]; }
    private:
        const IntArray& arr_;
        int index_;
    };

    ArrayIterator createIterator() const { return ArrayIterator(*this); }
private:
    int* data_; int capacity_; int size_;
};
```

## 運作邏輯重點

- 呼叫端只用 `hasNext()`/`next()` 走訪，不需知道 `IntArray` 內部是陣列——若之後把內部實作換成鏈結串列，只要迭代器內部邏輯跟著調整，呼叫端走訪程式碼一行都不用改
- 迭代器把「目前走到哪」（`index_`）獨立保存在自己身上，而非存在容器本身，因此同一容器可同時存在多個獨立迭代器，各自走訪不同位置互不干擾

## 與現代 C++ 的關係

標準函式庫的 `begin()`/`end()` 加上 `operator++`、`operator*`、`operator!=` 這套慣例本質上就是 Iterator 模式的語言層級實現（範圍 for 迴圈背後靠這套介面運作）。本篇因不使用標準函式庫，改用最原始的 `hasNext()`/`next()` 介面示範，邏輯上完全等價。

## 延伸閱讀

- [[design-patterns-overview]]
