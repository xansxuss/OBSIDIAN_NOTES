「static 函式」這個概念在 C++ 有兩種完全不同層級的意思

## 🧱 一、檔案層級的 static 函式（file-scope static）
### 🎯 定義

放在 .cpp 檔案中，用來限制該函式只在這個檔案可見。

``` cpp
// foo.cpp
#include <iostream>

static void internalHelper() {  // 只在本檔可見
    std::cout << "internal function\n";
}

void publicAPI() {
    internalHelper();  // ✅ 可以用
}

// bar.cpp
extern void internalHelper();  // ❌ 錯誤！找不到 symbol
```

### 🧠 用途：

封裝內部實作，避免外部檔案誤用。

與 anonymous namespace (namespace { ... }) 效果幾乎相同，但較古老。

編譯後該函式的 symbol 不會出現在外部連結表中。

## 🧩 二、類別內的 static 成員函式（class static function）
### 🎯 定義

屬於整個類別，而不是某個物件的函式。

``` cpp
#include <iostream>

class Counter {
public:
    static int count;           // 所有物件共用一份
    static void add() {         // 靜態成員函式
        count++;
        std::cout << "count = " << count << std::endl;
    }
};

int Counter::count = 0;  // 類別外初始化

int main() {
    Counter::add();  // ✅ 不需要物件
    Counter c;
    c.add();         // ✅ 也可以透過物件呼叫
}
```

### 🧠 重點：

靜態函式 不能存取非靜態成員（因為沒有 this 指標）。

用途：

提供工具功能（像是 Math::sqrt()）。

操作所有物件共用的狀態（像上例的 count）。

可以作為 callback 函式（C-style API 會要求普通函式指標）。

### ⚡ 兩者比較

| 類型               | 可見範圍       | 是否屬於物件 | 是否能訪問 this | 常見用途      |
| ---------------- | ---------- | ------ | ---------- | --------- |
| 檔案內 `static` 函式  | 該 `.cpp` 檔 | 否      | 否          | 封裝內部實作    |
| 類別 `static` 成員函式 | 全域可見（需類別名） | 否      | 否          | 工具函式、共用邏輯 |
