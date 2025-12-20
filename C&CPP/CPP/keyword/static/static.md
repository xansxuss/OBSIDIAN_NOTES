C++ 裡 static 關鍵字的四種常見用法，附上範例與行為差異。

🧩 1️⃣ 函式內的 static 變數（最常見）

➡️ 只初始化一次、跨呼叫保留值。

``` cpp
void counter() {
    static int count = 0;  // 只初始化一次
    count++;
    std::cout << "count = " << count << std::endl;
}

int main() {
    counter(); // count = 1
    counter(); // count = 2
    counter(); // count = 3
}
```

🧠 概念：
count 存在於程式整體生命周期中，但作用域只在 counter() 內。

🧩 2️⃣ 類別內的 static 成員變數

➡️ 所有物件共享同一份資料。

``` cpp
class Foo {
public:
    static int shared; // 宣告
    void add() { shared++; }
};

int Foo::shared = 0; // 定義（要放在類別外）

int main() {
    Foo a, b;
    a.add();
    b.add();
    std::cout << Foo::shared << std::endl; // 2
}
``` 

🧠 概念：
Foo::shared 存在於全域記憶體中，不隨物件生命週期消失。

🧩 3️⃣ 類別內的 static 成員函式

➡️ 不需要物件就能呼叫，不能存取非靜態成員。

``` cpp
class Math {
public:
    static int add(int x, int y) {
        return x + y;
    }
};

int main() {
    std::cout << Math::add(2, 3) << std::endl; // 5
}
```

🧠 概念：
靜態函式像全域函式，但可被包在 class 名稱空間裡。

🧩 4️⃣ 檔案層級的 static 函式或變數

➡️ 限制作用範圍只在該 .cpp 檔內（內部連結）。

``` cpp
// foo.cpp
static void helper() {
    std::cout << "internal only" << std::endl;
}

// main.cpp
extern void helper();  // ❌ 無法連結，因為 helper() 是 static 的
```

🧠 概念：
這種用法主要是封裝內部實作、避免與其他翻譯單元名稱衝突。
C++17 之後也常用 anonymous namespace {} 取代。

⚡️總整理表

| 用法類型    | 生命週期 | 可見範圍       | 是否共享 | 常見用途     |
| ------- | ---- | ---------- | ---- | -------- |
| 函式內靜態變數 | 整個程式 | 函式內        | 否    | 累加計數、緩存  |
| 類別靜態變數  | 整個程式 | 類別內（需外部定義） | 是    | 共用設定     |
| 類別靜態函式  | 整個程式 | 類別內        | 是    | 工具函式     |
| 檔案內靜態成員 | 整個程式 | 該檔案        | 否    | 隱藏內部 API |
