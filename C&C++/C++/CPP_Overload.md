## C++ 函式多載（Function Overloading） 的核心概念、編譯器行為、陷阱與進階應用

🧠 一、什麼是函式多載（Function Overloading）

在 C++ 裡，「多載（Overload）」指的是同一個作用域內允許有相同名稱但不同參數列表的函式。
換句話說，C++ 編譯器可以根據參數型別、個數、順序去辨識你要呼叫哪個版本。

``` cpp
void print(int x);
void print(double x);
void print(const std::string &s);
```

這三個函式都叫 print，但編譯器能根據傳入的參數決定要呼叫哪個。

🧩 二、編譯器判斷依據：參數列表（Parameter List）

C++ 編譯器會根據下列要素來區分多載函式：

<span style="color:#28FF28">可作為區別的項目:</sapn>

| 項目 | 說明                           |
| -------- | ---------------------------- |
| 參數個數     | 函式引數數量不同即可區分                 |
| 參數型別     | int vs double vs string      |
| 參數順序     | (int, float) vs (float, int) |
| const 修飾 | const 與非 const 可區分           |
| 引用類型     | `int&` vs `const int&`       |
| 模板型別推導   | 若涉及 template，會做額外解析          |


<span style="color:#FF0000">不可以用來區分的項目:</span>

| 項目    | 說明             |
| ----- | -------------- |
| 回傳型別  | 僅靠回傳值不同無法區分    |
| 參數名稱  | 名稱只是語法糖，對簽名沒影響 |
| 預設參數值 | 預設參數不會改變「函式簽名」 |

``` cpp
int func(int x);
double func(int x); // ❌ 錯誤：只有回傳型別不同，簽名相同
```

🧾 三、實際範例與解析
✅ 合法多載

``` cpp
void connect(const char* host, int port = 1883, int keepalive = 60);
void connect(const char* host, int port, int keepalive, const char* bind_address);
```

- 雖然第一個有預設值，但簽名不同（參數個數不一樣），可以共存。
- 若呼叫：
    ``` cpp
    connect("broker.hivemq.com");             // 呼叫第1個
    connect("broker.hivemq.com", 1883, 60, "192.168.1.100");  // 呼叫第2個
    ```

⚠️ 模稜兩可（ambiguous call）

``` cpp
void show(int a, double b);
void show(double a, int b);

show(10, 20); // ❌ 編譯錯誤：模稜兩可，int 可轉 double，double 可轉 int
```

🧩 const 修飾差異

``` cpp
void process(int& x);
void process(const int& x);

int a = 5;
process(a);   // 呼叫非 const 版本
process(10);  // 呼叫 const 版本（暫時物件無法綁定非 const 引用）
```

🧠 四、函式簽名（Function Signature）

函式簽名（signature） = 函式名稱 + 參數型別（不含回傳型別）
例如：

``` cpp
void f(int);
void f(double);
```

對編譯器來說：

``` bash
f__int
f__double
```

會被「改名」（name mangling），所以能共存。

🧰 五、C++ 與 C 的差別

C 語言不支援函式多載，因為它沒有「name mangling」。

``` c
// C
int func(int x);
double func(double x); // ❌ 錯誤，同名衝突
```

但在 C++ 中：

``` cpp
// C++
int func(int x);       // mangled 為 _Z4funci
double func(double x); // mangled 為 _Z4funcd
```

🧨 六、陷阱：預設參數 + 多載 = 爆炸組合

當多載同時使用「預設參數」時，很容易造成「呼叫模糊」。

``` cpp
void foo(int a);
void foo(int a, int b = 10);

foo(5); // ❌ 模稜兩可，哪個都有可能被匹配
```

解法：

移除預設參數或

改成不同型別的參數以避免衝突

🚀 七、進階：模板與多載

函式模板也可以與一般函式多載共存，編譯器會依以下優先順序選擇：

1. 完全匹配的一般函式（最高優先）
2. 模板函式的實例化版本

``` cpp
void print(int x) { std::cout << "普通 int\n"; }
template<typename T>
void print(T x) { std::cout << "模板版本\n"; }

print(42);    // 呼叫普通函式
print(3.14);  // 呼叫模板版本
```

💡 八、真實應用場景

在實務中，多載常見於：

- API 統一化：同名函式處理不同輸入類型
- 物件行為多型：配合 operator overloading
- 模板混合設計：像是 std::to_string()、std::swap()

例如：

``` cpp
void log(const std::string &msg);
void log(int code);
void log(double value, const std::string &tag);
```

呼叫方不用記住多個名稱，編譯器會自動分派。

🧩 九、Operator Overloading（運算子多載）

原理與函式多載相同，只是語法糖。

``` cpp
class Vec2 {
public:
    float x, y;
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& rhs) const {
        return Vec2(x + rhs.x, y + rhs.y);
    }
};
```

🧮 十、簡單實作與觀察 name mangling

可以用：

``` bash
g++ test.cpp -c && nm test.o | c++filt
```

看出各版本被「改名」的結果：

``` bash
0000000000000000 T print(int)
0000000000000010 T print(double)
0000000000000020 T print(std::string const&)
```

🧩 小結

| 主題   | 重點                 |
| ---- | ------------------ |
| 多載原則 | 名稱相同、參數列表不同        |
| 不可依據 | 回傳型別、參數名稱、預設參數     |
| 模板優先 | 普通函式 > 模板函式        |
| 常見陷阱 | 預設參數與多載衝突、型別轉換模糊   |
| 實用場景 | API 統一、算子多載、型別彈性接口 |
