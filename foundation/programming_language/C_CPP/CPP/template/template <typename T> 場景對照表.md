從最基本的 function/class 到進階的 operator overload、SFINAE、C++20 concept，一路排開，一眼就知道各種寫法怎麼用。
#### template <typename T> 場景對照表

| 場景 | 範例程式碼 | 說明 | 編譯失敗案例 (Fail Example)  |
| ---- | ---- | ---- | ---- |
| **函式模板 (Function Template)** | <pre><code class="language-cpp">template <typename T><br>T add(T a, T b) {<br>    return a + b;<br>}<br><br>int main() {<br>    auto x = add(3, 5);       // T=int<br>    auto y = add(2.5, 3.1);   // T=double<br>}</code></pre> | 泛型函式，型別由編譯器推導 | `add(3, "hi"); // 編譯失敗：int + const char*` |
| **類別模板 (Class Template)** | <pre><code class="language-cpp">template <typename T><br>class Box {<br>    T value;<br>public:<br>    Box(T v) : value(v) {}<br>    T get() const { return value; }<br>};<br><br>int main() {<br>    Box<int> b1(42);<br>    Box<std::string> b2("hi");<br>}</code></pre> | 泛型類別，型別在實例化時指定 | `Box<void> b3(nullptr); // 編譯失敗：無法儲存 void` |
| **函式模板 + 多參數** | <pre><code class="language-cpp">template <typename T, typename U><br>auto mul(T a, U b) {<br>    return a * b;<br>}<br><br>int main() {<br>    auto r = mul(3, 2.5); // T=int, U=double → 回傳 double<br>}</code></pre> | 可以同時處理多種型別 | `mul(3, "hi"); // 編譯失敗：int * const char*` |
| **Operator Overloading (泛型運算子重載)** | <pre><code class="language-cpp">template <typename T><br>class Vec2 {<br>public:<br>    T x, y;<br>    Vec2(T x, T y) : x(x), y(y) {}<br><br>    Vec2 operator+(const Vec2& other) const {<br>        return Vec2(x + other.x, y + other.y);<br>    }<br>};<br><br>int main() {<br>    Vec2<int> v1(1,2), v2(3,4);<br>    auto v3 = v1 + v2; // OK<br>}</code></pre> | 可以讓不同型別容器支援運算子 | `Vec2&lt;std::string&gt; v1("a","b"), v2("c","d"); auto v3 = v1 + v2; // 編譯失敗` |
| **SFINAE (Substitution Failure Is Not An Error)** | <pre><code class="language-cpp">#include <type_traits><br><br>template <typename T><br>std::enable_if_t<std::is_arithmetic<T>::value, T><br>safe_add(T a, T b) {<br>    return a + b;<br>}<br><br>int main() {<br>    auto x = safe_add(3, 5);   // OK<br>}</code></pre> | 利用 `std::enable_if` 過濾可用型別 | `safe_add("a", "b"); // 編譯失敗：const char* 不是 arithmetic` |
| **C++20 Concepts (比 SFINAE 更優雅)** | <pre><code class="language-cpp">#include <concepts><br><br>template <std::integral T><br>T sum(T a, T b) {<br>    return a + b;<br>}<br><br>int main() {<br>    auto x = sum(3, 4);   // OK<br>}</code></pre> | 使用 concept 限定型別，更清楚直觀 | `sum(3.5, 2.1); // 編譯失敗：double 不是 integral` |
| **C++20 Concepts + requires** | <pre><code class="language-cpp">#include <concepts><br><br>template <typename T><br>requires std::floating_point<T><br>T div(T a, T b) {<br>    return a / b;<br>}<br><br>int main() {<br>    auto x = div(3.0, 1.5); // OK<br>}</code></pre> | `requires` 子句更細膩控制型別約束 | `div(3, 2); // 編譯失敗：int 非 floating_point` |
| **模板特化 (Template Specialization)** | <pre><code class="language-cpp">template <typename T><br>struct Printer {<br>    void print(T v) { std::cout << v << "\n"; }<br>};<br><br>template <><br>struct Printer<bool> {<br>    void print(bool v) { std::cout << (v ? "true" : "false") << "\n"; }<br>};<br><br>int main() {<br>    Printer<int>().print(42);<br>    Printer<bool>().print(true);<br>}</code></pre> | 特殊化某個型別行為 | `Printer<void>().print(nullptr); // 編譯失敗` |
| **C++17 if constexpr (搭配 template)** | <pre><code class="language-cpp">template <typename T><br>void printType(T x) {<br>    if constexpr (std::is_integral<T>::value)<br>        std::cout << "int-like: " << x << "\n";<br>    else<br>        std::cout << "other: " << x << "\n";<br>}<br><br>int main() {<br>    printType(42);     // int-like<br>    printType(3.14);   // other<br>}</code></pre> | 利用 `if constexpr` 做編譯期分支 | 幾乎不會編譯失敗，因為 `if constexpr` 會在編譯期裁剪掉無效分支 |


TL;DR 🎯

- 函式模板：讓一個函式能吃不同型別。
- 類別模板：讓一個 class 能存放不同型別。
- operator overload：模板類別可以支援泛型運算子。
- SFINAE：舊時代的「型別限制黑魔法」，閱讀難度高。
- Concepts (C++20)：SFINAE 的進化版，語法更直觀。
- 特化 (Specialization)：針對特殊型別改寫模板行為。
- if constexpr：讓模板能做「型別判斷分支」。