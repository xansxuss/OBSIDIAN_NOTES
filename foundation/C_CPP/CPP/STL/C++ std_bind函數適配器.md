0. 初識

C++11 提供了一個名為 std::bind 的函式模板，用於生成可呼叫物件的轉發呼叫包裝器，相當於是一種通用的「函式適配器」（舊版的 bind1st / bind2nd 已被棄用）。
std::bind 可以適配任意可呼叫物件，包括：

- 函式指標
- 函式參考
- 成員函式指標
- 函式物件（function object）

它接受一個可呼叫物件，生成一個新的可呼叫物件，用來適配原本的參數列表。

bind 會將可呼叫物件與其部分參數綁定起來。綁定後的結果可以使用 std::function 儲存。
綁定完成後，bind 會回傳一個函式物件，其內部保存了原可呼叫物件的拷貝，並擁有 operator()。
回傳值型別會自動推導為原可呼叫物件的回傳型別。當呼叫時，該函式物件會將之前綁定的參數轉發給原函式完成呼叫。

1. 綁定普通函式

一般形式如下：

``` cpp
auto newCallable = std::bind(callable, arg_list);
```

第一個參數是要綁定的可呼叫物件，後面是參數列表（與原函式的參數對應）。
參數列表中可以包含形如 _n（n 為整數）的佔位符，表示生成的可呼叫物件中對應參數的位置。
佔位符位於 std::placeholders 命名空間中。

``` cpp
#include <iostream>
#include <functional>

// 相加
int add(int a, int b)
{
    std::cout << "a:" << a << "\tb:" << b << std::endl;
    return a + b;
}

void test_bind()
{
    // 第一個參數是函式名，普通函式作為實參時會被隱式轉換成函式指標，相當於 &add
    // std::placeholders::_1 佔位符表示保留第一個參數
    auto func_add_10 = std::bind(add, std::placeholders::_1, 10);
    int result = func_add_10(100); // 等於 100 + 10
    std::cout << "func_add_10(100):" << result << std::endl;

    // 對參數重新排序
    using namespace std::placeholders;
    auto func_add = std::bind(add, _2, _1); // 交換參數位置
    result = func_add(100, 10);
    std::cout << "func_add(100,10):" << result << std::endl;
}
```

輸出結果：
![[C++ std_bind函數適配器_1.png]]

2. 綁定引用參數

bind 會複製其參數。如果希望傳遞的是引用而非拷貝，就必須使用標準庫中的 ref 函式。
ref 會回傳一個可被拷貝的包裝物件，內部保存原變數的引用。
標準庫中還有 cref 函式，會回傳一個保存 const 引用的包裝物件。

``` cpp
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <sstream>
using namespace std::placeholders;
using namespace std;

ostream& print(ostream& os, const string& s, char c)
{
    os << s << c;
    return os;
}

int main()
{
    vector<string> words{ "hello", "world", "this", "is", "C++11" };
    ostringstream os;
    char c = ' ';
    for_each(words.begin(), words.end(),
             [&os, c](const string& s) { os << s << c; });
    cout << os.str() << endl;

    ostringstream os1;
    // ostream 無法拷貝，若希望傳給 bind 一個物件但不拷貝，就需使用 ref()
    for_each(words.begin(), words.end(),
             bind(print, ref(os1), _1, c));
    cout << os1.str() << endl;

    system("pause");
    return 0;
}
```

3. 綁定類別成員函式

綁定類別成員函式時，第一個參數是「成員函式指標」，第二個參數是「物件的位址」。

``` cpp
struct Foo {
    void print_sum(int n1, int n2)
    {
        std::cout << n1 + n2 << '\n';
    }
    int data = 10;
};

int main() 
{
    Foo foo;
    auto f = std::bind(&Foo::print_sum, &foo, 95, std::placeholders::_1);
    f(5); // 輸出 100
}
```

注意：
    - 必須明確指定 &Foo::print_sum，因為編譯器不會自動將成員函式轉為普通函式指標。
    - 使用成員函式指標時，必須知道該函式屬於哪個物件，因此第二個參數必須是該物件的位址（&foo）。

4. 其他示例

以下示例來自線上手冊：

``` cpp
#include <random>
#include <iostream>
#include <memory>
#include <functional>

void f(int n1, int n2, int n3, const int& n4, int n5)
{
    std::cout << n1 << ' ' << n2 << ' ' << n3 << ' ' << n4 << ' ' << n5 << '\n';
}

int g(int n1)
{
    return n1;
}

struct Foo {
    void print_sum(int n1, int n2)
    {
        std::cout << n1 + n2 << '\n';
    }
    int data = 10;
};

int main()
{
    using namespace std::placeholders;  // 使用 _1, _2, _3...

    // 演示參數重新排序與引用傳遞
    int n = 7;
    // _1 與 _2 來自 std::placeholders，代表未來傳入 f1 的參數
    auto f1 = std::bind(f, _2, 42, _1, std::cref(n), n);
    n = 10;
    f1(1, 2, 1001); // 對應呼叫 f(2, 42, 1, n, 7)

    // 巢狀 bind 表達式可共用佔位符
    auto f2 = std::bind(f, _3, std::bind(g, _3), _3, 4, 5);
    f2(10, 11, 12); // 對應呼叫 f(12, g(12), 12, 4, 5)

    // 常見應用：綁定亂數分布與亂數引擎
    std::default_random_engine e;
    std::uniform_int_distribution<> d(0, 10);
    std::function<int()> rnd = std::bind(d, e); // 儲存 e 的副本
    for (int n = 0; n < 10; ++n)
        std::cout << rnd() << ' ';
    std::cout << '\n';

    // 綁定成員函式指標
    Foo foo;
    auto f3 = std::bind(&Foo::print_sum, &foo, 95, _1);
    f3(5);

    // 綁定資料成員指標
    auto f4 = std::bind(&Foo::data, _1);
    std::cout << f4(foo) << '\n';

    // 智慧指標同樣可用於呼叫被引用物件的成員
    std::cout << f4(std::make_shared<Foo>(foo)) << '\n'
              << f4(std::make_unique<Foo>(foo)) << '\n';

    system("pause");
    return 0;
}
```

輸出結果：
![[C++ std_bind函數適配器_2.png]]

5. 參考資料

- 參考書籍：《C++ Primer》（中文版第五版）
- [參考文件](https://zh.cppreference.com/w/cpp/utility/functional/bind)
- [參考部落格](https://www.jianshu.com/p/f191e88dcc80)
- [參考部落格](https://www.cnblogs.com/sick-vld/p/10769187.html)