### 1. Template 基礎
1. 函式模板（Function Template）
模板允許我們用同一份程式碼處理不同型別資料。

``` cpp
#include <iostream>
using namespace std;

// 函式模板
template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    cout << add(3, 5) << endl;       // int
    cout << add(3.5, 2.1) << endl;   // double
}
```

- `template <typename T>`：告訴編譯器這是模板，`T` 是型別參數。  
- 型別參數 `T` 也可以寫成 `class`：`template <class T>`，兩者完全等價。

``` cpp
// 使用 typename
template <typename T>
void func(T arg) {
    // ...
}

// 使用 class（等價寫法）
template <class T>
void func(T arg) {
    // ...
}
```

2. 類別模板（Class Template）
類別模板可以用於資料結構，例如通用的 stack、vector。

``` cpp
#include <iostream>
using namespace std;

template <typename T>
class MyStack {
private:
    T data[100];
    int topIndex = -1;
public:
    void push(T value) { data[++topIndex] = value; }
    T pop() { return data[topIndex--]; }
    bool empty() { return topIndex == -1; }
};

int main() {
    MyStack<int> intStack;
    intStack.push(10);
    cout << intStack.pop() << endl;

    MyStack<string> strStack;
    strStack.push("Hello");
    cout << strStack.pop() << endl;
}
```

3. 模板與非型別參數
模板參數不只能是型別，也可以是常數。

``` cpp
template <typename T, int N>
class Array {
private:
    T data[N];
public:
    T& operator[](int i) { return data[i]; }
};

int main() {
    Array<int, 5> arr;
    arr[0] = 42;
    cout << arr[0] << endl;
}
```

### 2. Template 進階
1. 多型別模板

``` cpp
template <typename T1, typename T2>
void printPair(T1 a, T2 b) {
    cout << a << " - " << b << endl;
}

int main() {
    printPair(1, "apple");
    printPair(3.14, 42);
}
```

2. 模板特化（Template Specialization）
   1. 完全特化（Full Specialization）

    ``` cpp
    template <typename T>
    struct TypeName { static void print() { cout << "Unknown" << endl; } };

    template <>
    struct TypeName<int> { static void print() { cout << "int" << endl; } };

    int main() {
        TypeName<int>::print();    // int
        TypeName<double>::print(); // Unknown
    }
    ```

   2. 偏特化（Partial Specialization）
   
   ``` cpp
   template <typename T, typename U>
    struct Pair { static void print() { cout << "General Pair" << endl; } };

    template <typename T>
    struct Pair<T, T> { static void print() { cout << "Same Type Pair" << endl; } };

    int main() {
        Pair<int, double>::print(); // General Pair
        Pair<int, int>::print();    // Same Type Pair
    }
    ```

    3. 模板型別推導（Template Type Deduction）
    **C++11** 開始可利用 auto 或 decltype 做型別推導：

    ``` cpp
    template <typename T>
    void printType(T value) {
        cout << value << endl;
    }

    int main() {
        printType(3);     // T=int
        printType(3.14);  // T=double
    }
    ```
    
    4. 非型別模板參數的進階用法
    
    ``` cpp
    template <int N>
    struct Factorial {
        static constexpr int value = N * Factorial<N-1>::value;
    };

    template <>
    struct Factorial<0> {
        static constexpr int value = 1;
    };

    int main() {
        cout << Factorial<5>::value << endl; // 120
    }
    ```

### 3. 特殊用法與技巧
1. SFINAE（Substitution Failure Is Not An Error）
可以依條件選擇函式模板。

``` cpp
#include <type_traits>
#include <iostream>
using namespace std;

template <typename T>
typename enable_if<is_integral<T>::value, bool>::type
isEven(T x) {
    return x % 2 == 0;
}

int main() {
    cout << isEven(4) << endl; // OK
    // cout << isEven(3.14);   // 編譯失敗，因為 3.14 不是整數
}
```

2. 可變參數模板（Variadic Templates）
支援任意數量的模板參數。

``` cpp
#include <iostream>
using namespace std;

template <typename T>
void printAll(T t) { cout << t << endl; }

template <typename T, typename... Args>
void printAll(T t, Args... args) {
    cout << t << ", ";
    printAll(args...);
}

int main() {
    printAll(1, 2.5, "Hello", 'c');
}
```

3. 模板與 constexpr
模板可以結合 constexpr 做編譯期運算。

``` cpp
template <int N>
constexpr int fib() {
    if constexpr (N <= 1) return N;
    else return fib<N-1>() + fib<N-2>();
}

int main() {
    constexpr int f5 = fib<5>();
    cout << f5 << endl; // 5
}
```

4. 模板別名（Alias Template）
簡化複雜模板型別。

``` cpp
template <typename T>
using Vec = std::vector<T>;

Vec<int> v = {1,2,3}; // std::vector<int> v;
```

5. 模板模板參數（Template Template Parameter）
可以將模板當作參數傳入。

``` cpp
template <template <typename, typename> class Container, typename T>
void printContainer(Container<T, std::allocator<T>>& c) {
    for (auto& x : c) cout << x << " ";
    cout << endl;
}

#include <vector>
int main() {
    std::vector<int> v = {1,2,3};
    printContainer(v); // 1 2 3
}
```

6. CRTP（Curiously Recurring Template Pattern）
讓子類型在父類別中被引用，常用於靜態多型。

``` cpp
template <typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() { std::cout << "Derived impl\n"; }
};

int main() {
    Derived d;
    d.interface(); // Derived impl
}
```

### 4. 總結

| 主題                   | 用途                        |
| ---------------------- | --------------------------- |
| 函式模板               | 支援同一份函式處理不同型別  |
| 類別模板               | 支援通用資料結構            |
| 模板特化               | 對特定型別或條件做不同實作  |
| SFINAE                 | 條件性選擇函式模板          |
| 可變參數模板           | 任意數量參數的模板函式/類別 |
| `constexpr` + template | 編譯期運算                  |
| 模板別名               | 簡化複雜型別                |
| 模板模板參數           | 以模板為參數                |
| CRTP                   | 靜態多型，編譯期多型技巧    |

Template 全圖解示意圖
![[template_type.png]]