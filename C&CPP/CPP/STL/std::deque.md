C++ std::deque 用法與範例
本篇將介紹如何使用 C++ std deque 以及用法與範例，C++ std::deque 是一個雙向佇列(double-ended queue)，在頭尾兩端插入及刪除十分快速，在中間插入刪除元素比較費時。

std::deque 是 double-ended queue 而不是 double linked list，底層實做是用間接索引的方式實現的，類似一個 map 索引到若干個固定大小的資料區塊(連續記憶體空間)，利用兩次索引達成跟 vector 一樣的隨機訪問功能。

以下將依序介紹幾個 std::deque 容器常用的用法範例，分別為

deque 常用功能
- 範例1. 基本的 push_back, pop_front, push_front, pop_back 的用法範例
- 範例2. push_back 自定義類別
- 範例3. 用 for 迴圈遍歷 deque 容器
- 範例4. 用 while 迴圈在 deque 容器裡搜尋/尋找
- deque 的優點與缺點
<p><span style="color:#ff0000; font-size:20px; font-weight:bold;">要使用 deque 容器的話，需要引入的標頭檔：&lt;deque&gt;</span></p>

deque 常用功能
以下為 std::deque 內常用的成員函式，
1. 修改器
   - push_back：把一個元素添加到尾端
   - push_front：把一個元素插入到頭端
   - pop_back：移除最後一個元素(尾端)
   - pop_front：移除第一個元素(頭端)
   - insert：插入元素
   - erase：移除某個位置元素, 也可以移除某一段範圍的元素
   - clear：清空容器裡所有元素
2. 容量
   - empty：回傳是否為空
   - size：回傳目前長度
3. 元素存取
   - [i]：隨機存取索引值為i的元素
   - at(i)：隨機存取索引值為i的元素，與[i]不同at(i)會檢查元素i是否超出deque邊界。<span style= "color:#ff0000; font-weight:bold;">如果超出會丟出例外</span>。
   - back：取得最後一個元素
   - front：取得第一個的元素
4. 迭代器
   - begin：回傳指向第一個元素(頭端)的迭代器
   - cbegin：回傳指向第一個元素(頭端)的迭代器(const)
   - end：回傳指向最後一個元素(尾端)的迭代器
   - cend：回傳指向最後一個元素(尾端)的迭代器(const)
   - rbegin：回傳指向最後一個元素(尾端)的反向迭代器
   - crbegin：回傳指向最後一個元素(尾端)的反向迭代器(const)
   - rend：回傳指向第一個元素(頭端)的反向迭代器
   - crend：回傳指向第一個元素(頭端)的反向迭代器(const)

⚙️ `operator at()`與 `operator []` 比較

| 函式           | 邊界檢查 | 效能    | 適用情境              |
| ------------ | ---- | ----- | ----------------- |
| `operator[]` | ```❌``` 無  | 🚀 快  | 已確定 index 合法時     |
| `at()`       | ✅ 有  | 🧩 稍慢 | Debug、或不確定索引是否安全時 |

泛型、安全、不丟例外 的 safe_at() 範例，支援任意 std::deque &lt;T&gt;（甚至也能套用在 std::vector 等容器上）。
會使用 C++17 以上語法（std::optional）。

🧩 範例程式：泛型安全訪問函式

``` cpp
#include <deque>
#include <optional>
#include <iostream>

// 🧠 泛型安全訪問函式
template <typename Container>
auto safe_at(const Container& c, size_t index)
    -> std::optional<typename Container::value_type>
{
    if (index < c.size())
        return c.at(index);
    else
        return std::nullopt;
}
```

🚀 使用範例

``` cpp
int main() {
    std::deque<int> dq = {10, 20, 30};

    if (auto val = safe_at(dq, 1)) {
        std::cout << "索引 1 的值 = " << *val << std::endl;
    } else {
        std::cout << "索引 1 超出範圍！" << std::endl;
    }

    if (auto val = safe_at(dq, 10)) {
        std::cout << "索引 10 的值 = " << *val << std::endl;
    } else {
        std::cout << "索引 10 超出範圍！" << std::endl;
    }
}
```

輸出：

``` bash
索引 1 的值 = 20
索引 10 超出範圍！
```

🧠 延伸版本（帶預設值）
有時候你不想回傳 optional，只希望「超出範圍時給個預設值」，可以這樣改：

``` cpp
template <typename Container>
typename Container::value_type
safe_at_or(const Container& c, size_t index,
           const typename Container::value_type& default_value)
{
    return (index < c.size()) ? c.at(index) : default_value;
}
```

用法：

``` cpp
int main() {
    std::deque<int> dq = {10, 20, 30};
    int val = safe_at_or(dq, 5, -1);
    std::cout << "結果 = " << val << std::endl;
}
```

輸出：

``` bash
結果 = -1
```

⚡ 如果你想更泛用（不只限 deque）
可以支援所有有 .size() 和 operator[] 的容器：

``` cpp
template <typename Container>
auto safe_index(const Container& c, size_t index)
    -> std::optional<typename Container::value_type>
{
    if (index < c.size())
        return c[index];  // 改用 operator[]，支援更多容器
    else
        return std::nullopt;
}
```

範例1. 基本的 push_back, pop_front, push_front, pop_back 的用法範例
以下範例為push_back(), pop_front(), push_front(), pop_back() 用法，
其中 push_back() 與 pop_front() 應該是最常用到的函式了。

<span style="color:#4A4AFF; font-size:20px; font-weight:bold; background-color:#F0F0F0F0;">另外使用 deque 相對於 queue 的好處是deque可以使用隨機訪問的功能 [i]。</span>

```cpp
// g++ std-deque.cpp -o a.out -std=c++11
#include <iostream>
#include <deque>

using namespace std;

int main() {
    deque<int> d = {1, 2, 3, 4};  // [1, 2, 3, 4]

    d.push_back(5); // [1, 2, 3, 4, 5]
    d.pop_front(); // [2, 3, 4, 5]
    d.push_front(0); // [0, 2, 3, 4, 5]
    d.pop_back(); // [0, 2, 3, 4]

    // 印出 deque 內所有內容, c++11 才支援
    for (int &i : d) {
        cout << i << " ";
    }
    cout << "\n";

    cout << d[0] << " " << d[1] << " " << d[2] << "\n";

    return 0;
}
```
輸出內容如下：

``` bash
0 2 3 4
0 2 3
```

範例2. push_back 自定義類別
以下範例為 std::deque 容器使用 push_back() 來推放 Student 自定義類別的範例，使用 push_back() 來放進 deque 的最尾端，這個範例限制 deque 最多塞3個，多的會用 pop_front() 給 pop 掉，最後再將 deque 容器的所有的元素印出來。

```cpp
// g++ std-deque2.cpp -o a.out -std=c++11
#include <iostream>
#include <deque>

using namespace std;

class Student {
public:
    Student(int id) { this->id = id; }

    int id;
};

std::deque<Student> d;

void deque_push_back(Student a) {
    d.push_back(a);
    if (d.size() > 3) {
        d.pop_front();
    }
}

int main() {
    Student a1(1), a2(2), a3(3), a4(4);
    deque_push_back(a1);
    deque_push_back(a2);
    deque_push_back(a3);
    deque_push_back(a4);

    // 印出 deque 內所有內容, c++11 才支援
    for (auto &i : d) {
        cout << i.id << " ";
    }
    cout << "\n";

    return 0;
}
```

範例3. 用 for 迴圈遍歷 deque 容器
以下範例是用 for 迴圈配合 deque 容器的迭代器，去遍歷 deque 並且把值印出來，前兩種是從頭印到尾，後兩種是從尾印到頭。從頭端印到最尾端就是使用 begin() 搭配 end() ，從最尾端印到頭端就是使用 rbegin() 搭配 rend() 。

這裡的 begin() / end() 與 cbegin() / cend() 有什麼不同呢？begin() / end() 是回傳 iterator，而 cbegin() / cend() 是回傳 const_iterator，iterator 可以修改元素值，const_iterator 則不可修改，簡單說就是回傳的東西能不能被修改的差異，要用哪種就自行判斷要用哪種了。

cbegin(), cend(), crbegin(), crend() 是 C++11 新增的，要使用時記得編譯器要加入-std=c++11選項。

第一個 for 迴圈裡面的迭代器使用懶人快速寫法auto it = d.begin();，其全名為std::deque<int>::iterator it = d.begin();，如果不想寫這麼長的話，就可以像我一樣用 auto 的懶人快速寫法。

```cpp
// g++ std-deque3.cpp -o a.out -std=c++11
#include <iostream>
#include <deque>

using namespace std;

int main() {
    deque<int> d = {1, 2, 3, 4};

    // 從頭到尾
    //for (std::deque<int>::iterator it = d.begin(); it != d.end(); ++it) {
    for (auto it = d.begin(); it != d.end(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    // 從頭到尾
    for (auto it = d.cbegin(); it != d.cend(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    // 從尾到頭
    for (auto it = d.rbegin(); it != d.rend(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    // 從尾到頭
    for (auto it = d.crbegin(); it != d.crend(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    return 0;
}
```

輸出

``` bash
1 2 3 4
1 2 3 4
4 3 2 1
4 3 2 1
```

範例4. 用 while 迴圈在 deque 容器裡搜尋/尋找
以下範例是用 while 迴圈在 deque 容器裡搜尋/尋找數字為3，這裡的 auto 懶人快速寫法如上範例解釋相同就不贅述了。

在 while 迴圈裡如果有找到就印個有找到的訊息，假如整個 while 迴圈都沒找到，最後可以判斷 it == d.end() 代表整個 deque 容器都遍歷過了也沒有找到，就印個沒找到的訊息。

```cpp
// g++ std-deque4.cpp -o a.out -std=c++11
#include <iostream>
#include <deque>

using namespace std;

int main() {
    deque<int> d = {1, 2, 3, 4};

    // 從頭到尾
    int find_num = 3;
    //std::deque<int>::iterator it = d.begin();
    auto it = d.begin();
    while (it != d.end()) {
        if (*it == find_num) {
            cout << "Found " << find_num << endl;
            break;
        }
        it++;
    }
    if (it == d.end()) {
        cout << "Not found " << find_num << endl;
    }

    return 0;
}
```

輸出

``` bash
Found 3
```

如果將 find_num 變數改成 5 去搜尋/尋找的話則會找不到，輸出結果會變成

``` bash
Not found 5
```

deque 的優點

- 可以再兩端進行 push 和 pop 操作
- 支持隨機訪問[i]

deque 的缺點

- 佔用記憶體較多

std::deque 把指定 index 的元素移到另一個位置

這就是「deque 內局部重排」的典型需求。
由於 std::deque 沒有 splice()（不像 std::list 那樣能零拷貝移動節點），
我們得用一點 STL 魔法來做到「把指定 index 的元素搬到另一個位置」。

實作：move element by index（支援任意位置移動）

``` cpp
#include <deque>
#include <iostream>
#include <algorithm> // for std::rotate
#include <utility>   // for std::move

template <typename T>
void move_element(std::deque<T>& dq, size_t from, size_t to)
{
    if (from >= dq.size() || to >= dq.size()) {
        throw std::out_of_range("Index out of range");
    }

    // 📦 Case 1: 向前移動（from > to）
    if (from > to) {
        std::rotate(dq.begin() + to, dq.begin() + from, dq.begin() + from + 1);
    }
    // 📦 Case 2: 向後移動（from < to）
    else if (from < to) {
        std::rotate(dq.begin() + from, dq.begin() + from + 1, dq.begin() + to + 1);
    }
    // 📦 Case 3: from == to → 不動
}
```

🧩 使用範例

``` cpp
int main() {
    std::deque<int> dq = {10, 20, 30, 40, 50};

    move_element(dq, 1, 3); // 把 index 1 (20) 移到 index 3 的位置
    // 結果: 10, 30, 40, 20, 50

    for (int x : dq)
        std::cout << x << " ";
}
```

🔹輸出：

``` bash
10 30 40 20 50
```

🧠 為什麼用 std::rotate()？

std::rotate(first, middle, last) 會把 [first, middle] 移到尾巴，
剩下的 [middle, last] 往前推。

👉 我們藉此把「要移動的元素」視為 [middle, middle+1] 區間，
然後旋轉到新位置，就能達到「搬動」效果。

範例：

``` cpp
// 從 [1] 移到 [3]
rotate(begin+1, begin+2, begin+4);
```

這會讓：

``` bash
[10, 20, 30, 40, 50] → [10, 30, 40, 20, 50]
```

⚙️ 性能筆記
| 操作                    | 時間複雜度      | 是否 copy/move 元素      |
| --------------------- | ---------- | -------------------- |
| `std::rotate()`       | O(n)（區段長度） | ✅ 會做 move assignment |
| `std::list::splice()` | O(1)       | ```❌``` 不會 move/copy（零拷貝）  |


👉 對於小型元素（int、float）或中等 deque，rotate() 很 OK。
若你需要頻繁搬移大量元素、而且元素是大型物件 → 用 std::list 會更划算。

<pan style="font-weight:bold; font-size:15px;">可泛用於 std::vector、std::deque、甚至自定容器（有 random-access iterator） 的 move_element() 泛型函式。</span>

目標：

- 支援 任意容器類型（只要有 begin()、end() 和隨機訪問迭代器）
- 支援「向前」與「向後」移動
- 安全檢查 + 明確語意
- 只使用標準函式庫

🧠 泛型實作（C++17 起）

``` cpp
#include <algorithm>  // std::rotate
#include <iterator>   // std::begin, std::end
#include <stdexcept>  // std::out_of_range

template <typename Container>
void move_element(Container& c, size_t from, size_t to)
{
    using std::begin;
    using std::end;

    const size_t n = std::distance(begin(c), end(c));
    if (from >= n || to >= n)
        throw std::out_of_range("Index out of range");

    auto first = begin(c);

    if (from == to) return; // 無需移動

    if (from < to) {
        // 把 [from] 移到 [to]
        std::rotate(first + from, first + from + 1, first + to + 1);
    } else {
        // 把 [from] 移到 [to]
        std::rotate(first + to, first + from, first + from + 1);
    }
}
```

🧩 範例使用

``` cpp
#include <iostream>
#include <vector>
#include <deque>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    move_element(v, 1, 3); // 把 index=1 (2) 移到 index=3
    // 結果: 1 3 4 2 5

    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    std::deque<std::string> dq = {"A", "B", "C", "D"};
    move_element(dq, 2, 0); // 把 "C" 移到最前面
    // 結果: C A B D

    for (auto& s : dq) std::cout << s << " ";
}
```

🔹輸出：

``` bash
1 3 4 2 5
C A B D
```

🧩 為什麼這樣設計？

- 用 std::rotate() 處理「搬動」邏輯，避免手動 insert/erase 複雜操作
- 不需顧慮元素是否有 move constructor（rotate 會自動用 move）
- 適用於：
  - std::vector
  - std::deque
  - std::array
  - 甚至你自己的容器（只要支援 random access iterator）

⚙️ 時間複雜度

| 操作              | 複雜度                                       | 備註                   |
| --------------- | ----------------------------------------- | -------------------- |
| `std::rotate()` | O(n)（n 為移動區段長度）                           | 用 move assignment 實現 |
| 支援容器            | random access 型（`vector`、`deque`、`array`） | ✅                    |
| 不支援容器           | linked list 型（`list`, `forward_list`）     | ```❌```                    |

<span style="font-weight:bold; font-size:15px;">智慧泛型版 move_element()</span>

可同時支援：
- std::vector, std::deque, std::array（隨機訪問容器）
- std::list（雙向鏈結容器，用 splice() 零拷貝）

這樣無論你用什麼 STL 容器，都能優雅又高效地「移動指定 index 的元素到新位置」。

🧠 完整泛型實作（支援 list / deque / vector）

``` cpp
#include <algorithm>
#include <iterator>
#include <list>
#include <stdexcept>
#include <type_traits>

// ===================================================================
// 🔹 for random-access containers (vector, deque, array)
// ===================================================================
template <typename Container>
std::enable_if_t<
    std::is_same_v<typename std::iterator_traits<typename Container::iterator>::iterator_category,
                   std::random_access_iterator_tag>>
move_element(Container& c, size_t from, size_t to)
{
    if (from >= c.size() || to >= c.size())
        throw std::out_of_range("Index out of range");
    if (from == to)
        return;

    auto first = c.begin();
    if (from < to)
        std::rotate(first + from, first + from + 1, first + to + 1);
    else
        std::rotate(first + to, first + from, first + from + 1);
}

// ===================================================================
// 🔹 for std::list (use splice, zero-copy move)
// ===================================================================
template <typename T>
void move_element(std::list<T>& lst, size_t from, size_t to)
{
    if (from >= lst.size() || to >= lst.size())
        throw std::out_of_range("Index out of range");
    if (from == to)
        return;

    auto fromIt = lst.begin();
    std::advance(fromIt, from);

    auto toIt = lst.begin();
    std::advance(toIt, to);

    if (from < to)
        ++toIt; // 插入在目標之後，保持語意一致（移到 index=to）

    lst.splice(toIt, lst, fromIt); // 💥 O(1) 真正零拷貝搬移
}
```

🧩 使用範例

``` cpp
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <string>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50};
    move_element(v, 1, 3); // 把 20 移到 index=3
    for (auto x : v) std::cout << x << " ";
    std::cout << "\n";

    std::deque<std::string> dq = {"A", "B", "C", "D"};
    move_element(dq, 2, 0); // 把 "C" 移到最前面
    for (auto& s : dq) std::cout << s << " ";
    std::cout << "\n";

    std::list<char> lst = {'a', 'b', 'c', 'd', 'e'};
    move_element(lst, 4, 1); // 把 'e' 移到 index=1
    for (auto ch : lst) std::cout << ch << " ";
    std::cout << "\n";
}
```

🔹 輸出：

``` bash
10 30 40 20 50
C A B D
a e b c d
```

⚙️ 實作細節重點解釋
| 容器類型                                      | 移動方法            | 時間複雜度 | 是否 move/copy 元素   | 特點    |
| ----------------------------------------- | --------------- | ----- | ----------------- | ----- |
| `std::vector`, `std::deque`, `std::array` | `std::rotate()` | O(n)  | ✅ move assignment | 通用、安全 |
| `std::list`                               | `splice()`      | O(1)  | ```❌``` 不 move/copy     | 真正零拷貝 |


參考
[std::deque - cppreference.com](https://en.cppreference.com/w/cpp/container/deque)
[deque - C++ Reference](http://www.cplusplus.com/reference/deque/deque/)
[Deque in C++ Standard Template Library (STL) - GeeksforGeeks](https://www.geeksforgeeks.org/deque-cpp-stl/)
