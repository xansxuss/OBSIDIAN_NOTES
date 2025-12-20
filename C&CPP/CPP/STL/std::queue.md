C++ std::queue 用法與範例
本篇介紹 C++ 的 std::queue 用法，C++ std::queue 教學如下：

std::queue 是具有 FIFO 特性的容器配接器, 應用在有先進先出的情形。
queue 是一層容器的包裝, 背後是用 deque 實現的, 並且只提供特定的函數接口。

以下內容將分為這幾部份，

- queue 常用功能
- C++ queue 範例
- queue 的優點與缺點
<p><span style="color:#ff0000; font-size:20px; font-weight:bold;">c++ 要使用 queue 容器的話，需要引入的標頭檔：&lt;queue&gt;</span></p>

queue 常用功能
以下為 std::queue 內常用的成員函式
- push：把值加到尾巴
- pop：移除頭的值
- back：回傳尾巴的值
- front：回傳頭的值
- size：回傳目前長度
- empty：回傳是否為空

C++ queue 範例
以下為 c++ queue 的各種操作用法，把元素加進 queue 的尾部使用 push()，
- 把元素從 queue 頭部取出用 pop()，注意取出會將該元素從 queue 移除，
- 取得 queue 的最尾巴的元素使用 back()，
- 取得 queue 的最頭部的元素使用 front()，注意取得並不會將該元素從 queue 移除，
- 取得 queue 目前裡面有幾個元素使用 size()，

``` cpp
// g++ std-queue.cpp -o a.out -std=c++11
#include <iostream>
#include <queue>

using namespace std;

int main() {
    queue<int> q;
    q.push(1); // [1]
    q.push(2); // [1, 2]
    q.push(3); // [1, 2, 3]

    cout << q.front() << endl; // 1
    cout << q.back() << endl; // 3
    cout << q.size() << endl; // 3

    int a = q.front(); // copy
    int &b = q.front(); // reference

    cout << q.front() << " " << &q.front() << endl; // 印記體位置
    cout << a << " " << &a << endl;
    cout << b << " " << &b << endl; // 與 q.front() 記憶體位置相同

    // 印出 queue 內所有內容
    int size = q.size();
    for (int i = 0; i < size; i++) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << "\n";

    // 印出 queue 內所有內容
    /*while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << "\n";*/

    return 0;
}
```

輸出內容如下：

``` bash
1
3
3
1 0xb77c70
1 0x7ffe63ead460
1 0xb77c70
1 2 3
```

queue 的優點

- 快速的把頭的值拿掉

queue 的缺點

- 只能操作頭跟尾, 不能取得中間的值(根據FIFO特性)

### 型別定義（type aliases）

| 名稱                | 說明                           |
| ----------------- | ---------------------------- |
| `container_type`  | 底層容器的型別（預設為 `std::deque<T>`） |
| `value_type`      | 儲存的資料型別                      |
| `size_type`       | 表示大小的型別（通常是 `std::size_t`）   |
| `reference`       | `value_type&`                |
| `const_reference` | `const value_type&`          |
### 建構與指定

| 函式                                      | 說明       |
| --------------------------------------- | -------- |
| `queue()`                               | 預設建構子    |
| `explicit queue(const Container& cont)` | 用指定容器初始化 |
| `explicit queue(Container&& cont)`      | 移動建構子    |
| `queue(const queue& other)`             | 複製建構子    |
| `queue(queue&& other)`                  | 移動建構子    |
| `operator=(const queue& other)`         | 複製指定     |
| `operator=(queue&& other)`              | 移動指定     |

### ### 元素操作（核心 API）

|函式|說明|
|---|---|
|`void push(const value_type& value)`|將元素推入隊尾（複製）|
|`void push(value_type&& value)`|將元素推入隊尾（移動）|
|`template<class... Args> void emplace(Args&&... args)`|**原地構造（in-place construct）** 一個新元素於隊尾|
|`void pop()`|移除隊首元素（但不返回值）|
|`reference front()`|取得隊首元素（可修改）|
|`const_reference front() const`|取得隊首元素（唯讀）|
|`reference back()`|取得隊尾元素（可修改）|
|`const_reference back() const`|取得隊尾元素（唯讀）|

### 容量相關

|函式|說明|
|---|---|
|`bool empty() const`|檢查是否為空|
|`size_type size() const`|回傳元素數量|

### 比較運算子（C++20 起支援）

| 運算子                                                  | 說明              |
| ---------------------------------------------------- | --------------- |
| `operator==`, `operator!=`                           | 比較內容是否相等        |
| `operator<`, `operator<=`, `operator>`, `operator>=` | 根據底層容器的比較結果決定順序 |

**`std::queue` 不能直接拿元素指標**，因為它**不暴露底層容器的 iterator 或 reference 列表**，只有 `front()` 和 `back()` 可以取得「隊首/隊尾」的引用（reference），而沒有像 vector/deque 那樣的 `begin()` 或 `operator[]`。

---

### 🧩 為什麼？

`std::queue` 是個 **container adapter**，它的設計目的就是：

> 限制操作介面，只允許 FIFO 模式（先進先出）。

也就是說它**封裝**了一個底層容器（預設是 `std::deque<T>`），但不讓你直接碰裡面的細節。  
因此：

``` CPP
std::queue<int> q;
q.push(10);
int *ptr = &q.front();  // ✅ OK：取得第一個元素的指標
int *ptr2 = &q.back();  // ✅ OK：取得最後一個元素的指標
```

但是：

``` cpp
// ❌ 不行：沒有迭代器或索引操作
// auto it = q.begin();      // compile error
// auto val = q[0];          // compile error
```

拿元素指標

有三個安全做法：

#### ✅ 方法 1：使用底層容器（官方推薦等價物）

若你願意使用「等價底層容器」：
``` cpp
std::deque<int> dq;
std::queue<int> q(std::move(dq)); // q 用 dq 建構

// 你可以操作 dq
dq.push_back(1);
dq.push_back(2);
int *ptr = &dq[0];
```

但注意：此時 `q` 只是包裝 `dq` 的**拷貝**，修改 `dq` 不會影響 `q`。

---

#### ✅ 方法 2：直接用 `std::deque` 取代

如果你需要頻繁存取任意元素或指標：

``` cpp
std::deque<MyStruct> dq;
dq.push_back({1});
dq.push_back({2});
MyStruct *p = &dq[0]; // ✅ 直接拿元素指標
```

`std::queue` 本質上只是 `std::deque` 的受限版：

``` cpp
template <class T, class Container = std::deque<T>>
class queue {
protected:
    Container c; // 底層容器
public:
    // push, pop, front, back, size, empty...
};
```

#### 🚨 方法 3（Hack 用途）：C++ 非標準手段（**不建議正式用**）

若你**一定要取 queue 的底層容器**，可用：

``` cpp
#include <queue>
#include <deque>

int main() {
    std::queue<int> q;
    q.push(1);
    q.push(2);

    // ⚠️ 非標準技巧：存取底層容器
    auto &container = const_cast<std::deque<int>&>(
        *reinterpret_cast<const std::deque<int>*>(&q)
    );

    for (auto &x : container)
        std::cout << x << " ";
}
```

> ⚠️ **這是未定義行為（UB）**，不同實作、不同編譯器都可能炸。  
> 只適合「debug」或「短期實驗」，不應進入正式程式碼。

---

### 🔍 總結表

| 想拿什麼       | 能不能做 | 方法              |
| ---------- | ---- | --------------- |
| 取第一個元素指標   | ✅    | `&q.front()`    |
| 取最後一個元素指標  | ✅    | `&q.back()`     |
| 遍歷整個 queue | ❌    | 不能（無 iterator）  |
| 取得底層容器     | ⚠️   | 非標準、未定義行為       |
| 想操作整體元素    | ✅    | 改用 `std::deque` |

參考
[std::queue - cppreference.com](https://en.cppreference.com/w/cpp/container/queue)
[queue - C++ Reference](http://www.cplusplus.com/reference/queue/queue/)
[Queue in Standard Template Library (STL) - GeeksforGeeks](https://www.geeksforgeeks.org/queue-cpp-stl/)