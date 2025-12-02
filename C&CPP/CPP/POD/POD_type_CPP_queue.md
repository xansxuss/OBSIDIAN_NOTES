符合 POD type 的 C++ queue」，其實要先拆解兩個概念：

POD (Plain Old Data)
1. 在 C++11 之後，POD 可以理解為簡單的結構或基本型態，沒有自訂建構/解構函式、沒有虛擬函式、沒有非靜態成員初始化。
例如：

``` cpp
struct MyData {
    int a;
    float b;
}; // 這是 POD
```

2. Queue
一般可以用 STL 的 std::queue<T>，但要注意 STL container（像 std::queue 的底層 std::deque）會呼叫元素的建構與析構。如果你只想要純 POD，想最大化效能（例如用 memcpy 移動），可能要使用自訂環形 buffer / ring buffer 或**boost::lockfree::queue**。

**簡單 POD queue 範例（使用環形陣列）**

``` cpp
#include <cstddef>
#include <cassert>
#include <iostream>

template<typename T, std::size_t N>
class PODQueue {
    static_assert(std::is_trivial<T>::value, "T 必須是 POD type");
public:
    PODQueue() : head(0), tail(0), count(0) {}

    bool push(const T& val) {
        if (count >= N) return false; // queue 滿了
        buffer[tail] = val;
        tail = (tail + 1) % N;
        ++count;
        return true;
    }

    bool pop(T& val) {
        if (count == 0) return false; // queue 空了
        val = buffer[head];
        head = (head + 1) % N;
        --count;
        return true;
    }

    bool empty() const { return count == 0; }
    bool full() const { return count == N; }

private:
    T buffer[N];
    std::size_t head;
    std::size_t tail;
    std::size_t count;
};

struct MyData {
    int a;
    float b;
};

int main() {
    PODQueue<MyData, 4> q;
    q.push({1, 1.1f});
    q.push({2, 2.2f});

    MyData val;
    while(q.pop(val)) {
        std::cout << val.a << ", " << val.b << "\n";
    }
}
```

這種寫法：

- 完全 POD-safe：T 必須是 trivial（std::is_trivial）
- 零額外配置：沒有 new/delete，全部在 stack/固定陣列
- 快速 memcpy-style 移動