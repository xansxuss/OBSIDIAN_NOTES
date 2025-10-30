## 1️⃣ 基本概念

`std::scoped_lock` 是 **一個多 mutex 的 RAII 封裝**，可以自動在作用域結束時釋放鎖，避免死鎖問題。

特點：

1. 支援單一或多個 mutex。
    
2. RAII 風格：離開作用域自動 unlock。
    
3. 可避免多 mutex deadlock（C++17 支援內部 `std::lock` 的 deadlock-safe 鎖定）。
    

對比：

| 類別               | 特點                                      |
| ------------------ | ----------------------------------------- |
| `std::lock_guard`  | 單 mutex，作用域自動 unlock               |
| `std::unique_lock` | 可 lock/unlock 多次、可延遲 lock          |
| `std::scoped_lock` | 可同時 lock 多 mutex，RAII，deadlock safe |

---

## 2️⃣ 基本語法

``` cpp
#include <mutex>

std::mutex m1, m2;

void func() {
    std::scoped_lock lock(m1, m2); // 同時鎖定 m1 與 m2
    // 這裡是臨界區
} // lock 會自動釋放

```

重點：

- 傳入的 mutex 可以是任意數量。
- 在構造函數中自動鎖定，析構時自動釋放。
- 適合多線程共享資源保護。

3️⃣ 單 mutex 範例

``` cpp
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;

void print_thread_id(int id) {
    std::scoped_lock lock(mtx); // 自動鎖定 mtx
    std::cout << "Thread " << id << " is running\n";
}

int main() {
    std::thread t1(print_thread_id, 1);
    std::thread t2(print_thread_id, 2);

    t1.join();
    t2.join();
}

```
✅ 行為：每個線程會依序進入臨界區，互不干擾，無需手動 unlock。

---

## 4️⃣ 多 mutex 範例（避免死鎖）

假設我們有兩個資源需要同時鎖定：

``` cpp
#include <iostream>
#include <mutex>
#include <thread>

std::mutex m1, m2;

void task1() {
    std::scoped_lock lock(m1, m2); // deadlock-safe
    std::cout << "Task1 running\n";
}

void task2() {
    std::scoped_lock lock(m2, m1); // 仍然安全
    std::cout << "Task2 running\n";
}

int main() {
    std::thread t1(task1);
    std::thread t2(task2);

    t1.join();
    t2.join();
}

```

⚡ 解析：

- `std::scoped_lock` 內部使用了 `std::lock(m1, m2)`。
- `std::lock` 會一次性鎖住所有 mutex，避免死鎖。
- 不必擔心不同順序鎖定造成的 deadlock。   

---

## 5️⃣ 與 `std::unique_lock` 比較

| 功能             | std::scoped_lock      | std::unique_lock                   |
| ---------------- | --------------------- | ---------------------------------- |
| 多 mutex         | ✅ 支援                | ```❌``` 不支援一次鎖多 mutex       |
| 手動 lock/unlock | ```❌``` 不支援        | ✅ 支援                             |
| 延遲 lock        | ```❌``` 不支援        | ✅ 支援                             |
| RAII 自動 unlock | ✅                     | ✅                                  |
| 適合情境         | 簡單臨界區、死鎖-safe | 需要條件變量、延遲鎖、轉移鎖擁有權 |

---

## 6️⃣ 注意事項

1. 不可拷貝或移動 `std::scoped_lock`：
```cpp
	std::scoped_lock lock1(mtx);
	// std::scoped_lock lock2 = lock1; // ❌ 錯誤
```
2. 構造時就鎖定，析構時才釋放。
3. 適合**多 mutex 場景**，比多個 `lock_guard` 更安全。
4. C++17 以上才可使用。

## 7️⃣ 綜合範例：多資源安全更新

```cpp
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex mtx1, mtx2;
int counter1 = 0, counter2 = 0;

void increment() {
    for (int i = 0; i < 1000; ++i) {
        std::scoped_lock lock(mtx1, mtx2);
        ++counter1;
        ++counter2;
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
        threads.emplace_back(increment);

    for (auto &t : threads)
        t.join();

    std::cout << "counter1: " << counter1 << ", counter2: " << counter2 << "\n";
}

```

✅ 保證多線程下 `counter1` 和 `counter2` 不會出現 race condition，且不會死鎖。

---

總結：
- `std::scoped_lock` = **C++17 的多 mutex RAII 解決方案**。
- 適合簡單、死鎖安全、一次性鎖定多 mutex。
- 與 `unique_lock` 搭配條件變量也可用，但不能手動 lock/unlock。