## 🧩 一、`std::lock_guard` 是什麼？

`std::lock_guard` 是一個 **RAII（Resource Acquisition Is Initialization）鎖管理類別**。  
它的任務只有一個：

> 在建構時鎖定 mutex，解構時自動解鎖。

這樣就不用你手動 `lock()` / `unlock()`，減少「忘記解鎖」或「例外導致死鎖」的悲劇。

---

## 🧠 二、基本語法

```cpp
#include <mutex>
std::mutex mtx;
void safeFunction()
{
std::lock_guard<std::mutex> lock(mtx);  // 鎖定
// ---- 臨界區開始 ----
std::cout << "Thread-safe operation\n";
// ---- 臨界區結束 ---- 
} // 離開作用域自動解鎖
```

💡 一離開作用域（包括因為例外或 `return`），`lock` 會自動呼叫 `mtx.unlock()`。  
這是 RAII 的精隨。

---

## 🧱 三、建構子參數

`std::lock_guard` 有兩種主要建構方式：

``` cpp
explicit lock_guard(mutex_type& m);             // 自動 lock() 
lock_guard(mutex_type& m, adopt_lock_t tag);    // 不 lock，只接手已鎖的 mutex
```

### ✅ 常見用法（自動 lock）

``` cpp
std::lock_guard<std::mutex> guard(mtx);
```

### ⚠️ `adopt_lock` 用法（你先手動 lock）

``` cpp
mtx.lock();
std::lock_guard<std::mutex> guard(mtx, std::adopt_lock); // 告訴 guard 這個鎖已經被鎖住 
// guard 不會再 lock() 一次，但會在解構時 unlock()
```

如果你沒用 `adopt_lock` 而直接傳入一個已被 lock 的 mutex，  
會造成 **double lock（死鎖）**。

---

## ⚔️ 四、與 `std::unique_lock` 的比較

| 特性                | `std::lock_guard` | `std::unique_lock`    |
| ----------------- | ----------------- | --------------------- |
| 鎖定策略              | 建構時立即鎖定           | 可延遲鎖定、可手動 lock/unlock |
| 成本                | 輕量級（幾乎零開銷）        | 稍重一點（多些狀態記錄）          |
| 可移動性              | ```❌``` 不可移動            | ✅ 可移動（常用於條件變數）        |
| 可手動 unlock/relock | ```❌``` 不可              | ✅ 可                   |
| 適用場合              | 簡單臨界區             | 複雜控制（條件變數、動態鎖定）       |

👉 一句話總結：

> 「**lock_guard 是最簡單的保險套，unique_lock 是多功能安全套。**」

---

## 🧮 五、實際多執行緒範例

```cpp
#include <iostream>
#include <thread>
#include <mutex>
std::mutex coutMutex;
void printSafe(int id)
{
	std::lock_guard<std::mutex> lock(coutMutex);
	std::cout << "Thread " << id << " says hello!\n";
}
int main()
{
	std::thread t1(printSafe, 1);
	std::thread t2(printSafe, 2);
	t1.join();
	t2.join();
}
```

🧠 這裡 `lock_guard` 確保同一時間只有一個執行緒能寫入 `std::cout`。  
否則可能會交錯輸出像：

``` bash
	Thread 1 says Threa2 syss hello!
```

---

## ⚠️ 六、常見陷阱

### ❌ 1. 不同函式鎖同一把 mutex

``` cpp
void funcA()
{
	std::lock_guard<std::mutex> lock(mtx);
	funcB(); // funcB 裡也鎖同一個 mtx
}
```

→ 造成**同一執行緒重入死鎖**（非 recursive_mutex 時）。  
✅ 解法：使用 `std::recursive_mutex` 或重構鎖範圍。

---

### ❌ 2. 多把 mutex 鎖順序不一致

``` cpp
std::mutex m1, m2;
void t1()
{
	std::lock_guard<std::mutex> l1(m1);
	std::lock_guard<std::mutex> l2(m2);
}
void t2()
{
	std::lock_guard<std::mutex> l1(m2);
	std::lock_guard<std::mutex> l2(m1);
}
```

→ 死鎖。  
✅ 解法：保持固定鎖順序，或使用 `std::scoped_lock`（C++17）。

``` cpp
std::scoped_lock lock(m1, m2); // 同時鎖，保證無死鎖
```

---

## 🧰 七、結合 STL 容器使用

多執行緒 push/pop：

``` cpp
std::queue<int> q;
std::mutex qMutex;
void producer()
{
for (int i = 0; i < 5; ++i)
	{
		std::lock_guard<std::mutex> lock(qMutex);
		q.push(i);
	}
}
void consumer()
{
	while (true)
	{
		std::lock_guard<std::mutex> lock(qMutex);
		if (!q.empty())
		{
			std::cout << "Consume " << q.front() << "\n";
			q.pop();
		}
	}
}
```


---

## 🧩 八、C++17 的 `std::scoped_lock` 是誰？
[[std::scoped_lock]]
可以把它當成：

`std::scoped_lock guard(m1, m2);`

= 同時鎖多個 mutex、並避免死鎖。  
功能像多個 `lock_guard` 的超進化版。

---

## 🔮 九、最佳實踐建議

1. ✅ **總是用 RAII 鎖**（`lock_guard` / `unique_lock`），別用裸 `lock()` / `unlock()`。
    
2. ✅ **鎖範圍最小化** — 縮到只包住必要臨界區。
    
3. ✅ **固定鎖順序** — 避免死鎖。
    
4. ⚠️ **不要 copy `lock_guard`**，它不可複製。
    
5. ✅ **用 `scoped_lock` 鎖多 mutex**。
    
6. ✅ **用 `unique_lock`** 搭配條件變數（`std::condition_variable`）。
    

---

## 🔍 十、總結心智模型

| 狀況                                    | 選擇                                         |
| ------------------------------------- | ------------------------------------------ |
| 簡單的臨界區                                | `std::lock_guard`                          |
| 需要 unlock/relock / condition variable | `std::unique_lock`                         |
| 同時鎖多個 mutex                           | `std::scoped_lock`                         |
| 同一執行緒重入                               | `std::recursive_mutex` + `std::lock_guard` |