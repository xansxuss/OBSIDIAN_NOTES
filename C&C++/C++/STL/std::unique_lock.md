## 🧠 一、核心概念：什麼是 `std::unique_lock`

`std::unique_lock` 是 C++ `<mutex>` 裡的一個「**可移動、可延遲上鎖、可解鎖再上鎖**」的 RAII 鎖管理器。  
它比 `std::lock_guard` 更靈活，但也稍微重一些（多一點 runtime 成本）。

簡單來說：

> `std::unique_lock` = RAII + 彈性鎖控制  
> `std::lock_guard` = RAII + 簡單的立即鎖住解鎖

---

## 🔩 二、使用時機比較

|功能項目|`std::lock_guard`|`std::unique_lock`|
|---|---|---|
|自動上鎖解鎖|✅ 是|✅ 是|
|延遲上鎖（稍後才 lock）|```❌``` 否|✅ 是|
|手動解鎖/再上鎖|```❌``` 否|✅ 是|
|搭配 `std::condition_variable`|```❌``` 否|✅ 必須用它|
|可移動（move）|```❌``` 否|✅ 是|
|較輕量（速度快）|✅ 是|⚠️ 否，略慢|
|適合場合|簡單臨界區|複雜同步控制|

---

## 🧩 三、基本語法

```cpp
#include <mutex> 
#include <thread> 
#include <iostream>  
std::mutex mtx;  
void worker()
{
std::unique_lock<std::mutex> lock(mtx); // 自動上鎖
std::cout << "工作中...\n";
} // 離開 scope 自動解鎖
```

---

## 🔄 四、延遲上鎖（`std::defer_lock`）

有時你想要**先建立鎖物件**但**稍後才上鎖**：

```cpp 
std::mutex mtx;
void example()
{
std::unique_lock<std::mutex> lock(mtx, std::defer_lock); // 不立即鎖
// 做點其他事
lock.lock(); // 需要時再鎖
std::cout << "Critical section\n";
lock.unlock(); // 可自行解鎖
}
```

> ✅ 適合在你要控制 lock timing 或使用多 mutex（例如 `std::lock` 同時鎖多個）時。

---

## ⚙️ 五、試圖上鎖（`try_lock`）

``` cpp 
void try_example()
{
	std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
	if (lock.try_lock())
		{
		std::cout << "成功拿到鎖！\n";
		}
	else
		{
		std::cout << "有人在用，跳過！\n";
		}
}
```

> ⚡ 適合高效能場景（例如 lock-free queue 裡面想試試能不能拿鎖）。

---

## ⏰ 六、時間限制鎖（`try_lock_for`, `try_lock_until`）

配合 `std::timed_mutex` 使用：

``` cpp
#include <mutex>
#include <chrono>
std::timed_mutex tmtx;
void timed_lock()
{
	std::unique_lock<std::timed_mutex> lock(tmtx, std::defer_lock);
	if (lock.try_lock_for(std::chrono::milliseconds(100)))
		{
			std::cout << "在 100ms 內成功上鎖\n";
		}
	else
		{
			std::cout << "等太久，放棄\n";
		}
}
```

---

## 🧵 七、可手動解鎖再上鎖

``` cpp 
void relock_example()
{
	std::unique_lock<std::mutex> lock(mtx);
	std::cout << "鎖住中...\n";
	lock.unlock();
	std::cout << "暫時釋放鎖\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	lock.lock();
	std::cout << "重新上鎖！\n";
}
```
---

## 💤 八、搭配 `std::condition_variable`

`condition_variable`[[std::condition_variable]] 需要「**能解鎖再上鎖的鎖**」，  
所以**只能用 `unique_lock`**，不能用 `lock_guard`。

``` cpp
#include <condition_variable> 
#include <mutex> 
#include <thread> 
#include <iostream>
std::mutex mtx;
std::condition_variable cv;
bool ready = false;
void worker()
	{
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [] { return ready; }); // 自動解鎖等待，再上鎖
		std::cout << "收到通知，開始工作！\n";
	}
void notifier()
{
	std::this_thread::sleep_for(std::chrono::seconds(1));
	{
		std::lock_guard<std::mutex> lock(mtx);
		ready = true;
	}
	cv.notify_one();
}
```

---

## 🔁 九、可移動性（Move Semantics）

`unique_lock` 可以轉移所有權，這在多執行緒控制中很方便。

``` cpp
std::unique_lock<std::mutex> f() {
	std::unique_lock<std::mutex> lock(mtx);     // 做一些初始化
return lock; // move semantics
}  
void g() {
	auto lock = f(); // 拿到 f() 的鎖所有權 
}
```

> ⚙️ 注意：不能複製（copy），只能移動（move）。

---

## 🧨 十、常見錯誤與陷阱

|錯誤情境|說明|
|---|---|
|重複上鎖|會導致死鎖|
|忘記 `defer_lock` 時呼叫 `lock()`|會 double lock|
|`unique_lock` 超出 scope 仍需要鎖|可能提前解鎖造成 race condition|
|多執行緒同時用同一個 `unique_lock`|❌ 錯誤，鎖物件非 thread-safe|

---

## 🧮 十一、效能小結

|特性|`std::lock_guard`|`std::unique_lock`|
|---|---|---|
|鎖操作開銷|輕量|稍重（多狀態判斷）|
|適用場景|高頻短鎖、低延遲系統|條件變數、多鎖協調|
|性能差距|約 +5%~10%（依平台而異）|—|

---

## 🧱 十二、綜合範例（多 mutex + defer_lock）

``` cpp
#include <mutex>
#include <iostream>
std::mutex m1, m2;
void dual_lock()
{
std::unique_lock<std::mutex> lock1(m1, std::defer_lock);
std::unique_lock<std::mutex> lock2(m2, std::defer_lock);
std::lock(lock1, lock2); // 一次性避免死鎖
std::cout << "同時鎖住 m1 和 m2\n";
}
```

---

## 🧭 十三、總結 — 何時該用哪個？

| 狀況                       | 用法                                                     |
| ------------------------ | ------------------------------------------------------ |
| 只要簡單保護一個臨界區              | `std::lock_guard`                                      |
| 要解鎖再上鎖、延遲鎖               | `std::unique_lock`                                     |
| 要搭配 `condition_variable` | `std::unique_lock`                                     |
| 要同時鎖多個 mutex             | `std::unique_lock` + `std::defer_lock` + `std::lock()` |