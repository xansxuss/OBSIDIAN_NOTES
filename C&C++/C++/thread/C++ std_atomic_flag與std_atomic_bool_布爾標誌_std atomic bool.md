### 1. `std::atomic_flag`

`std::atomic_flag` 是最簡單的標準原子型別，它代表一個布林旗標（Boolean flag），沒有拷貝建構函式與拷貝賦值運算子（=delete）。  
`std::atomic_flag` 物件只能有兩種狀態：**設定（set）** 或 **清除（clear）**。  
而且必須用 `ATOMIC_FLAG_INIT` 進行初始化，該初始化會將旗標設定為「清除」狀態（也就是說，這個旗標永遠以清除狀態開始）。範例如下：

``` cpp
std::atomic_flag flag=ATOMIC_FLAG_INIT;
``` 

旗標一旦初始化完成，只能執行三種操作：

- 銷毀（析構）
- 清除（`clear()`）
- 設定並查詢其先前的值（`test_and_set()`）

這三者分別對應：

- 析構函式
- `clear()` 函式
- `test_and_set()` 函式

`clear()` 與 `test_and_set()` 操作都可以指定一個**記憶體順序（memory order）**，  
其中：

- `clear` 是**儲存（store）**操作
- `test_and_set` 是**讀改寫（read-modify-write）**操作

關於記憶體順序的更多內容可參見 [C++ 線上文件](https://zh.cppreference.com/w/cpp/atomic/memory_order)。

在原子型別上的每個操作都可帶有一個可選的「記憶體順序」參數，用於指定所需的記憶體順序語意。

- **store（儲存）操作** 可使用：  
    `memory_order_relaxed`、`memory_order_release`、`memory_order_seq_cst`
- **load（載入）操作** 可使用：  
    `memory_order_relaxed`、`memory_order_consume`、`memory_order_acquire`、`memory_order_seq_cst`
- **read-modify-write（讀改寫）操作** 可使用：  
    `memory_order_relaxed`、`memory_order_consume`、`memory_order_acquire`、`memory_order_release`、`memory_order_acq_rel`、`memory_order_seq_cst`

預設的順序為 `memory_order_seq_cst`（最嚴格的順序一致性）。

---

C++ 線上手冊提供了一個使用 `std::atomic_flag` 實作自旋鎖（spinlock）的範例：

``` cpp
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>

std::atomic_flag lock = ATOMIC_FLAG_INIT;

void f(int n)
{
	for (int cnt = 0; cnt < 5; ++cnt) {
		while (lock.test_and_set(std::memory_order_acquire))  // 嘗試取得鎖
			; // 自旋等待
		std::cout << "Thread " << n << " count:" << cnt << std::endl;
		lock.clear(std::memory_order_release);               // 釋放鎖
	}
}

int main(int argc, char* argv[])
{
	std::vector<std::thread> v;
	for (int n = 0; n < 4; ++n) {
		v.emplace_back(f, n); // 使用參數初始化執行緒
	}
	for (auto& t : v) {
		t.join(); // 等待執行緒結束
	}

	system("pause");
	return 0;
}
```


執行結果如下：

![[std_atomic_flag_1.png]]

<!-- <img alt="" class="has" height="345" src="https://i-blog.csdnimg.cn/blog_migrate/6c7644f1af177c0ee5533b58cc164cfa.png" width="314" />  -->

由於 `std::atomic_flag` 功能受限，它甚至不能作為一般的布林旗標使用，  
因為它**不支援簡單的「不修改值」的查詢操作**。  
若只是想要一個可安全查詢與設定的原子布林變數，建議使用 `std::atomic<bool>`。

---

### 2. `std::atomic<bool>`

最基本的原子布林型別是 `std::atomic<bool>`（也可以使用別名 `std::atomic_bool`）。  
它比 `std::atomic_flag` 功能更完整，並且可以用非原子的 `bool` 進行初始化或賦值。

  ``` cpp
std::atomic&lt;bool&gt; b(true);
b = false;
```

需要注意的是：**原子型別的賦值操作返回值而非引用**。

![[std_atomic_flag_2.png]]

<!-- <img alt="" class="has" height="38" src="https://i-blog.csdnimg.cn/blog_migrate/850ec0b5dd1d15ed36e944b206d50179.png" width="360" />  -->

**与std::atomic_flag使用clear不同，std::atomic<bool>写(存储)操作是通过store来完成的&#xff1b;test_and_set也由exchange函数替代，它可以用新值替代原值，同时获取原值。std::atomic&lt;bool&gt;支持对值得无修改查询，通过隐式转换为普通得bool，或者调用load函数，load是一个载入操作。** 

``` cpp  
std::atomic<bool> b;
bool x = b.load(std::memory_order_require);
b.store(true);
x=b.exchange(false,std::memory_order_acq_rel);
```

在 **C++20** 中，`std::atomic` 模板甚至加入了類似「條件變數（condition variable）」的功能：

![[std_atomic_flag_3.png]]

<!-- <img alt="" class="has" height="103" src="https://i-blog.csdnimg.cn/blog_migrate/5a5c775f80dba58c265c2bbf4db123bf.png" width="517" /> -->

這意味著在新標準中，`std::atomic<bool>` 不僅能做原子存取，  
還能具備基本的等待／喚醒同步機制（例如 `wait()` / `notify_one()` / `notify_all()`）。