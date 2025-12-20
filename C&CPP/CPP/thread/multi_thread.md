🧩 一、基本概念
1. 為什麼要用多執行緒？

多執行緒讓程式能：

- 同時執行多項任務（例如接收 MQTT 資料 + 處理影像 + 寫入資料庫）
- 充分利用多核心 CPU
- 避免 I/O 阻塞導致主程式卡死

🚀 二、最基本的範例

``` cpp
#include <iostream>
#include <thread>

void task(int id) {
    std::cout << "Thread " << id << " is running\n";
}

int main() {
    std::thread t1(task, 1);
    std::thread t2(task, 2);

    t1.join(); // 等待執行緒結束
    t2.join();

    std::cout << "All threads finished.\n";
    return 0;
}
```

🔹 std::thread 代表一個執行緒。
🔹 join() 等待執行緒執行完畢。
🔹 若忘了 join() 或 detach()，程式會丟例外錯誤（terminated）。

⚙️ 三、detach 模式（背景執行）

``` cpp
#include <iostream>
#include <thread>
#include <chrono>

void backgroundTask() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Background thread done.\n";
}

int main() {
    std::thread t(backgroundTask);
    t.detach(); // 背景執行，主程式不等待

    std::cout << "Main thread ends early.\n";
    std::this_thread::sleep_for(std::chrono::seconds(3));
}
```

🔹 detach()：執行緒變成「孤兒」，自行運行直到結束。
⚠️ 一旦 detach()，你就無法再管理該執行緒（無法 join、無法安全地訪問共享資料）。

🧵 四、共享資料與互斥鎖（mutex）

多執行緒最大問題 → 資料競爭（data race）。
解法：使用 std::mutex 或 std::scoped_lock。

``` cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;
int counter = 0;

void add() {
    for (int i = 0; i < 1000; ++i) {
        std::scoped_lock lock(mtx); // 自動上鎖+解鎖
        counter++;
    }
}

int main() {
    std::thread t1(add);
    std::thread t2(add);
    t1.join();
    t2.join();

    std::cout << "Counter = " << counter << "\n";
}
```

🪢 五、條件變數（std::condition_variable）

讓執行緒「等待事件」發生後再執行，比如生產者–消費者模型：

``` cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::mutex mtx;
std::condition_variable cv;
std::queue<int> q;

void producer() {
    for (int i = 0; i < 5; ++i) {
        {
            std::scoped_lock lock(mtx);
            q.push(i);
            std::cout << "Produced: " << i << "\n";
        }
        cv.notify_one(); // 通知消費者
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer() {
    while (true) {
        std::unique_lock lock(mtx);
        cv.wait(lock, [] { return !q.empty(); });
        int v = q.front();
        q.pop();
        lock.unlock();

        std::cout << "Consumed: " << v << "\n";
        if (v == 4) break;
    }
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join();
    t2.join();
}
```

🧠 六、進階主題（建議逐步研究）

| 主題                       | 說明                                     |
| ------------------------ | -------------------------------------- |
| **thread pool（執行緒池）**    | 重複利用執行緒，避免頻繁建立/銷毀                      |
| **futures & async**      | 非同步任務返回結果（`std::future`、`std::async`）  |
| **atomic operations**    | 用於輕量級同步（例如 `std::atomic<int> counter`） |
| **lock-free 結構**         | 高效能多執行緒共享資料結構（難度高）                     |
| **std::jthread (C++20)** | 自動 join 的 thread wrapper，更安全           |


🧩 七、簡單 thread pool 範例

``` cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i)
            workers.emplace_back([this] { this->loop(); });
    }

    ~ThreadPool() {
        {
            std::scoped_lock lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto &t : workers)
            t.join();
    }

    void enqueue(std::function<void()> job) {
        {
            std::scoped_lock lock(mtx);
            jobs.push(std::move(job));
        }
        cv.notify_one();
    }

private:
    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock lock(mtx);
                cv.wait(lock, [this] { return stop || !jobs.empty(); });
                if (stop && jobs.empty())
                    return;
                job = std::move(jobs.front());
                jobs.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> jobs;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
};
```

使用：

``` cpp
int main() {
    ThreadPool pool(4);
    for (int i = 0; i < 8; ++i)
        pool.enqueue([i] {
            std::cout << "Job " << i << " running in thread "
                      << std::this_thread::get_id() << "\n";
        });
}
```