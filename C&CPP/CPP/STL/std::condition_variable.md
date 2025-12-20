std::condition_variable 用法與範例
本篇介紹 C++ 的 std::condition_variable 用法，使用 std::condition_variable 的 wait 會把目前的執行緒 thread 停下來並且等候事件通知，而在另外一個執行緒裡我們可以使用 std::condition_variable 的 notify_one 或 notify_all 去發送通知那些正在等待的事件，這在多執行绪程式裡經常使用到，以下將開始介紹 std::condition_variable 的用法，並展示一些範例，建議閱讀以下文章前需先對建立 std::thread 多執行緒與std::mutex 鎖有一定程度的熟悉。

<span style="color:#ff0000; font-size:15px; font-weight:bold;">需要引入的標頭檔：&lt;condition_variable&gt;</span>

以下為 condition_variable 常用的成員函式與說明，
- wait：阻塞當前執行緒直到條件變量被喚醒
- notify_one：通知一個正在等待的執行緒
- notify_all：通知所有正在等待的執行緒

使用 std::condition_variable 的 wait 必須要搭配 std::unique_lock&lt;std::mutex&gt; 一起使用。

範例1. 用 notify_one 通知一個正在 wait 的執行緒
下面的例子是先開一個新的執行緒 worker_thread 然後使用 std::condition_variable 的 wait 事件的通知，
此時 worker_thread 會阻塞(block)直到事件通知才會被喚醒，
之後 main 主程式延遲個 5 ms 在使用 std::condition_variable 的 notify_one 發送，
之後 worker_thread 收到 來自主執行緒的事件通知就離開 wait 繼續往下 cout 完就結束該執行緒，

這裡主程式的延遲 5ms 是避免一開始執行緒還沒建立好來不及 wait 等待通知，主程式就先發送 notify_one 事件通知了，

``` cpp
// g++ std-condition_variable.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex m;
std::condition_variable cond_var;

void worker_thread()
{
    std::unique_lock<std::mutex> lock(m);
    std::cout << "worker_thread() wait\n";
    cond_var.wait(lock);

    // after the wait, we own the lock.
    std::cout << "worker_thread() is processing data\n";
}

int main()
{
    std::thread worker(worker_thread);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::cout << "main() notify_one\n";
    cond_var.notify_one();

    worker.join();
    std::cout << "main() end\n";
}
```

輸出：

``` bash
worker_thread() wait
main() notify_one
worker_thread() is processing data
main() end
```

本範例是學習了用 notify_one 通知單一個等待的執行緒，
下個範例要介紹的是 notify_all 用來通知所有正在等待的執行緒，

範例2. 用 notify_all 通知全部多個 wait 等待的執行緒
以下範例主要目的是建立5個執行緒並等待通知，
之後主程式執行go函式裡的cond_var.notify_all()去通知所有正在等待的執行緒，也就是剛剛建立的5個執行緒，
這5個執行緒分別收到通知後從wait函式離開，之後檢查ready變數為true就離開迴圈，
接著印出thread id然後結束該執行緒。

``` cpp
// g++ std-condition_variable2.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex m;
std::condition_variable cond_var;
bool ready = false;

void print_id(int id) {
    std::unique_lock<std::mutex> lock(m);
    while (!ready) {
        cond_var.wait(lock);
    }
    std::cout << "thread " << id << '\n';
}

void go() {
    std::unique_lock<std::mutex> lock(m);
    ready = true;
    cond_var.notify_all();
}

int main()
{
    std::thread threads[5];
    // spawn 5 threads:
    for (int i=0; i<5; ++i)
        threads[i] = std::thread(print_id,i);

    std::cout << "5 threads ready to race...\n";
    go();

    for (auto& th : threads)
        th.join();

    return 0;
}
```

輸出如下，可以看見這5個執行緒不按順序地收到通知並且各別印出thread id，

``` bash
5 threads ready to race...
thread 4
thread 1
thread 2
thread 3
thread 0
```

這個範例多使用了一個額外的 ready 變數來輔助判斷，也間接介紹了cond_var.wait的另一種用法，
使用一個 while 迴圈來不斷的檢查 ready 變數，條件不成立的話就cond_var.wait繼續等待，
等到下次cond_var.wait被喚醒又會再度檢查這個 ready 值，一直迴圈檢查下去，
這技巧在某些情形下可以避免假喚醒這個問題，
簡單說就是「cond_var.wait被喚醒後還要多判斷一個 bool 變數，一定要條件成立才會結束等待，否則繼續等待」。

而這邊的 while 寫法

``` cpp
while (!ready) {
    cond_var.wait(lock);
}
```

可以簡化寫成下面這個樣子，也就是 wait 的另一種用法，
多帶一個謂詞在第二個參數，關於這個寫法不熟悉可以看看這篇，

``` cpp
cond_var.wait(lock, []{return ready;});
```

因為 wait 內部的實作方法如下，等價於上面這種寫法，

``` cpp
template<typename _Predicate>
void wait(unique_lock<mutex>& __lock, _Predicate __p)
{
    while (!__p())
        wait(__lock);
}
```

範例3. wait 等待通知且有條件式地結束等待
上個範例簡單介紹了cond_var.wait帶入第二個參數的用法了，所以本範例來實際演練這個用法，

worker_thread裡的cond_var.wait第一參數傳入一個 unique_lock 鎖，
第二個參數傳入一個可(被)呼叫的物件，來判斷是否要停止等待；這個可(被)呼叫的物件的需要回傳一個 bool 變數，
如果回傳 true 的話，condition_variable 就會停止等待、繼續往下執行，
如果回傳 false 的話，則會重新開始等待下一個通知。
因此等價於 while (!pred()) { wait(lock); }

這邊要注意 main 裡是有一個 lock_guard 與 unique_lock，worker_thread 裡有一個 unique_lock。

``` cpp
// g++ std-condition_variable3.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex m;
std::condition_variable cond_var;
std::string data;
bool ready = false;
bool processed = false;

void worker_thread()
{
    // Wait until main() sends data
    std::unique_lock<std::mutex> lock(m);
    std::cout << "worker_thread() wait\n";
    cond_var.wait(lock, []{return ready;});

    // after the wait, we own the lock.
    std::cout << "worker_thread() is processing data\n";
    data += " after processing";

    // Send data back to main()
    processed = true;
    std::cout << "worker_thread() signals data processing completed\n";

    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lock.unlock();
    cond_var.notify_one();
}

int main()
{
    std::thread worker(worker_thread);

    data = "Example data";
    // send data to the worker thread
    {
        std::lock_guard<std::mutex> lock(m);
        ready = true;
        std::cout << "main() signals data ready for processing\n";
    }
    cond_var.notify_one();

    // wait for the worker
    {
        std::unique_lock<std::mutex> lock(m);
        cond_var.wait(lock, []{return processed;});
    }
    std::cout << "Back in main(), data = " << data << '\n';

    worker.join();
}
```

程式輸出結果如下：

``` bash
main() signals data ready for processing
worker_thread() wait
worker_thread() is processing data
worker_thread() signals data processing completed
Back in main(), data = Example data after processing
```

範例4. 典型的生產者與消費者的範例
在設計模式(design pattern)中，這是一個典型的生產者與消費者(producer-consumer)的例子，
範例裡有一位生產者每1秒生產了1個東西放到 condvarQueue 裡，
這個 condvarQueue 會在去通知消費者，消費者收到通知後從 queue 裡拿出這個東西來作事情。

``` cpp
// g++ std-condition_variable4.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>

class condvarQueue
{
    std::queue<int> produced_nums;
    std::mutex m;
    std::condition_variable cond_var;
    bool done = false;
    bool notified = false;
public:
    void push(int i)
    {
        std::unique_lock<std::mutex> lock(m);
        produced_nums.push(i);
        notified = true;
        cond_var.notify_one();
    }

    template<typename Consumer>
    void consume(Consumer consumer)
    {
        std::unique_lock<std::mutex> lock(m);
        while (!done) {
            while (!notified) {  // loop to avoid spurious wakeups
                cond_var.wait(lock);
            }
            while (!produced_nums.empty()) {
                consumer(produced_nums.front());
                produced_nums.pop();
            }
            notified = false;
        }
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m);
            done = true;
            notified = true;
        }
        cond_var.notify_one();
    }
};

int main()
{
    condvarQueue queue;

    std::thread producer([&]() {
        for (int i = 0; i < 5; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "producing " << i << '\n';
            queue.push(i);
        }
        queue.close();
    });

    std::thread consumer([&]() {
         queue.consume([](int input){
             std::cout << "consuming " << input << '\n';
         });
    });

    producer.join();
    consumer.join();
}
```

程式輸出結果如下：

``` bash
producing 0
consuming 0
producing 1
consuming 1
producing 2
consuming 2
producing 3
consuming 3
producing 4
consuming 4
```

使用上的小細節
看了很多範例，通知端執行緒 notify_one 通知前到底要不要加鎖？
如果要加鎖要加 unique_lock 還 lock_guard 呢？

我的經驗是
如果不需要修改共享變數，則 notify_one/notify_all 通知前不用加鎖，
示意如下：

``` bash
Thread A                Thread B
                        unique_lock lock(m)
                        cond.wait()
cond.notify_one()
```

如果有需要修改共享變數，則 notify_one/notify_all 通知前要加鎖，
加鎖的範圍不用包含到 cond.notify_one/cond.notify_all，
注意這邊的鎖是要保護共享資料，而不是 cond.notify_one/cond.notify_all，
示意如下：

``` bash
Thread A                Thread B
                        unique_lock lock(m)
                        cond.wait(lock, []{return ready;})
{
lock_guard lock(m)
ready = true
}
cond.notify_one()
```

這兩者有效能上的差異，這部分我們以後有機會來細說談談。

重點歸納
簡單歸納一下幾個重點，
等待的執行緒應有下列幾個步驟：

獲得 std::unique_lock 鎖，並用該鎖來保護共享變數。
下面三步驟或使用 predicate 的 wait 多載版本，
2-1. 檢查有沒有滿足結束等待的條件，以預防資料早已經被更新與被通知了。
2-2. 執行 wait 等待，wait 操作會自動地釋放該 mutex 並且暫停該執行緒。
2-3. 當 condition variable 通知時，該執行緒被喚醒，且該mutex自動地被重新獲得，該執行緒應該檢查一些條件決定要不要繼續等待。
通知的執行緒應有下列幾個步驟：

獲取一個 std::mutex (通常透過std::lock_guard來取得)。
在上鎖的範圍內完成變數的修改。
執行 std::condition_variable 的notify_one/notify_all (不需被該鎖包覆)。

參考
1. [std::condition_variable - cppreference.com](https://en.cppreference.com/w/cpp/thread/condition_variable)
2. [condition_variable - C++ Reference - cplusplus.com](http://www.cplusplus.com/reference/condition_variable/condition_variable/)
3. [邁向王者的旅途: \[C++\] Use std::condition_variable for Parallellism](https://shininglionking.blogspot.com/2018/08/c-use-stdconditionvariable-for.html)
4. [C++11 Thread 的 condition variable – Heresy’s Space](https://kheresy.wordpress.com/2014/01/09/c11-condition-variable/)
5. [C++/STL/ConditionVariable - 維基教科書，自由的教學讀本](https://zh.wikibooks.org/zh-tw/C%2B%2B/STL/ConditionVariable)
6. [Is this use of condition variable safe (taken from cppreference.com)](https://stackoverflow.com/questions/61104388/is-this-use-of-condition-variable-safe-taken-from-cppreference-com)

謂詞函數predicate相關文章
1. [C++ 具名要求： 謂詞 (Predicate) - cppreference.com](https://zh.cppreference.com/w/cpp/named_req/Predicate)
2. [函數對象 - 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/%E5%87%BD%E6%95%B0%E5%AF%B9%E8%B1%A1)
3. [C++ 標準程式庫的函式物件 | Microsoft Docs](https://docs.microsoft.com/zh-tw/cpp/standard-library/function-objects-in-the-stl?view=vs-2019)

這裡翻譯為述詞Predicate，是個傳回布林值的函式物件。

pthread 相關文章
1. [pthread_cond_wait 为什么需要传递mutex 参数？ - 知乎](https://www.zhihu.com/question/24116967)
2. [c++ - Calling pthread_cond_signal without locking mutex - Stack Overflow](https://stackoverflow.com/questions/4544234/calling-pthread-cond-signal-without-locking-mutex)
3. [pthread_cond_wait.c source code [glibc/nptl/pthread_cond_wait.c] - Woboq Code Browser](https://code.woboq.org/userspace/glibc/nptl/pthread_cond_wait.c.html)