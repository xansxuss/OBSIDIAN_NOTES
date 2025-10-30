std::mutex 用法與範例
本篇介紹 C++ 的 std::mutex 的用法教學，並提供一些入門的 std::mutex C++ 範例程式碼，std::mutex 算是多執行緒中常用到的基本功能，mutex 用來上鎖一段多執行緒會交互存取的程式區塊，確保同一時間內只有一個執行緒能夠存取這段程式區塊，避免程式發生不預期的意外狀況，這裡把常用到的用法與範例在這邊紀錄一下，建議閱讀以下文章前需先對建立 std::thread 多執行緒有一定程度的熟悉。

std::mutex 通常不單獨使用而是搭配使用，例如：[[std::lock_guard]]、[[std::unique_lock]]、[[std::scoped_lock(C++17)]]，其中最常搭配 std::lock_guard 一起使用。

<span style="color:#ff0000; font-size:15px; font-weight:bold;">需要引入的標頭檔：&lt;mutex&gt;</span>

<p><span style="font-weight:bold;">範例.</span></p>

多執行緒呼叫同一個函式(沒有 mutex 鎖)
以下範例是多執行緒最基本也是最常遇見的情形，main 建立了兩個執行緒並且會同時存取 print 函式的資源，
print 會將帶入的參數 c 字元印 n 次，且每次印完會將 g_count 全域變數的次數加1，print 函式最後再將這 g_count 全域變數印出來。
第一個執行緒為 t1 執行緒，會印出 10 個 A，
第二個執行緒為 t2 執行緒，會印出 5 個 B，
如果我們今天想讓 print 某一時間只能某個執行緒來執行存取的話，
我們來看看如果沒有 mutex 的保護臨界區，這個程式的輸出會是怎樣。

``` cpp
#include <iostream>
#include <thread>

using namespace std;

int g_count = 0;

int print(int n, char c) {
    for (int i = 0; i < n; ++i) {
        std::cout << c;
        g_count++;
    }
    std::cout << '\n';

    std::cout << "count=" << g_count << std::endl;
}

int main() {
    std::thread t1(print, 10, 'A');
    std::thread t2(print, 5, 'B');
    t1.join();
    t2.join();

    return 0;
}

```

如果沒上鎖的話，可能造成不預期的輸出，如下count=5A所示，t2 執行緒的 g_count 還沒來得及印完\n，另一個執行緒 t1 已經開始搶著印了。
另外補充一下，t1 與 t2 誰先執行並沒有一定誰先誰後，每次執行的結果都有可能不同。

``` cpp
BBBBB
count=5A
AAAAAAAAA
count=15
```

<p><span style="font-weight:bold;">範例.</span></p>

多執行緒呼叫同一個函式(有 mutex 鎖)
根據上面的範例進行延伸修改，因為這兩個執行緒都共同存取 g_count 這個全域變數，如果要讓執行結果符合預期的話，這邊需要上鎖，
已確保同一時間內只有一個執行緒能夠存取這個 g_count 全域變數，當有執行緒占用時，其它執行緒要存取該資源時，就會被擋住，
直到該資源被執行緒釋放後，才能被其它執行緒存取。

這裡我們在 print 函式裡使用 g_mutex.lock() 手動上鎖，
並且在 print 函式最後尾巴使用 g_mutex.unlock() 手動解鎖，

``` cpp
// g++ std-mutex.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

std::mutex g_mutex;
int g_count = 0;

int print(int n, char c) {
    // critical section (exclusive access to std::cout signaled by locking mtx):
    g_mutex.lock();
    for (int i = 0; i < n; ++i) {
        std::cout << c;
        g_count++;
    }
    std::cout << '\n';

    std::cout << "count=" << g_count << std::endl;
    g_mutex.unlock();
}

int main() {
    std::thread t1(print, 10, 'A');
    std::thread t2(print, 5, 'B');
    t1.join();
    t2.join();

    return 0;
}
```

輸出如下，這樣就達成我們的目的，符合我們預期的輸出了。
下個範例會介紹更智慧的寫法。

``` bash

AAAAAAAAAA
count=10
BBBBB
count=15
```

<p><span style="font-weight:bold;">範例. 使用 lock_guard 來上鎖與解鎖</span></p>

直接使用 std::mutex 的成員函式 lock/unlock 來上鎖是可以的，只是要注意 lock 要有對應的 unlock ，一旦沒有解鎖到程式很可能就會發生死鎖，
那有沒有比較智慧的寫法來避免這種忘記解鎖而照成死鎖的問題發生呢？
有的！答案就是配合 std::lock_guard 使用，學會用 std::lock_guard 就可以避免手動上鎖解鎖，進而減少在寫程式上出現死鎖的機會，
以下就來介紹 mutex 配合 lock_guard 來上鎖與解鎖，
根據前一個範例進行修改，將原本使用 g_mutex 上鎖與解鎖的動作，換成了 lock_guard，如下範例所示，
在 lock_guard 建構時帶入一個 mutex，就會自動將其 mutex 上鎖，而在 lock_guard 解構時會對其 mutex 解鎖，
簡單說就是「lock_guard 建構時對 mutex 上鎖，解構時對 mutex 解鎖」，
lock_guard 利用生命週期這概念來進行上鎖與解鎖，lock_guard 本身並不管理 mutex 的生命週期，也就是 lock_guard 生命週期結束不代表 mutex 生命週期也結束，
以下面這個例子為例，在進入 print 後將 g_mutex 帶入 lock_guard 建構時上鎖，之後離開 print 函式時，lock_guard 生命週期也隨之結束，lock_guard 進行解構時對 g_mutex 解鎖
lock_guard 詳細介紹與實作原理請看這篇。

``` cpp
// g++ std-mutex2.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

std::mutex g_mutex;
int g_count = 0;

int print(int n, char c) {
    // critical section (exclusive access to std::cout signaled by locking mtx):
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < n; ++i) {
        std::cout << c;
        g_count++;
    }
    std::cout << '\n';

    std::cout << "count=" << g_count << std::endl;
}

int main() {
    std::thread t1(print, 10, 'A');
    std::thread t2(print, 5, 'B');
    t1.join();
    t2.join();

    return 0;
}
```

輸出結果如下，效果跟前一個範例一樣

``` bash
AAAAAAAAAA
count=10
BBBBB
count=15
```

參考
[1] std::mutex - cppreference.com
https://en.cppreference.com/w/cpp/thread/mutex
[2] mutex - C++ Reference
http://www.cplusplus.com/reference/mutex/mutex/
[3] C++11 併發指南三(std::mutex 詳解) - IT閱讀
https://www.itread01.com/content/1546128929.html
完整且複雜的一篇