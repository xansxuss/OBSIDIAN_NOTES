C++ std::thread 建立多執行緒用法與範例
本篇介紹 C++ 的 std::thread 建立多執行緒的用法教學，並提供一些入門的 std::thread C++ 範例程式碼，std::thread 建立執行緒算是多執行緒的基本必學，這邊把常用到的用法與範例紀錄一下。

在c++11 thread 出來之前, 跨平台開發執行緒程式一直需要依賴平台的 api，例如 Windows 要呼叫 CreateThread, Unix-like 使用 pthread_create 等等情形。c++11 thread 出來之後最大的好處就是開發者只需寫一種 thread，到各平台去編譯就行了，這當然還要編譯器支援c++11。

<span style="color:#ff0000; font-size:20px; font-weight:bold;">需要引入的標頭檔：&lt;thread&gt;

接下來就介紹簡單的 c++ thread 寫法，內容分為以下幾部分：

- 基本 std::thread 的用法
- std::thread 常用的成員函式
- 範例1. 建立新 thread 來執行一個函式，且帶入有/無參數
- 範例2. 建立新 thread 來執行一個類別函式
- 範例3. 建立新 thread 來執行 lambda expression
- 範例4. join 等待 thread 執行結束
- 範例5. detach 不等待 thread 執行結束
- std::thread 用陣列建立多個 thread
- std::thread 用 vector 建立多個 thread
- std::thread 參數傳遞使用傳參考的方法

### 基本 std::thread 的用法

c++ 最簡單的 std::thread 範例如下所示，呼叫 thread 建構子時會立即同時地開始執行這個新建立的執行緒，之後 main() 的主執行緒也會繼續執行，基本上這就是一個基本的建立執行緒的功能了。詳細說明請看後面的範例。

``` cpp
#include <iostream>
#include <thread>

void myfunc() {
    std::cout << "myfunc\n";
    // do something ...
}

int main() {
    std::thread t1(myfunc);
    t1.join();
    return 0;
}
```

std::thread 常用的成員函式
以下為 c++ std::thread 常用的成員函式，

- get_id(): 取得目前的執行緒的 id，回傳一個為 std::thread::id 的類型。
- joinable(): 檢查是否可join。
- join(): 等待執行緒完成。
- detach(): 與該執行緒分離，一旦該執行緒執行完後它所分配的資源會被釋放。
- native_handle(): 取得平台原生的native handle (例如Win32的Handle, unix的pthread_t)。

其他相關的常用函式有，

- sleep_for(): 停止目前執行緒一段指定的時間。
- yield(): 暫時放棄CPU一段時間，讓給其它執行緒。

範例1. 建立新 thread 來執行一個函式，且帶入有/無參數
以下例子為建立新 c++ thread 來執行一個函式，其中 t1 是呼叫無參數的 foo() 函式，而 t2 執行緒是呼叫 bar() 有參數的函式。

```cpp
// g++ std-thread1.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>

void foo() {
    std::cout << "foo\n";
}

void bar(int x) {
    std::cout << "bar\n";
}

int main() {
    std::thread t1(foo); // 建立一個新執行緒且執行 foo 函式
    std::thread t2(bar, 0); // 建立一個新執行緒且執行 bar 函式
    std::cout << "main, foo and bar now execute concurrently...\n"; // synchronize threads

    std::cout << "sleep 1s\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "join t1\n";
    t1.join(); // 等待 t1 執行緒結束
    std::cout << "join t2\n";
    t2.join(); // 等待 t2 執行緒結束

    std::cout << "foo and bar completed.\n";

    return 0;
}
```

輸出

``` bash
main, foo and bar now execute concurrently...
foo
bar
sleep 1s
join t1
join t2
foo and bar completed.
```

注意！在多執行緒程式中常常會互相存取某段程式碼、某個函式、某個變數，需要對這些程式碼進行上鎖，以確保同一時間只有某一個執行緒能進行存取。

範例2. 建立新 thread 來執行一個類別函式
c++ std::thread 的建構子可以傳入 class 類別的函式，如下範例所示，
AA::start 分別建立 t1、t2 兩個執行緒，而 t1 建構子帶入 AA::a1 類別函式，AA::a1 前面記得要加上&，第二參數代表的是哪個類別，之後的參數為帶入類別函式的參數就像 t2 這樣。

``` cpp
// g++ std-thread2.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>

class AA {
public:
    void a1() {
        std::cout << "a1\n";
    }

    void a2(int n) {
        std::cout << "a2 " << n << "\n";
    }

    void start() {
        std::thread t1(&AA::a1, this);
        std::thread t2(&AA::a2, this, 10);

        t1.join();
        t2.join();
    }
};

int main() {
    AA a;
    a.start();

    return 0;
}
```

輸出

``` bash
a1
a2 10
```

範例3. 建立新 thread 來執行 lambda expression
std::thread 的建構子也可以傳入來 lambda expression，如下範例所示：

``` cpp
auto f = [](int n) {
    // Do Something
};
std::thread t1(f, 3);
```

也可以寫成

``` cpp
std::thread t1([](int n) {
    // Do Something
};, 3);
```

範例4. join 等待 thread 執行結束
在 main 主執行緒建立 t1 執行緒後，主執行緒便繼續往下執行，
如果主執行緒需要等 t1 執行完畢後才能繼續執行的話就需要使用 join，
即等待 t1 執行緒執行完 foo 後主執行緒才能繼續執行，
否則主執行緒會一直卡(blocking)在 join 這一行。

``` cpp
#include <iostream>
#include <thread>
#include <chrono>

void foo() {
    this_thread::sleep_for(chrono::milliseconds(200));
    cout<<"foo";
}

int main() {
    std::thread t1(foo);
    cout<<"main 1";
    t1.join();
    cout<<"main 2";
    return 0;
}
```

範例5. detach 不等待 thread 執行結束
承上例子，如果主執行緒不想等或是可以不用等待 t1 執行緒的話，
就可以使用 detach 來讓 t1 執行緒分離，接著主執行緒就可以繼續執行，t1執行緒也在繼續執行，
在整個程式結束前最好養成好習慣確保所有子執行緒都已執行完畢，
因為在 linux 系統如果主執行緒執行結束還有子執行緒在執行的話會跳出個錯誤訊息。

``` cpp
#include <iostream>
#include <thread>
#include <chrono>

void foo() {
    this_thread::sleep_for(chrono::milliseconds(200));
    cout<<"foo";
}

int main() {
    std::thread t1(foo);
    cout<<"main 1";
    t1.detach();
    cout<<"main 2";
    return 0;
}
```

std::thread 用陣列建立多個 thread
這邊示範建立多個執行緒，這個例子是建立 3 個執行緒，用陣列的方式來存放 std::thread，寫法如下，

``` cpp
// g++ std-thread-array.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>

void foo(int n) {
    std::cout << "foo() " << n << "\n";
}

int main() {
    std::thread threads[3];

    for (int i = 0; i < 3; i++) {
        threads[i] = std::thread(foo, i);
    }

    for (int i = 0; i < 3; i++) {
        threads[i].join();
    }

    std::cout << "main() exit.\n";

    return 0;
}
```

結果輸出如下，

``` bash
foo() 1
foo() 0
foo() 2
main() exit.
```

std::thread 用 vector 建立多個 thread
這邊示範建立多個執行緒，與上述例子不同，這是我們改用 vector 來存放 std::thread，寫法如下，

``` cpp
// g++ std-thread-vector.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <vector>

void foo(int n) {
    std::cout << "foo() " << n << std::endl;
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < 3; i++) {
        threads.push_back(std::thread(foo, i));
    }

    for (int i = 0; i < 3; i++) {
        threads[i].join();
    }

    std::cout << "main() exit.\n";

    return 0;
}
```

結果輸出如下，

``` bash
foo() 0
foo() 2
foo() 1
main() exit.
```

std::thread 參數傳遞使用傳參考的方法
如果我今天有個 myfunc 參數傳遞方式為傳參考，內容如下，

``` cpp
void myfunc(int& n) {
    std::cout << "myfunc n=" << n << "\n";
    n+=10;
}
```

我希望透過建立另外一個執行緒去執行 myfunc，之後我要取得這個 myfunc 的運算結果，那我建立執行緒時如果寫 std::thread t1(myfunc, n); 這樣的話馬上編譯會出現錯誤，為什麼會這樣呢？原因是因為在 std::thread 的參數傳遞方式為傳值，要傳參考的話需要透過 std::ref 來輔助達成，所以程式就會寫成這樣，myfunc 與 myfunc2 的參數傳遞方式不同，可以看看這兩者之間的差異，

``` cpp
// g++ std-thread3.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
 
void myfunc(int& n) {
    std::cout << "myfunc n=" << n << "\n";
    n+=10;
}

void myfunc2(int n) {
    std::cout << "myfunc n=" << n << "\n";
    n+=10;
}
 
int main() {
    int n1 = 5;
    std::thread t1(myfunc, std::ref(n1));
    t1.join();
    std::cout << "main n1=" << n1 << "\n";
    
    int n2 = 5;
    std::thread t2(myfunc2, n2);
    t2.join();
    std::cout << "main n2=" << n2 << "\n";

    return 0;
}
```

結果輸出如下，

``` cpp
myfunc n=5
main n1=15
myfunc n=5
main n2=5
```

其他參考
1. [thread - C++ Reference](http://www.cplusplus.com/reference/thread/thread/)
2. [std::thread - cppreference.com](https://en.cppreference.com/w/cpp/thread/thread)
3. [Multithreading in C++ - GeeksforGeeks](https://www.geeksforgeeks.org/multithreading-in-cpp/)
4. [C++ std::thread | 菜鳥教程](https://www.runoob.com/w3cnote/cpp-std-thread.html)
5. [C++ 的多執行序程式開發 Thread：基本使用 – Heresy’s Space](https://kheresy.wordpress.com/2012/07/06/multi-thread-programming-in-c-thread-p1/)