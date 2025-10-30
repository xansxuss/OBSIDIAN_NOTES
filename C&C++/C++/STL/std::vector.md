C++ std::vector 用法與範例
C++ vector 是一個可以改變陣列大小的序列容器。C++ vector 是陣列的升級版，主要因為 vector 能高效地對記憶體進行管理以及動態增長。vector 其實就是將陣列和方法封裝形成的一個類別。

vector 底層實現是一個連續記憶體空間，當容量不夠的時候就會重新申請空間，並把原本資料複製或搬移到新的空間。

vector 的容器大小可以動態增長，但是並不意味著每一次插入操作都進行 reallocate。記憶體的分配與釋放耗費的資源是比較大的，因此應該減少它的次數。這也就意味著容器的容量(capacity)與容器目前容納的大小(size)是不等的，前者應大於後者。

vector 分配新的空間時，容量(capacity)可能為原有容量的 2 倍或者原有容量的 1.5 倍，各個編譯器可能不同，稍後會介紹。

![[std-vector.png]]

c++ std vector
以下 C++ vector 內容將分為這幾部份，

- vector 常用功能
- vector 初始化
- 存取 vector 元素的用法
- 在 vector 容器尾巴新增元素的用法
- 在 vector 容器尾巴移除元素的用法
- vector for 迴圈遍歷
- vector 實際範例
- vector 使用 [] operator 與 at() 的差異
- vector size() 與 capacity() 的差異
- vector reserve() 預先配置容器大小的用法
- vector shrink_to_fit() 收縮的用法
- vector resize() 的用法
- 兩個 vector 串連
- vector 的優點與缺點
- vector 使用小技巧

<span style="color:#ff0000; font-size:20px; font-weight:bold;">C++ 要使用 vector 容器的話，需要引入的標頭檔：&lt;vector&gt;</span>

vector 常用功能
以下為 C++ std::vector 內常用的成員函式，
- push_back：把元素加到尾巴，必要時會進行記憶體配置
- pop_back：移除尾巴的元素
- insert：插入元素
- erase：移除某個位置元素, 也可以移除某一段範圍的元素
- clear：清空容器裡所有元素
- size：回傳目前長度
- empty：回傳是否為空
- [i]：隨機存取索引值為i的元素，跟陣列一樣索引值從 0 開始
- at(i)：隨機存取索引值為i的元素，跟上面 operator[] 差異是 at(i) 會作邊界檢查，存取越界會拋出一個例外
- reserve()：預先配置大小

vector 初始化
這邊介紹 C++ 幾種 vector 初始化，
這樣是宣告一個 int 整數類型的 vector，裡面沒有任何元素(空)，size 為 0 表示 vector 容器中沒有任何元素，capacity 也是 0，

``` cpp
#include <vector>
using namespace std;

int main() {
    vector<int> v;
    return 0;
}
```

先宣告一個空的 vector，再透過 push_back 將資料一直推進去，

``` cpp
vector<int> v;
v.push_back(1);
v.push_back(2);
v.push_back(3);
```

你也可以寫成一行，但這語法需要編譯器 C++11 支援，

``` cpp
vector<int> v = {1, 2, 3};
```

或者是這樣寫也可以，

``` cpp
vector<int> v({1, 2, 3});
```

假如要從另外一個 vector 容器複製資料過來當作初始值的話可以這樣寫，

``` cpp
vector<int> v1 = {1, 2, 3};
vector<int> v2 = v1;
```

或者這樣，

``` cpp
vector<int> v1 = {1, 2, 3};
vector<int> v2(v1);
```

也可以從傳統陣列裡複製過來當作初始值，

``` cpp
int n[3] = {1, 2, 3};
vector<int> v(n, n+3);
```

不想複製來源 vector 全部的資料，想要指定複製 vector 的範圍的話也可以，例如我要複製 v1 vector 的第三個元素到倒數第二個元素，

``` cpp
vector<int> v1 = {1, 2, 3, 4, 5};
vector<int> v2(v1.begin()+2, v1.end()-1); // {3, 4}
```

如果是指定複製傳統陣列的範圍的話，可以這樣寫，

``` cpp
int n[5] = {1, 2, 3, 4, 5};
vector<int> v(n+2, n+4); // {3, 4}
```

存取 vector 元素的用法
vector 用 [] 來隨機存取元素，第一個元素為 v[0]，索引值是 0，第二個元素為 v[1]，索引值是 1，依此類推，[] 不只可以讀取元素也可以用來修改元素，例如 v[0] = 4 像下面範例這樣寫，

``` cpp
vector<int> v = {1, 2, 3};
cout << v[0] << "\n"; // 1
cout << v[1] << "\n"; // 2
v[0] = 4;
cout << v[0] << "\n"; // 4
```

在 vector 容器尾巴新增元素的用法
在前面已經有稍微透漏了怎麼新增 vector 元素的方法了，沒錯就是用 push_back() 這個方法，它會把元素加在 vector 容器的尾巴，
先宣告一個空的 vector，再透過 push_back 將資料一直推進去，

``` cpp
vector<int> v = {1, 2, 3};
v.push_back(4); // {1, 2, 3, 4}
v.push_back(5); // {1, 2, 3, 4, 5}
v.push_back(6); // {1, 2, 3, 4, 5, 6}
```

在 vector 容器尾巴移除元素的用法
移除 vector 容器尾巴的元素用 pop_back()，一次只能從尾端移除一個元素，不能指定移除的數量，

``` cpp
vector<int> v = {1, 2, 3};
v.pop_back(); // {1, 2}
v.pop_back(); // {1}
```

vector for 迴圈遍歷
以下介紹 vector 的 for 的三種遍歷寫法，第一種是一般很常見的寫法，

``` cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> vec({1, 2, 3});
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << "\n";
    return 0;
}
```

第二種是使用 iterator 迭代器來印出 vector 內所有內容，其中 vector<int>::iterator it 可以簡化寫成 auto it = vec.begin() 這樣

``` cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> vec({1, 2, 3});
    // for (vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    // or
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";
    return 0;
}
```

第三種是個很方便的寫法，c++11 才有支援，適合追求快速(懶惰)的人，相較於第一種的優點是不用多寫陣列索引去存取，直接就當變數使用，

``` cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> vec({1, 2, 3});
    for (auto &v : vec) {
        cout << v << " ";
    }
    cout << "\n";
    return 0;
}
```

vector 實際範例
實際的範例寫起來像這樣，

``` cpp
// g++ std-vector.cpp -o a.out -std=c++11
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> vec; // 宣告一個放 int 的 vector

    vec.push_back(1); // {1}
    vec.push_back(2); // {1, 2}
    vec.push_back(3); // {1, 2, 3}
    vec.push_back(4); // {1, 2, 3, 4}
    vec.push_back(5); // {1, 2, 3, 4, 5}

    vec.pop_back(); // {1, 2, 3, 4} 移除尾巴的值
    vec.pop_back(); // {1, 2, 3}

    cout << "size: " << int(vec.size()) << endl; // 印出大小

    // 印出 vector 內所有內容
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << "\n";

    // 用 iterator 來印出 vector 內所有內容
    for (vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    vec[0] = 99; // {99, 2, 3} 改變裡面的值

    vector<int>::iterator it = vec.begin();
    vec.insert(it+2, 6); // {99, 2, 6, 3}
    vec.erase(it+2); // {99, 2, 3}

    // 快速(懶人)寫法, c++11 才支援
    for (auto &v : vec) {
        cout << v << " ";
    }
    cout << "\n";

    return 0;
}
```

輸出內容如下：

``` bash
size: 3
1 2 3 
1 2 3 
99 2 3
```

vector 使用 [] operator 與 at() 的差異
C++ std::vector 提供了 [] operator 的方式讓我們在取得元素時就像 C-style 陣列那樣使用，另外 std::vector 還提供了 at() 這個方法也是可以取得元素，那這兩種方式到底有什麼差別？

[] operator 在回傳元素時是不會作任何的邊界檢查，而在 at() 取得元素時會作邊界的處理，如果你存取越界時 std::vector 會拋出一個 out_of_range 例外，例外處理可以參考我的另一篇文章， 所以 at() 提供了較為安全的存取方式。

[] operator 隨機存取與 at() 各有好壞，使用上時挑選符合需求的方式。

vector size() 與 capacity() 的差異
vector 使用 size() 是取得目前 vector 裡的元素個數，vector 使用 capacity() 是取得目前 vector 裡的預先配置的空間大小，當容量(capacity)空間不夠使用時 vector 就會重新申請空間，容量(capacity)會增加為原來的 2 倍或 1.5 倍，例如：1、2、4、8、16、32 增長下去，各個編譯器可能不同，來看看下面範例，

``` cpp
vector<int> v;
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
```

輸出結果如下，從以下的輸出可以發現在我使用的 clang 編譯器中 capacity 是以 1、2、4、8 兩倍的方式增長下去，

``` bash
size=0, capacity=0
size=1, capacity=1
size=2, capacity=2
size=3, capacity=4
size=4, capacity=4
size=5, capacity=8
```

vector reserve() 預先配置容器大小的用法
vector 使用 reserve() 是預留空間的意思，如果我們一開始就知道容器的裡要放置多少個元素的話，可以透過 reserve() 來預先配置容器大小，這樣可以減少一直配置記憶體的機會。

如下例所示，先宣告一個 int 的 vector，假設我想要預先配置好 5 個大小的話可以這樣寫 vector.reserve(5)，這樣會預留 5 個元素的空間，使用 capacity() 會得到 5，但裡面還沒有任何元素所以使用 size() 會得到 0，之後用 push_back 將元素推進去，然後我們來觀察看看 size 與 capacity 的變化，

之後將 vector push_back 2 個元素進去，再次使用 capacity() 還是 5，而使用 size() 會得到 2，

``` cpp
vector<int> v;
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.reserve(5);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
```

輸出如下，

``` bash
size=0, capacity=0
size=0, capacity=5
size=2, capacity=5
```

那 vector reserve 預留 5 個元素的空間後，之後使用超過 5 個元素 capacity 會發生什麼變化呢？

``` cpp
vector<int> v;
v.reserve(5);
v.push_back(1);
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
v.push_back(1);
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
```

輸出如下，可以發現當 vector 的元素超過預留的 5 個元素時，會將容量增長為原本 capacity 的兩倍，

``` bash
size=2, capacity=5
size=5, capacity=5
size=6, capacity=10
```

在 vector 建構子帶入數量 n 會初始化 n 個元素且預設初始值為 0，所以使用 size() 會回傳 n，跟上述的 reserve() 用途是不一樣的，詳見下列範例，

``` cpp
vector<int> v(2);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1); // {0, 0, 1}
v.push_back(2); // {0, 0, 1, 2}
v.push_back(3); // {0, 0, 1, 2, 3}
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}
cout << "\n";
```

輸出如下，一開始在 vector 建構子帶入的數量 2 會初始化 2 個元素，

``` bash
size=2, capacity=2
size=5, capacity=8
0 0 1 2 3
```

vector shrink_to_fit() 收縮的用法
呈上述 reserve 例子，這時 vector 再使用 shrink_to_fit 成員函式的話，會釋放（free）那些尚未使用的空間，

``` cpp
vector<int> v;
v.reserve(5);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.push_back(1);
v.push_back(1);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.shrink_to_fit();
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
```

輸出如下，可以發現 vector 使用 shrink_to_fit() 後，容量 capacity 收縮回目前元素的 2 個數大小，

``` cpp
size=0, capacity=5
size=2, capacity=5
size=2, capacity=2
```

如果 size() 剛好等於 capacity() 的話，那麼使用 shrink_to_fit() 則不會有空間被釋放。

vector resize() 的用法
vector 使用 resize 跟 reserve 不太一樣，resize 變大時會把多的元素補 0，例如：

``` cpp
vector<int> v;
v.resize(5);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}
cout << "\n";
```

輸出如下，印出來的元素都是 0，

```bash
size=5, capacity=5
0 0 0 0 0
```

resize 如果要順便指定元素初始值的話，可以將初始值帶入 resize() 的第二個引數，像這樣寫，

``` cpp
vector<int> v;
v.resize(5, 10);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}
cout << "\n";
```

輸出如下，這些新增的元素初始值都設成 10 了，

``` bash
size=5, capacity=5
10 10 10 10 10
```

如果 resize 的大小超過 capacity 容量大小會怎麼樣呢？

```cpp
vector<int> v = {1, 2, 3};
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
v.resize(5);
cout << "size=" << v.size() << ", capacity=" << v.capacity() << "\n";
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}
cout << "\n";
```

輸出如下，原本的 1, 2, 3 元素有保留以外，剩下新增的元素補 0，

``` bash
size=3, capacity=3
size=5, capacity=6
1 2 3 0 0
```

兩個 vector 串連
這邊介紹 C++ 如何將兩個 vector 串連，最簡單方式就是寫迴圈一個一個複製過去，這邊要介紹更方便的方法，就是用 std::copy，例如：兩個 vector 分別為 src 與 dst，那麼可透過 std::copy 將 src 的內容複製到 dst 後面，大致用法如下，

1
std::copy(src.begin(), src.end(), std::back_inserter(dst));
要使用 std::copy 需要引入的標頭檔 <algorithm>，使用 std::back_inserter 需要引入的標頭檔 <iterator>。

以下實際範例示範兩個 vector 串連，將 vec1 的內容複製到 vec2 後面，然後再將 vec2 印出來，

``` cpp
// g++ std-vector2.cpp -o a.out -std=c++11
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

int main() {
    vector<int> vec1 = {1, 2, 3};
    vector<int> vec2 = {4, 5, 6};

    std::copy(vec1.begin(), vec1.end(), std::back_inserter(vec2));

    for (auto &v : vec2) {
        cout << v << " ";
    }
    cout << "\n";
    return 0;
}
```

輸出結果如下，

``` bash
4 5 6 1 2 3
```

vector 的優點

- 宣告時可以不用確定大小
- 節省空間
- 支援隨機訪問[i]

vector 的缺點

- 進行插入刪除時效率低
- 只能在末端進行 pop 和 push

vector 使用小技巧
使用 vector 時提前分配足夠的空間以避免不必要的重新分配記憶體和搬移資料

開發者喜歡使用 vector，因為他們只需要往向容器中添加元素，而不用事先操心容器大小的問題。但是由一個容量為 0 的 vector 開始往裡面持續添加元素會花費大量的運行性能。如果你預先就知道 vector 需要保存多少元素，就應該提前為其分配足夠的空間reserve()以提升性能。