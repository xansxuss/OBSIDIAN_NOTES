### 1. C++ 指標基礎
1. 指標的概念
指標是存放記憶體地址的變數。用 * 表示指標類型，用 & 取得變數地址。

``` cpp
int a = 10;
int* p = &a;   // p 指向 a 的地址
std::cout << "a = " << a << ", *p = " << *p << std::endl;
```

輸出：

``` bash
a = 10, *p = 10
```

- int* p → p 是指向 int 的指標
- &a → 取得 a 的地址
- *p → 解參考（dereference）指標，取得 a 的值

2. nullptr 與指標初始化
- 不初始化指標可能造成野指標（wild pointer）。
- C++11 起建議使用 nullptr：

``` cpp
int* p1 = nullptr;  // 空指標
int* p2;             // 未初始化（危險）
```

3. 指標與陣列
陣列名本身就是指標：

```cpp
int arr[3] = {1, 2, 3};
int* p = arr;
std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
```

4. 指標與函式
- 可以透過指標修改函式外的變數：

``` cpp
void increment(int* x) { (*x)++; }
int main() {
    int a = 5;
    increment(&a);
    std::cout << a << std::endl; // 6
}
```

- 函式指標（function pointer）：

```cpp
int add(int a, int b) { return a + b; }
int (*fp)(int, int) = add;
std::cout << fp(2, 3) << std::endl; // 5
```

### 2. C++ 指標進階用法
1. 指標與動態記憶體
- 使用 new / delete（C++11 可用 unique_ptr / shared_ptr 取代）：

```cpp
int* p = new int(10); // 在堆上分配
std::cout << *p << std::endl;
delete p;             // 釋放記憶體
```

- 動態陣列：

``` cpp
int* arr = new int[5]{1,2,3,4,5};
delete[] arr;
```

1. 指標與多維陣列

``` cpp
int** mat = new int*[3]; // 建立 3x3 矩陣
for(int i=0; i<3; i++)
    mat[i] = new int[3]{0,1,2};

std::cout << mat[1][2] << std::endl; // 2

for(int i=0;i<3;i++)
    delete[] mat[i];
delete[] mat;
```

3. 指標與 const

``` cpp
int a = 10;
const int* p1 = &a; // 指向常數，不能改值，但可以改指標指向
int* const p2 = &a; // 常指標，指標不能改，但可以改值
const int* const p3 = &a; // 常指標指向常數
```

4. 指標與引用
```cpp
int a = 5;
int& r = a;     // 引用
int* p = &a;    // 指標
r = 10;         // 改變 a
*p = 20;        // 同樣改變 a
```

- 引用本質上是指向變數的別名，指標是變數的地址。

5. 指標運算
- 支援加減（依類型大小移動）

``` cpp
int arr[3] = {1,2,3};
int* p = arr;
p++; // 移到 arr[1]
std::cout << *p << std::endl; // 2
```

- 支援比較：

``` cpp
if(p < arr+3) { /* 在範圍內 */ }
```

### 3. C++ 指標特殊用法
1. 指標陣列

``` cpp
int a=1,b=2,c=3;
int* arr[] = {&a, &b, &c};
for(auto p : arr)
    std::cout << *p << " "; // 1 2 3
```

2. 指標到指標

``` cpp
int a = 10;
int* p = &a;
int** pp = &p;    // 指向指標的指標
std::cout << **pp << std::endl; // 10
```

- 可以用於多層資料結構或函式修改指標本身。

3. void* 泛型指標

``` cpp
void* ptr = &a;
int* ip = static_cast<int*>(ptr);
```

- void* 可以指向任何型別，但需要 cast 才能使用。

4. 指標與結構

``` cpp
struct Point { int x, y; };
Point p1 = {1,2};
Point* ptr = &p1;
ptr->x = 10; // 使用 -> 操作成員
```

5. 智慧指標（C++11+）

- unique_ptr → 單一擁有權
- shared_ptr → 多重擁有權
- weak_ptr → 避免循環引用

``` cpp
#include <memory>

auto p = std::make_unique<int>(42); // unique_ptr
auto sp = std::make_shared<int>(100); // shared_ptr
```

工程上推薦使用智慧指標代替裸指標管理資源。

6. 指標技巧/進階場景
   1. 零拷貝 buffer / DMA：GPU 或影像處理常用 pointer 操作。
   2. 函式指標陣列：實作 callback table。
   3. C API interfacing：很多舊 C library 都使用指標，C++ 封裝時會用到。
   4. 自訂記憶體池：手動管理記憶體地址，提升效能。

### 4. 指標常見陷阱

| 陷阱               | 說明                                  |
| ------------------ | ------------------------------------- |
| 野指標             | 未初始化的指標或已 delete 的指標      |
| 雙重 delete        | 對同一個 pointer delete 多次          |
| 指標越界           | array + pointer 運算超過範圍          |
| 指標與對象生命週期 | 指向 local stack 變數離開作用域後使用 |
| const 混淆         | 指向常數 vs 常指標 vs 常指向常數      |


指標類型、操作、進階、智慧指標、陷阱示意圖
![[pointer_type.png]]