### C/C++ 記憶體對齊操作
目錄
什麼是記憶體對齊
為什麼要進行記憶體對齊
如何進行記憶體對齊
  1️⃣ 透過編譯器指定對齊長度
  2️⃣ 使用記憶體分配函式

### 1. 什麼是記憶體對齊
在現代電腦系統中，記憶體是以「位元組（byte）」為單位劃分的。
理論上，任何型別的變數都可以從任意位址開始存取，但實際上，大多數處理器對基本資料型別在記憶體中的起始位址都有特定的限制。
通常會要求資料的首位址必須是某個數字 k 的倍數（常見是 4 或 8）。
這個 k 稱作「對齊邊界（alignment boundary）」，而變數儲存在該邊界上的行為，就是所謂的 記憶體對齊（Memory Alignment）。
為什麼要進行記憶體對齊
然記憶體是以位元組為單位存取的，但大多數處理器實際上是以「多位元組」為單位存取資料的，例如：
- 2 位元組（16-bit）
- 4 位元組（32-bit）
- 8 位元組（64-bit）
- 甚至 16 或 32 位元組（支援 SIMD 指令集時）

這個最小的存取單位就叫做 記憶體訪問粒度（memory access granularity）。
舉個例子：
假設有一顆以 4 位元組為存取粒度的處理器（常見於 32-bit 系統），
它只能從「4 的倍數位址」開始抓取一個 int（4 bytes）型別的資料。

如果變數沒有對齊，例如一個 int 儲存在位址 0x0001 開始的位置，
那麼 CPU 為了讀取這 4 個 bytes，就必須：

1. 先從 0x0000 開始抓取一個 4-byte 區塊（但這裡有 1 byte 是多餘的）
2. 再從 0x0004 再抓一次，取出剩下需要的資料

這樣的「跨區塊讀取」不只多花時間，也浪費 CPU 計算資源。

因此，為了提升效能，資料結構（尤其是堆疊資料）應盡量對齊在自然邊界（natural boundary）上。
因為：
- ✅ 對齊的存取：只需一次記憶體讀取
- ❌ 未對齊的存取：可能要讀兩次、甚至更多次

如何進行記憶體對齊
1️⃣ 透過編譯器指定對齊長度

每個平台上的編譯器都有預設的「對齊係數（alignment modulus）」。
例如在 GCC 中，預設是 #pragma pack(4)，
可以使用以下語法修改：

``` cpp
#pragma pack(n)   // n 可以是 1、2、4、8、16
```

範例：

``` cpp
#pragma pack(8)
struct Data {
    char  a;
    int   b;
    double c;
};
#pragma pack()
```

💡 註：#pragma pack() 沒有參數時會恢復預設設定。
但這種方式有幾個限制：
1. 若要使用 AVX 指令集，必須達到 32 位元組對齊，#pragma pack 無法保證。
2. 不同 CPU 架構對非對齊存取的支援程度不同，有些嵌入式 CPU 若遇到未對齊存取，程式可能直接當機。

2️⃣ 使用記憶體分配函式

除了透過編譯器設定外，也可以用專門的記憶體分配函式，手動控制對齊方式。
官方定義如下：

``` c
/* Allocate SIZE bytes aligned to ALIGNMENT bytes. */
void *memalign(size_t __alignment, size_t __size)
    __THROW __attribute_malloc__ __wur;
```

範例：

``` cpp
#include <malloc.h>
#include <stdio.h>

int main() {
    int *ptr = (int *)memalign(32, 4 * 100); // 分配 400 bytes 的空間，並以 32 bytes 對齊
    if (ptr == nullptr) {
        printf("記憶體分配失敗\n");
        return -1;
    }
    printf("分配的位址：%p\n", ptr);
    free(ptr);
    return 0;
}
```

延伸補充：
在 C11 與 C++17 之後，標準庫也提供了更安全、可攜性更高的函式：
C11:

``` cpp
void *aligned_alloc(size_t alignment, size_t size);
```

C++17:

``` cpp
#include <cstdlib>
void* p = std::aligned_alloc(32, 1024);
```

這些函式都能確保返回的指標符合指定的對齊要求。

小結

| 方法                               | 優點                | 缺點               |
| -------------------------------- | ----------------- | ---------------- |
| `#pragma pack(n)`                | 簡單、直接、方便結構體設計     | 無法保證高對齊、跨平台兼容性差  |
| `memalign()` / `aligned_alloc()` | 可手動控制對齊邊界，支援 SIMD | 需要手動釋放、舊版系統可能不支援 |
