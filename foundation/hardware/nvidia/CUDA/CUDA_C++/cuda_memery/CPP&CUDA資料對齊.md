# C++ / CUDA 資料對齊 (Data Alignment)

## 簡介
資料對齊是現代電腦硬體計算中非常重要的特性。當資料自然對齊時，CPU 讀寫記憶體的效率最高。所謂自然對齊通常指資料位址是資料大小的倍數，例如在 32 位元架構中，若一個 4 位元組的資料位於 4 的倍數地址，存取效率最佳。

除了效能，資料對齊也是程式語言的假設條件。雖然高階語言會幫我們自動處理對齊問題，但在低階程式中，未對齊的資料存取可能會導致 **未定義行為**。

本文將介紹 C++ 與 CUDA 中的資料對齊概念、對齊要求，以及如何安全分配記憶體。

---

## C++ 資料對齊

### 對齊概念
當記憶體位址是 2 的冪次倍時，我們稱該位址是「位元組對齊 (byte-aligned)」。  
存取對齊資料的效率高，存取未對齊資料會浪費記憶體存取頻寬，甚至可能導致錯誤。

單位字節的存取總是對齊的，但大於 1 字節的資料，如果未對齊，C++ 標準假設會被破壞，可能出現未定義行為。

### 對齊要求與範例

```cpp
#include <cassert>
#include <iostream>

struct float4_4_t
{
    float data[4];
};

// 對齊到 32 bytes，對 SIMD 指令可能有用
struct alignas(32) float4_32_t
{
    float data[4];
};

// alignas(1) 不會降低原本的對齊需求
struct alignas(1) float4_1_t
{
    float data[4];
};

// 強制 1 byte 對齊，可能導致未定義行為
#pragma pack(push, 1)
struct alignas(1) float4_1_ub_t
{
    float data[4];
};
#pragma pack(pop)

int main()
{
    assert(alignof(float4_4_t) == 4);
    assert(alignof(float4_32_t) == 32);
    assert(alignof(float4_1_t) == 4);
    assert(alignof(float4_1_ub_t) == 1);

    assert(sizeof(float4_4_t) == 16);
    assert(sizeof(float4_32_t) == 32);
    assert(sizeof(float4_1_t) == 16);
    assert(sizeof(float4_1_ub_t) == 16);
}
```

記憶體分配對齊

在 GNU 系統中，malloc 或 realloc 回傳的位址預設為：
- 32 位元系統：8 byte 對齊
- 64 位元系統：16 byte 對齊
若要自訂對齊，可使用：
- alignas(T) → 靜態陣列對齊
- std::aligned_alloc → 動態記憶體對齊

範例：

``` cpp
#include <cstdlib>
#include <iostream>

int main()
{
    alignas(int) unsigned char buf[sizeof(int)];
    void* p = std::aligned_alloc(1024, sizeof(int));
    std::cout << "1024-byte 對齊地址: " << p << std::endl;
    free(p);
}
```

未定義行為
若資料對齊不正確，在陣列或動態緩衝區上建立物件可能發生 讀寫錯誤，尤其使用 reinterpret_cast 或未對齊記憶體地址增量時。

例如：

``` cpp
struct Bar
{
    char arr[3]; // 3 byte + 1 byte padding
    short s;     // 2 byte
};
```

- alignof(Bar) == 2
- sizeof(Bar) == 6
若記憶體對齊符合 alignof(Bar)，則存取每個成員都是安全的。

CUDA 資料對齊
全域記憶體的合併訪問 (Coalesced Access)
- CUDA 全域記憶體以 32、64 或 128 byte 的記憶體事務存取，必須 自然對齊（起始位址是事務大小的倍數）才能被單次讀取。
- 32 個執行緒組成的 warp 存取 4 byte 資料，若資料彼此相鄰且 32 byte 對齊，GPU 只需一次 32 byte 記憶體事務。
- 若資料未對齊，需額外事務，吞吐量下降。
- 運算能力 6.0 以上的 GPU，warp 記憶體存取會自動合併，但自然對齊仍可最大化效率。
字大小與對齊要求
全域記憶體支援讀寫大小：1、2、4、8、16 byte。
唯有大小符合且資料自然對齊，才會編譯為單一全域記憶體指令。
非自然對齊的 8/16 byte 可能讀取錯誤，需特別注意。
CUDA 對齊範例

``` cpp
// 使用內建向量類型或 __align__ 指定結構對齊
struct __align__(4) int8_3_4_t
{
    int8_t x, y, z;
};

struct __align__(16) float3_16_t
{
    float x, y, z;
};
```

結論
- 永遠使用 1、2、4、8、16 byte 的字，並確保自然對齊。
- CUDA 動態分配的記憶體至少 256 byte 對齊，可安全存放同類型字序列。
- 若同一塊記憶體存放不同型別或非自然對齊資料，必須小心維護對齊，否則可能錯誤或效能低下。

參考資料
[C++ Data Alignment](https://leimao.github.io/blog/CPP-Data-Alignment/)
[CUDA Device Memory Access](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#device-memory-accesses)
[Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
[alignas specifier](https://en.cppreference.com/w/cpp/language/alignas.html)
[std::aligned_alloc](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)