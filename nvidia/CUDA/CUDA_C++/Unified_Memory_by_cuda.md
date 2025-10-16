1. 什麼是 Unified Memory（統一記憶體）？

在 CUDA 6 中，NVIDIA 引入了 CUDA 歷史上最重要的程式設計模型改進之一 —— Unified Memory（統一記憶體，以下簡稱 UM）。

在一般的 PC 系統中，CPU 與 GPU 的記憶體是物理上分離的，兩者透過 PCIe 匯流排進行資料交換。
在 CUDA 6.0 之前，程式設計師必須非常清楚這一點，並在程式碼中手動處理記憶體的分配與資料的傳輸，也就是：

在 CPU 與 GPU 各自分配記憶體空間

使用 cudaMemcpy 進行資料拷貝（Host ↔ Device）

範例比較
傳統 CPU 寫法：

``` cpp
void sortfile(FILE *fp, int N)                    
{                                                   
    char *data = (char*) malloc(N);                               
    fread(data, 1, N, fp);                                 
    qsort(data, N, 1, compare);                          
    usedata(data);                                        
    free(data);                                           
}
```

使用 Unified Memory 的 GPU 寫法：

``` cpp
void sortfile(FILE *fp, int N)
{
    char *data;
    cudaMallocManaged(&data, N);
  
    fread(data, 1, N, fp);
  
    qsort<<<...>>>(data, N, 1, compare);
    cudaDeviceSynchronize();  // 等待 GPU 執行完成
  
    usedata(data);
    cudaFree(data);
}
```


可以發現兩段程式碼幾乎一模一樣。
唯一的差別是：
1. 使用 cudaMallocManaged() 分配記憶體（取代 malloc()）
2. GPU 執行完後需要 cudaDeviceSynchronize() 同步
3. 不再需要手動的 Host ↔ Device 拷貝

在 CUDA 6.0 之前，對應的程式會長這樣：

``` cpp
void sortfile(FILE *fp, int N)    
{
    char *h_data, *d_data;                                        
    h_data = (char*) malloc(N); 
    cudaMalloc(&d_data, N); 
 
    fread(h_data, 1, N, fp);   
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
 
    qsort<<<...>>>(d_data, N, 1, compare);
 
    cudaMemcpy(h_data, d_data, N, cudaMemcpyDeviceToHost);
     
    usedata(h_data);
    free(h_data); 
    cudaFree(d_data);
}
```

Unified Memory 的優點
   1. 簡化記憶體管理：不再需要分別分配 host/device 記憶體。
   2. CPU/GPU 共用同一個指標：大幅減少程式碼量與錯誤風險。
   3. 語言整合更自然：與原生 C/C++ 語法更一致。
   4. 更方便的程式移植：減少不同平台間的修改成本。

2. Deep Copy 的情境

前面看起來 UM 好像只是減少了幾行程式碼，但當我們面對更複雜的資料結構時，它的威力才真正顯現。
假設我們有以下結構體：

``` cpp
struct dataElem {
    int data1;
    int data2;
    char *text;
};
```

在沒有 UM 的情況下，要將它傳給 GPU，就得這樣寫：

```cpp
void launch(dataElem *elem) 
{
    dataElem *d_elem;
    char *d_text; 
 
    int textlen = strlen(elem->text); 
 
    cudaMalloc(&d_elem, sizeof(dataElem));
    cudaMalloc(&d_text, textlen);

    cudaMemcpy(d_elem, elem, sizeof(dataElem), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, elem->text, textlen, cudaMemcpyHostToDevice);

    // 更新 GPU 端的 text 指標
    cudaMemcpy(&(d_elem->text), &d_text, sizeof(d_text), cudaMemcpyHostToDevice); 
 
    kernel<<<...>>>(d_elem);
}
```

這樣非常繁瑣。
而使用 Unified Memory 之後，只需要：

``` cpp
void launch(dataElem *elem) 
{   
    kernel<<<...>>>(elem); 
}
```

是不是清爽多了？

對於像是 鏈結串列（linked list） 這種多層指標結構，在沒有 UM 的情況下要在 GPU 上處理幾乎是惡夢；但有了 UM：
   1. 可以在 CPU/GPU 間直接傳遞整個鏈結串列
   2. 可以任意端修改節點內容
   3. 不必擔心記憶體同步與對應問題
雖然在 UM 出現之前，也可以透過 Zero-Copy Memory（pinned host memory） 來達到類似效果，但 pinned memory 的存取速度受限於 PCIe 頻寬，效能仍然有限。UM 則能在許多情況下帶來更好的效能。

3. 在 C++ 中使用 Unified Memory

現代 C++ 通常不直接用 malloc()，而是透過 new 進行封裝。
我們可以覆寫 operator new 與 operator delete 來讓類別自動使用 UM：

``` cpp
class Managed {
public:
    void* operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaFree(ptr);
    }
};
```

任何繼承 Managed 的類別，都可以自動使用 Unified Memory。
舉例來說，一個自訂的 String 類別：

``` cpp
class String : public Managed {
    int length;
    char *data;

public:
    // 複製建構子實現 pass-by-value
    String(const String &s) {
        length = s.length;
        cudaMallocManaged(&data, length);
        memcpy(data, s.data, length);
    }
};
```

這樣就能讓物件自動在 Unified Memory 上配置，並在 CPU/GPU 間共用。

4. Unified Memory vs Unified Virtual Addressing（UVA）

別搞混這兩個概念。
UVA（Unified Virtual Addressing） 早在 CUDA 4.0 就出現了，UM 是建立在 UVA 之上，但它們並不是同一件事。

UVA 的目標是讓以下三種記憶體共用同一虛擬位址空間：

1. GPU device memory
2. Shared memory（on-chip）
3. Host memory

注意：thread-local 的記憶體（register、local memory）不屬於 UVA 範圍。

UVA 只是「統一位址空間」，但不會自動幫你搬資料。
UM 則是進一步在 runtime 期間由 CUDA 自動進行頁面遷移（page migration），達到真正的「記憶體共享」效果。

5. 常見疑問

Q1：Unified Memory 會消除 CPU 與 GPU 之間的拷貝嗎？
→ 不會。只是這部分的拷貝由 CUDA runtime 自動處理，對程式設計師透明而已。
拷貝的開銷仍然存在，也仍需注意 race condition 與資料一致性問題。
簡單說，如果你已經很會手動優化記憶體搬移，UM 不會更快，但它會讓開發更輕鬆。

Q2：既然還是會拷貝資料，為什麼需要 Compute Capability 3.0 以上的 GPU？
→ 因為實際的 UM 實作依賴於硬體的虛擬記憶體與頁遷移機制。

從 Pascal 架構開始，GPU 提供 49-bit 虛擬位址 與 按需頁遷移（on-demand page migration）：

- GPU 可以直接尋址整個系統記憶體與多張 GPU 的記憶體空間
- 支援跨 GPU 的記憶體共用與系統層級原子操作
- 支援「out-of-core」運算（資料量超過 GPU 實體記憶體）

這些特性讓 GPU 可以像 CPU 一樣，在需要時才載入資料頁（page fault driven），更有效率地使用記憶體資源。

簡單總結：

Unified Memory = 自動搬資料的 Unified Virtual Addressing + Page Migration + Runtime 管理。
在開發體驗上，UM 讓 CUDA 更接近一般 C++ 的程式設計邏輯，也讓多 GPU 或大規模資料處理變得更簡潔。