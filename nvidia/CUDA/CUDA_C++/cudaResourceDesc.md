
在 CUDA 程式設計中，`cudaResourceDesc`（CUDA 資源描述符）是一個關鍵的結構體（struct），主要用於 **Texture Object（紋理物件）** 和 **Surface Object（表面物件）** 的建立。

常在處理高效能運算或影像處理時用到它。它告訴 CUDA 你的資料「在哪裡」以及「長什麼樣子」。

---

## 1. 結構定義 (Structure Definition)

`cudaResourceDesc` 的設計非常精巧，它使用 `union`（聯集）來節省空間，因為一個資源在同一時間只能是其中一種型態。

``` C++
struct cudaResourceDesc {
    enum cudaResourceType resType;      // 資源類型
    union {
        struct {
            cudaArray_t array;          // CUDA array (常用於 2D/3D 紋理)
        } array;
        struct {
            cudaMipmappedArray_t mipmap; // Mipmapped array
        } mipmap;
        struct {
            void *devPtr;               // 裝置記憶體指標 (Linear memory)
            struct cudaChannelFormatDesc desc; // 通道描述 (如 float4, int1)
            size_t sizeInBytes;         // 總位元組數
        } linear;
        struct {
            void *devPtr;               // 帶有 Pitch 的裝置記憶體指標
            struct cudaChannelFormatDesc desc;
            size_t width;               // 寬度 (以元素數量計)
            size_t height;              // 高度 (以元素數量計)
            size_t pitchInBytes;        // 每一列的間距 (Pitch)
        } pitch2D;
    } res;
};
```

---

## 2. 資源類型 (`resType`)

你必須根據你的原始資料格式設定 `resType`。常見的選項包括：

| **常數**                           | **說明**                                                              |
| -------------------------------- | ------------------------------------------------------------------- |
| `cudaResourceTypeArray`          | 使用 `cudaMallocArray` 配置的 CUDA Array。**這是做 2D Texture 效能最好、最推薦的方式。** |
| `cudaResourceTypeLinear`         | 一般的線性記憶體（`cudaMalloc`），適合將一維陣列當作 1D Texture 使用。                     |
| `cudaResourceTypePitch2D`        | 使用 `cudaMallocPitch` 配置的 2D 記憶體，常用於需要頻繁在 Host/Device 間搬運的 2D 資料。    |
| `cudaResourceTypeMipmappedArray` | 用於多級漸遠紋理（Mipmapping）。                                               |

---

## 3. 使用範例：建立一個 2D 紋理物件

這是最典型的用法。你需要先填好 `cudaResourceDesc`，再搭配 `cudaTextureDesc`（定義如何取樣，如線性過濾、邊界處理等），最後呼叫 `cudaCreateTextureObject`。

``` C++
// 1. 準備資源描述符
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc)); // 務必先歸零，因為裡面有 union
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray; // 假設你已經配置好一個 cudaArray_t

// 2. 準備紋理描述符 (定義 sampling 方式)
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.readMode = cudaReadModeElementType;
texDesc.filterMode = cudaFilterModeLinear; // 線性插值

// 3. 建立紋理物件
cudaTextureObject_t texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

// 4. 在 Kernel 中使用
// tex2D<float>(texObj, u, v);
```

---

## 💡 Pro-tips

1. **記憶體對齊**：如果你使用 `cudaResourceTypeLinear` 或 `pitch2D`，指標 `devPtr` 必須符合裝置的對齊要求（通常透過 `cudaGetDeviceProperties` 檢查 `textureAlignment` 屬性）。
    
2. **安全性**：使用 `memset` 初始化結構體是非常重要的習慣，因為 `union` 成員若留有殘餘數值，可能會導致驅動程式解析錯誤。
    
3. **現代化 API**：比起舊有的 Texture Reference（全域變數綁定），使用 `cudaResourceDesc` 建立的 Texture Object 是 **綁定到 Handle 的**，這讓你的程式碼更容易模組化，也支援在同一個 Kernel 中處理動態數量的紋理。