在圖形處理單元（GPU）的架構中，**紋理過濾單元（Texture Filtering Unit, TFU）** 是負責將 2D 影像（紋理）映射到 3D 模型表面，並處理縮放、旋轉或透視變形時產生的像素計算的核心元件。
在處理電腦圖學或 GPU 加速計算時，如何高效地讀取並處理這些非連續性的記憶體存取是一大挑戰。以下是 TFU 的核心功能與技術細節：

---

## 核心功能：解決「鋸齒」與「模糊」

當一個 3D 物件距離鏡頭太近或太遠時，顯示器上的像素（Pixel）與紋理上的紋理元素（Texel）通常不會是一對一的關係。TFU 的存在就是為了透過演算法計算出最接近真實視覺的顏色值。

#### 1. 雙線性過濾 (Bilinear Filtering)

這是最基礎的處理方式。當一個像素落在四個紋理元素之間時，TFU 會對這四個點進行加權平均採樣。

- **公式：**
    
    $$Color = (1-u)(1-v)C_{00} + u(1-v)C_{10} + (1-u)vC_{01} + uvC_{11}$$
    
- **優點：** 消除明顯的方塊感。
    
- **缺點：** 當視角傾斜或距離過遠時，畫面會顯得模糊。
    

#### 2. 三線性過濾 (Trilinear Filtering)

基於 **Mipmapping** 技術。TFU 會在兩層不同解析度的紋理層級（Mip levels）之間進行雙線性插值。這解決了不同距離紋理切換時產生的明顯「斷層」現象。

#### 3. 各向異性過濾 (Anisotropic Filtering, AF)

這是現代 GPU 最重要的功能之一。當物體表面與視角呈大幅度傾斜時（例如賽道向遠方延伸），標準的過濾會導致遠處極度模糊。AF 會根據視角的斜率，在非正方形的區域內進行多次採樣（如 2x, 4x, 16x），大幅提升遠景的清晰度。

---

## TFU 在 GPU 管線中的位置

TFU 通常整合在 **紋理映射單元（Texture Mapping Unit, TMU）** 之中。其運作流程如下：

1. **位址計算：** 接收來自著色器（Shader）的 UV 座標。
    
2. **紋理快取（Texture Cache）：** 檢查所需的紋理數據是否已在 L1/L2 Cache 中。
    
3. **取樣與過濾：** TFU 執行硬體加速的加權計算。
    
4. **回傳結果：** 將計算後的色彩值回傳給像素著色器（Pixel Shader）進行最終混合。
    

---

## 技術觀點：效能與頻寬

對工程師而言，TFU 的效能指標通常與 **Texture Fillrate**（紋理填充率）掛鉤。

- **記憶體對齊：** TFU 偏好紋理數據以 Tiling（平鋪）格式儲存，而非線性格式，以增進空間局部性（Spatial Locality）。
    
- **硬體瓶頸：** 執行高倍數的各向異性過濾（16x AF）會增加對記憶體頻寬的壓力，因為它需要讀取更多的紋理元素。


在 GPU 高度並行的架構下，**記憶體頻寬 (Memory Bandwidth)** 往往是效能的終極瓶頸。紋理快取（Texture Cache）的設計核心，就是為了在非連續性的 UV 採樣中，盡可能提高資料的**空間局部性 (Spatial Locality)**。

以下是針對 GPU 紋理快取優化的幾個深度策略：

---

## 1. 紋理平鋪 (Texture Tiling / Swizzling)

在記憶體中，一般的影像通常是以列優先（Row-major）儲存，但這對 GPU 採樣極為不利。當 TFU 需要進行雙線性過濾時，它需要讀取上下相鄰的像素，但在 Row-major 中，這兩行在記憶體位址上可能相隔數千個位元組。

- **優化方案：** 使用 **Z-order curve (Morton order)** 或矩形平鋪（Tiling）。
    
- **原理：** 將 2D 紋理切成小方塊（例如 4x4 或 8x8 的 Tile），確保在空間上相鄰的像素，在實體記憶體位址上也彼此接近。這能大幅提升 L1 Texture Cache 的命中率（Hit Rate）。
    

---

## 2. Mipmapping 與 4x4 採樣對齊

當物體縮小時，直接在原始高解析度紋理採樣會導致 **Aliasing (鋸齒)** 且極度浪費頻寬（因為相鄰像素跳躍太大，快取完全失效）。

- **優化方案：** 預先生成 Mipmaps。
    
- **架構層面：** 現代 GPU 的紋理單元通常以 **2x2 Quad** 為單位進行運算。確保紋理解析度是 2 的冪次方 (Power of Two, POT)，能讓硬體更有效地映射到快取行（Cache Line）中。
    

---

## 3. 紋理壓縮技術 (ASTC / BC / ETC)

這不是單純的檔案壓縮，而是**硬體級別的隨機存取壓縮**。

- **主流格式：**
    
    - **BC (Block Compression / DXT):** PC 端主流。
        
    - **ASTC (Adaptive Scalable Texture Compression):** 行動端與現代 GPU 的標竿，支援彈性的塊大小。
        
- **優化原理：** 數據以壓縮形式存在於顯存與 L2 Cache 中，直到進入 TFU 前才解壓。這相當於**變相增加了顯存頻寬**與快取的有效容量（Effective Capacity）。
    

---

## 4. 減少採樣器狀態切換 (Bindless Textures)

傳統的圖形 API（如舊版 OpenGL）在切換紋理時需要頻繁修改狀態（Binding），這會導致 CPU 開銷增加，且可能造成 GPU 管線的氣泡（Stall）。

- **優化方案：** 使用 **Bindless Textures** (現代 API 如 Vulkan/Metal/DX12 的核心特點)。
    
- **效果：** 將紋理句柄（Handle）直接傳入 Shader，讓 GPU 可以像處理普通指標一樣處理紋理，避免快取頻繁失效。
    

---

## 5. 虛擬紋理與串流 (Virtual Texturing / Sparse Textures)

當場景中的紋理總量超過顯存容量時（例如開放世界遊戲），我們不能一次加載所有內容。

- **策略：** 類似於作業系統的虛擬記憶體。將紋理切成數個 **Pages (Tiles)**，只將攝影機視角內的 Page 載入顯存。
    
- **技術：** **Feedback Analysis**。GPU 在渲染時會回傳哪些 Tile 是缺失的（Page Fault），由 CPU 非同步加載，這能極大化快取的利用效率，避免載入無用的數據。
    

---

## 效能調校工具建議 (Profile Tools)

如果你在實作時遇到瓶頸，建議使用以下工具觀察 **Texture Cache Hit Rate**：

- **NVIDIA Nsight Graphics / Systems:** 可以精確看到 SM (Streaming Multiprocessor) 的紋理吞吐量。
    
- **AMD Radeon GPU Profiler (RGP):** 觀察紋理位址跨度是否過大導致快取崩潰（Cache Thrashing）。


在 CUDA 核心（Kernel）開發中，**紋理過濾單元 (TFU)** 是透過 **紋理記憶體 (Texture Memory)** 或 **紋理物件 (Texture Object)** 來存取的。這與一般的 `Global Memory` 存取不同，TFU 提供硬體級別的內插（Interpolation）與邊界處理，且擁有獨立的 **Texture Cache**。

在處理影像預處理（如 Resize, Affine Transform）時，使用 TFU 通常比手寫 Bilinear 插值快得多。

---

## 1. CUDA 紋理物件的初始化 (Host Side)

在 CUDA 中，你必須先將資料綁定到一個 `cudaTextureObject_t`。這就像是告訴 TFU：數據在哪裡、該如何過濾。

``` C++
// 1. 分配 CUDA Array (專為紋理優化的佈局)
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaArray_t cuArray;
cudaMallocArray(&cuArray, &channelDesc, width, height);
cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

// 2. 設定採樣器資源
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

// 3. 設定 TFU 行為 (Texture Desc)
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0]   = cudaAddressModeClamp;   // 邊界拉伸
texDesc.addressMode[1]   = cudaAddressModeClamp;
texDesc.filterMode       = cudaFilterModeLinear;  // 啟用 TFU 硬體線性過濾
texDesc.readMode         = cudaReadModeElementType; // 直接回傳原始型別
texDesc.normalizedCoords = 1;                     // 使用 0.0~1.0 的座標系統

// 4. 建立紋理物件
cudaTextureObject_t texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
```

---

## 2. 在 Kernel 中使用 TFU (Device Side)

在 Device 端，你不再使用索引 `data[y * w + x]`，而是呼叫 `tex2D<T>()`。這會觸發 TFU 硬體執行過濾。

``` C++
__global__ void transformKernel(cudaTextureObject_t texObj, float* output, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 將整數座標轉換為 0.0 ~ 1.0 的歸一化座標
        float u = (float)x / (float)width;
        float v = (float)y / (float)height;

        // TFU 會自動讀取周圍 4 個像素並完成插值計算
        // 這是在獨立於 CUDA Core 的硬體單元中完成的
        output[y * width + x] = tex2D<float>(texObj, u, v);
    }
}
```

---

## 3. 為什麼要用 TFU 而非 Global Memory？

在實作 **Spatial Transformer Networks (STN)** 或 **Image Warping** 時，TFU 有以下優勢：

1. **硬體加速插值：** 雙線性插值（Bilinear）需要 4 次讀取 + 多次乘加運算。TFU 在一個時脈週期內就能完成，且不占用 CUDA Core (ALU) 的運算資源。
    
2. **空間局部性快取：** 紋理快取是針對 2D 空間設計的。如果你的演算法會存取相鄰像素（如卷軸或旋轉），紋理快取的命中率會遠高於普通的 L1 Cache。
    
3. **自動邊界處理：** 不需要寫大量的 `if (x < 0 || x >= width)`，硬體會自動幫你 `Clamp` 或 `Wrap`，減少程式碼的分支預測壓力。
    

---

## 4. 效能優化小撇步

- **使用 `cudaArray`：** 務必將資料轉存入 `cudaArray` 而非 `Linear Memory`。`cudaArray` 會自動進行 **Tiling (Swizzling)**，讓相鄰像素在記憶體位址上也相鄰，這對 TFU 的效能至關重要。
    
- **唯讀限制：** 紋理記憶體在 Kernel 執行期間是唯讀的。如果你需要讀寫同一個 Buffer，則無法直接利用 TFU。
    
- **資料精度：** TFU 內部的插值運算通常是固定精度（Fixed-point），對於極高精度的科學計算，需確認是否符合需求（通常影像處理 32-bit float 已足夠）。