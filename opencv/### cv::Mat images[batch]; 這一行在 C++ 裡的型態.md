### cv::Mat images[batch]; 這一行在 C++ 裡的型態 (type)

🧠 型態分析
假設：

``` cpp
int batch = 8;
cv::Mat images[batch];
```

那 images 的型態是：

``` cpp
cv::Mat [batch]
```

也就是「一個長度為 batch 的 cv::Mat 陣列」。

📦 元素型態
陣列中每一個元素的型態都是：
**cv::Mat**
所以你可以像這樣使用：

```cpp
images[0] = cv::imread("a.jpg");
images[1] = cv::imread("b.jpg");
```

🧩 指標推導
在函式參數或傳遞時，如果你把這個陣列名稱（images）當成參數傳遞，會自動衰退 (decay) 成指標：

``` cpp
cv::Mat* ptr = images;
```

這裡 ptr 的型態就是：
**cv::Mat***

🧮 對照總覽表

| 表達式                      | 型態                | 說明         |
| :----------------------- | :---------------- | :--------- |
| `cv::Mat images[batch];` | `cv::Mat [batch]` | 固定長度陣列     |
| `images`（用在傳遞時）          | `cv::Mat*`        | 指向第一個元素的指標 |
| `images[i]`              | `cv::Mat`         | 陣列中的單一影像物件 |

⚠️ 注意事項
1. 陣列長度 batch 必須是編譯期常數（在 C++ 標準語法裡）。
若你寫：

``` cpp
int batch = 8;
cv::Mat images[batch];  // ❌ 非標準 C++
```

這在 GCC/Clang 可行（GNU 擴充的 VLA 語法），但不屬於標準 C++。
正統寫法是用 std::vector：

``` cpp
std::vector<cv::Mat> images(batch);
```

2. cv::Mat 是個智慧指標樣的 class（內部共享記憶體），所以用陣列或 vector 管理都安全。

### cv::Mat images[batch] 跟 std::vector<cv::Mat> images(batch) 在記憶體配置與可傳遞性上的差異

🧩 一、基本定義與儲存差異

| 面向   | `cv::Mat images[batch];`       | `std::vector<cv::Mat> images(batch);`                          |
| ---- | ------------------------------ | -------------------------------------------------------------- |
| 型態   | 固定長度陣列（stack object array）     | 動態長度容器（heap allocation）                                        |
| 配置位置 | 通常在 **stack（堆疊）** 上配置          | `std::vector` 自身在 **stack** 上，但元素容器（`cv::Mat`）在 **heap（堆積）** 上 |
| 長度可變 | ❌ 不可（編譯期常數）                    | ✅ 可變（可 `resize()`、`push_back()`）                               |
| 支援範圍 | 僅限標準 C++ 靜態陣列大小（GNU 支援變長，但非標準） | 完全標準、通用                                                        |

⚙️ 二、記憶體配置細節（以 batch = 8 為例）
- cv::Mat images[8];
- 分配在 stack 上：

``` cpp
| Stack frame |
├── images[0] : cv::Mat (約 32 bytes)
├── images[1] : cv::Mat
├── ...
└── images[7] : cv::Mat
```

- 這些 cv::Mat 是物件（非實際影像 buffer），每個 Mat 的資料（像素）仍存在 heap 或 GPU。
所以「每個 Mat 的 header 在 stack，上層實際影像記憶體在 heap」。
- std::vector<cv::Mat> images(8);
- images 物件本身在 stack 上，內部管理一塊 heap buffer：

``` cpp
| Stack frame |
└── std::vector<cv::Mat> images
       ↓
      | Heap |
      ├── cv::Mat[0]
      ├── cv::Mat[1]
      └── cv::Mat[7]
```

- 每個元素的 cv::Mat header 在 heap 上。
實際影像 buffer 還是獨立存在 heap/GPU。

🔁 三、傳遞（Passability）

| 操作             | `cv::Mat images[batch];`               | `std::vector<cv::Mat> images(batch);`                        |
| -------------- | -------------------------------------- | ------------------------------------------------------------ |
| 函式傳遞           | 會 decay 成 `cv::Mat*`，所以需另外傳 `batch` 大小 | 可直接傳 `std::vector<cv::Mat>&` 或 `const std::vector<cv::Mat>&` |
| 回傳使用           | ❌ 不可直接 return 陣列                       | ✅ 可 return vector                                            |
| move semantics | ❌ 無法移動（陣列是固定結構）                        | ✅ 完整支援 move / copy / swap                                    |
| STL 相容性        | ❌ 不支援                                  | ✅ 可搭配 `std::for_each`、`std::transform`、range-based loop      |

🧠 四、在 GPU / CUDA 前處理中的影響
- cv::Mat images[batch];
✅ 優點：
- 適合「固定 batch、低開銷」場景。
- stack 分配速度快（不會 malloc/free）。
- header 連續，可用 pointer 遍歷，方便傳給 kernel。
⚠️ 缺點：
- 無法動態調整 batch。
- 不能直接傳給需要容器語義的 API（例如 std::ranges 或 std::vector<cv::cuda::GpuMat> 混合）。
- 如果 batch 太大，stack 容易爆（>1MB）。
- std::vector<cv::Mat> images(batch);
✅ 優點：
- 容器可動態管理、傳遞安全。
- 更適合 pipeline 式架構（像你在做的 multi-stream inference）。
- 支援 move，能安全在任意函式之間傳遞。
- 跟 STL / thrust / TensorRT binding 更相容。
⚠️ 缺點：
- 建立時會呼叫 heap allocator（多一層指標間接性）。
- 大量小 vector 可能導致記憶體碎片化（但可用 reserve() 解決）。

🧩 五、實務建議（針對你的場景）

| 場景                                                         | 建議寫法                                                                         |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| CUDA kernel 前的 batch warpAffine、固定 batch 大小                | `cv::Mat images[BATCH_SIZE];`（快速存取）                                          |
| 多 stream / 動態 batch pipeline（含 pybind 或 TensorRT pipeline） | `std::vector<cv::Mat>`（靈活、安全）                                                |
| GPU buffer pool / zero-copy buffer reuse                   | 儲存成 `std::vector<cv::cuda::GpuMat>` 或自定義 struct，並搭配 pointer array 傳遞給 kernel |

💡 小延伸（如果你要給 CUDA 用）
假設你要傳多張圖片的 data pointer 給 GPU kernel：

``` cpp
std::vector<cv::Mat> images(batch);
std::vector<uint8_t*> d_ptrs(batch);
for (int i = 0; i < batch; ++i)
    d_ptrs[i] = images[i].data;

cudaMemcpy(d_srcArray, d_ptrs.data(), batch * sizeof(uint8_t*), cudaMemcpyHostToDevice);
```

這時 std::vector 比陣列更方便（因為 .data() 可以直接取指標，且長度動態）。

總結一句話：
🚀 如果你在寫底層 CUDA kernel demo，用陣列。
如果你在設計模組化 inference pipeline，用 vector。