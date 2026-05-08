-cv::Mat 是 OpenCV 最核心的資料結構，主要由兩部分組成：矩陣標頭（Header）與資料指標（Pointer to Data）。

以下為您整理 cv::Mat 初始化的幾種主要方式與核心觀念：

1. 常用建構式 (Constructors)
這是最直覺的初始化方式，您可以指定矩陣的大小與資料類型。

- 指定大小與類型：

``` cpp
// 建立一個 640x480 的 8-bit 三通道矩陣 (BGR)
cv::Mat img(480, 640, CV_8UC3); 

// 使用 cv::Size
cv::Mat img2(cv::Size(640, 480), CV_32FC1); // 單通道 32-bit 浮點數
```

- 帶有初始值的初始化：

``` cpp
// 建立一個 3x3 矩陣，所有元素初始化為 0
cv::Mat m = cv::Mat::zeros(3, 3, CV_8UC1);

// 初始化為全 1
cv::Mat m2 = cv::Mat::ones(3, 3, CV_32F);

// 指定特定顏色 (例如純紅色，假設是 BGR)
cv::Mat redImg(480, 640, CV_8UC3, cv::Scalar(0, 0, 255));
```
2. 深度與通道：理解 CV_8UC3 等巨集
在初始化時，類型參數的格式通常為 CV_<bit_depth><S|U|F>C<number_of_channels>：

bit_depth: 8, 16, 32, 64

S/U/F: Signed int, Unsigned int, Float

C: Channels (1, 2, 3, 4...)

|巨集範例|意義|適用場景|
|---|---|---|
|CV_8UC1|"8-bit Unsigned| 1 Channel"|灰階圖|
|CV_8UC3|"8-bit Unsigned| 3 Channels"|標準 BGR 彩色圖|
|CV_32FC1|"32-bit Float|1 Channel"|深度圖或特徵描述子|


3. 使用外部資料初始化 (不複製記憶體)
如果您手邊已經有一塊預先分配好的記憶體（例如來自相機 SDK 的 Buffer），可以直接包裝成 cv::Mat，這不會發生記憶體拷貝。

``` cpp
unsigned char* rawData = getFrameFromCamera();
// 建立標頭指向既有資料
cv::Mat img(height, width, CV_8UC3, rawData);
```

注意： 這種方式下，cv::Mat 不負責該記憶體的生命週期，您必須確保 rawData 在 img 使用期間不會被釋放。


4. 複製與賦值 (淺拷貝 vs 深拷貝)
這是在開發中最容易踩坑的地方：

淺拷貝 (Shallow Copy)： 僅複製標頭，共用同一份資料塊。

``` cpp
cv::Mat A = B; // A 和 B 指向相同的資料
```

深拷貝 (Deep Copy)： 完整複製資料。

``` cpp
cv::Mat A = B.clone();
// 或者
cv::Mat C;
B.copyTo(C);
```

5. 矩陣運算與賦值對於小型矩陣（例如相機矩陣、旋轉矩陣），可以使用 `cv::Mat_` 或列表初始化(C++11)

``` cpp
// 列表初始化 (C++11)
cv::Mat K = (cv::Mat_<double>(3,3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);

// 或者更現代的寫法
cv::Mat M = (cv::Mat_<int>(2,2) << 1, 2, 3, 4);
```

6. 記憶體布局 (Memory Layout)
`cv::Mat` 的資料在記憶體中通常是連續存放的，但為了效能優化（記憶體對齊），每行末尾可能會有些許填充（Padding）。
- 可以使用 `img.isContinuous()` 檢查。
- 計算特定像素位址： $addr(M_{i,j}) = M.data + M.step[0] \cdot i + M.step[1] \cdot j$

總結
1. 一般建立：用 cv::Mat(rows, cols, type)。
2. 固定值：用 `zeros()`, `ones()`, 或 `Scalar`。
3. 效能考量：避免不必要的 clone()，但要注意淺拷貝產生的副作用。
4. 外部 Buffer：直接傳入指標，但需自行管理生命週期。