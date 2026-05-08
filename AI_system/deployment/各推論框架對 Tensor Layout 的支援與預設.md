🚦 各推論框架對 Tensor Layout 的支援與預設（圖像分類為例）
| 框架             | Tensor layout | 圖像輸入 Shape     | Channel 順序 | 資料類型                         | 備註                         |
| -------------- | ------------- | -------------- | ---------- | ---------------------------- | -------------------------- |
| **ONNX**       | NCHW（預設）      | `(1, 3, H, W)` | RGB        | `float32` / `uint8`          | 與 PyTorch 相容，部分模型支援 NHWC   |
| **TensorRT**   | NCHW（效能最佳）    | `(1, 3, H, W)` | RGB        | `float32` / `int8`           | 必須手動處理歸一化與格式轉換             |
| **TFLite**     | NHWC（預設）      | `(1, H, W, 3)` | RGB        | `float32` / `uint8` / `int8` | 移動端導向，效能最佳於 NHWC           |
| **OpenCV DNN** | NCHW          | `(1, 3, H, W)` | BGR        | `float32`                    | 若直接載入 OpenCV 模型注意格式差異      |
| **PyTorch**    | NCHW          | `(1, 3, H, W)` | RGB        | `float32`                    | 與 ONNX 輸出一致                |
| **TensorFlow** | NHWC          | `(1, H, W, 3)` | RGB        | `float32`                    | 可支援 NCHW，但預設和 GPU 加速為 NHWC |

推論時的前處理策略（總整理）：
| 前處理步驟                 | TensorRT (NCHW) | TFLite (NHWC)     |
| --------------------- | --------------- | ----------------- |
| resize                | ✅ (640x640)     | ✅                 |
| BGR → RGB             | ✅               | ✅                 |
| normalize (`/255.0`)  | ✅               | 視 quant config 決定 |
| layout HWC → CHW      | ✅               | ❌（保持 HWC）         |
| to tensor (NCHW/NHWC) | ✅ (`(1,3,H,W)`) | ✅ (`(1,H,W,3)`)   |

轉換流程總表（用 C++ + OpenCV + std::vector 實作）
| 步驟 | 動作                | 說明                         |
| -- | ----------------- | -------------------------- |
| 1  | BGR → RGB         | 一般模型用 RGB                  |
| 2  | resize            | 固定輸入尺寸（如 640x640）          |
| 3  | convertTo float32 | 歸一化或標準化前處理                 |
| 4  | HWC to NHWC       | reshape 成 `(1, H, W, C)`   |
| 5  | NHWC to NCHW      | rearrange 成 `(1, C, H, W)` |

C++ 程式碼：轉換 cv::Mat → NCHW（中間可取得 NHWC）

```
#include <opencv2/opencv.hpp>
#include <vector>

// 輸入 OpenCV Mat（BGR），轉成 NCHW 格式 float32 Tensor（std::vector）
std::vector<float> convertMatToNCHW(const cv::Mat& inputImage, int targetHeight, int targetWidth) {
    // 1. BGR → RGB
    cv::Mat imgRGB;
    cv::cvtColor(inputImage, imgRGB, cv::COLOR_BGR2RGB);
    // 2. Resize
    cv::resize(imgRGB, imgRGB, cv::Size(targetWidth, targetHeight));
    // 3. convert to float and normalize to [0, 1]
    imgRGB.convertTo(imgRGB, CV_32FC3, 1.0 / 255.0);
    int H = imgRGB.rows;
    int W = imgRGB.cols;
    int C = imgRGB.channels(); // == 3
    // Optional: NHWC buffer (1, H, W, C)
    std::vector<float> nhwc(1 * H * W * C);
    // 4. Fill NHWC
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            cv::Vec3f pixel = imgRGB.at<cv::Vec3f>(h, w);
            for (int c = 0; c < C; ++c) {
                int nhwc_index = h * W * C + w * C + c;
                nhwc[nhwc_index] = pixel[c];
            }
        }
    }
    // 5. Rearrange NHWC → NCHW
    std::vector<float> nchw(1 * C * H * W); // batch = 1
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int nhwc_index = h * W * C + w * C + c;
                int nchw_index = c * H * W + h * W + w;
                nchw[nchw_index] = nhwc[nhwc_index];
            }
        }
    }
    return nchw;
}
```

