在 C++17 之後，&lt;filesystem&gt; 函式庫成為了處理檔案系統路徑的標準工具。作為 AI 工程師，你可能經常需要處理資料集路徑、模型權重載入或日誌輸出，`std::filesystem::path` 能極大地簡化這些跨平台的繁瑣操作。

以下是 ` std::filesystem::path ` 的核心概念與常用操作整理：

1. 基本宣告與初始化
path 物件會根據作業系統自動處理路徑分隔符號（Windows 使用 \，Linux/macOS 使用 /）。

``` cpp
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    // 自動處理斜線方向
    fs::path p1 = "/usr/local/bin"; 
    fs::path p2 = "C:\\Users\\Desktop"; 
    
    // 使用 /= 運算子進行路徑拼接（自動補上分隔符）
    fs::path dataset_path = "data";
    dataset_path /= "train";
    dataset_path /= "image.png"; 
    // 結果：data/train/image.png (Linux) 或 data\train\image.png (Windows)
}
```

2. 常用成員函式（路徑分解）
這在處理模型檔案路徑時非常實用，可以輕鬆提取檔名或副檔名。

| 函式 | 說明 | 範例 (/home/user/model.onnx) |
| --- | --- | --- |
| .filename() | 取得完整檔名 | model.onnx |
| .stem() | 取得主檔名（不含副檔名） | model |
| .extension() | 取得副檔名 | .onnx |
| .parent_path() | 取得父目錄路徑 | /home/user |
| .is_absolute() | 是否為絕對路徑 | true |
### 3. 實用的非成員工具函式

這些函式通常配合 `std::filesystem` 命名空間使用：

- **檢查狀態**：`fs::exists(p)`、`fs::is_directory(p)`、`fs::is_regular_file(p)`。
    
- **建立目錄**：`fs::create_directories("path/to/subdir")`（會遞迴建立不存在的中間目錄）。
    
- **取得當前路徑**：`fs::current_path()`。
    
- **檔案大小**：`fs::file_size(p)`（回傳 bytes）。
    

---

### 4. 遍歷目錄 (Directory Iteration)

在讀取大量 Training Data 時，這是最有效率的做法：

```cpp
fs::path img_dir = "./datasets/coco";

// 非遞迴遍歷
for (const auto& entry : fs::directory_iterator(img_dir)) {
    std::cout << entry.path() << std::endl;
}

// 遞迴遍歷（包含子目錄）
for (const auto& entry : fs::recursive_directory_iterator(img_dir)) {
    if (entry.path().extension() == ".jpg") {
        // 處理圖片...
    }
}
```

### 💡 工程師小提點

1. **效能考量**：`std::filesystem::path` 物件在頻繁建立時會有小量開銷，但在一般 I/O 密集型任務中幾乎可以忽略。
    
2. **與舊代碼相容**：若要將 `path` 物件傳給只支援 `std::string` 或 `const char*` 的 API（如舊版的 `fopen`），請使用 `p.string().c_str()`。
    
3. **例外處理**：檔案操作（如 `copy`, `remove`）建議放在 `try-catch` 區塊內，捕捉 `fs::filesystem_error`。