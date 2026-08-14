```CMake
cmake_minimum_required(VERSION 3.10)
project(StreamDecoder LANGUAGES CXX C)
# ==============================================================================
# 平台偵測與訊息輸出
# ==============================================================================
message(STATUS "--------------------------------------------------")
message(STATUS "Checking Target Platform...")
message(STATUS " - CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
# 1. 先判斷是否為 ARM64 架構
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
# 2. 在 ARM64 下，區分是否為 Jetson (檢查 Tegra 驅動路徑)
if(EXISTS "/usr/lib/aarch64-linux-gnu/tegra")
set(CURRENT_PLATFORM "PLATFORM_JETSON")
message(STATUS ">>> DETECTED: NVIDIA Jetson (ARM64) <<<")
else()
set(CURRENT_PLATFORM "PLATFORM_ARM64_GENERIC")
message(STATUS ">>> DETECTED: Generic ARM64 Platform (Non-NVIDIA) <<<")
endif()
# 3. 其他架構（通常為 x86_64）
else()
set(CURRENT_PLATFORM "PLATFORM_X86_64")
message(STATUS ">>> DETECTED: x86_64 PC <<<")
endif()
message(STATUS " - Defined Macro: ${CURRENT_PLATFORM}")
message(STATUS "--------------------------------------------------")
```

這份 `CMakeLists.txt` 的核心目的，是為了讓你的 C/C++ 專案能夠**跨平台編譯**。它會自動偵測編譯環境的硬體架構，並定義對應的 C/C++ 巨集（Macro），讓你在原始碼中可以用 `#ifdef` 區隔不同平台的實作（例如 Jetson 的 NVMM 記憶體硬體解碼、一般 ARM64 的處理、或是 x86_64 PC 的軟體解碼）。

以下為你逐段詳細拆解程式碼的功能：

## 1. 專案基本設定



``` CMake
cmake_minimum_required(VERSION 3.10)
project(StreamDecoder LANGUAGES CXX C)
```

- **`cmake_minimum_required(VERSION 3.10)`**：指定執行此檔案所需的最低 CMake 版本。版本 3.10 算是非常保守且安全的設定，不論是在舊版的 Ubuntu 18.04 (JetPack 4.x) 還是新版的環境都能完美相容。
    
- **`project(StreamDecoder LANGUAGES CXX C)`**：定義專案名稱為 `StreamDecoder`。後面的 `LANGUAGES CXX C` 告訴 CMake 這個專案會同時用到 C++ (`CXX`) 與 C 語言 (`C`)，CMake 會據此去尋找系統中的 `g++` 和 `gcc` 編譯器。
    

## 2. 平台偵測與邏輯判斷

這一段是整份檔案的核心，利用了 CMake 的條件控制流程來做三叉分支判斷。

``` CMake
message(STATUS "--------------------------------------------------")
message(STATUS "Checking Target Platform...")
message(STATUS "  - CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
```

- **`message(STATUS "...")`**：在終端機（Terminal）印出帶有 `--` 前綴的通知訊息。這裡用來做視覺隔離，並印出 CMake 內建變數 `${CMAKE_SYSTEM_PROCESSOR}` 的數值（例如 `x86_64` 或 `aarch64`），方便除錯。
    

### 第一層判斷：是否為 ARM64 架構？

``` CMake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
```

- **`MATCHES "aarch64"`**：使用正規表示法比對系統處理器名稱。如果是在 NVIDIA Jetson、Raspberry Pi 4/5 或是 Apple Silicon 的 Linux 虛擬機上編譯，這個條件就會成立，進入 ARM64 的專屬區塊。
    

### 第二層判斷：區分 Jetson 與一般 ARM64

``` CMake

    # 2. 在 ARM64 下，區分是否為 Jetson (檢查 Tegra 驅動路徑)
    if(EXISTS "/usr/lib/aarch64-linux-gnu/tegra")
        set(CURRENT_PLATFORM "PLATFORM_JETSON")
        message(STATUS ">>> DETECTED: NVIDIA Jetson (ARM64) <<<")
    else()
        set(CURRENT_PLATFORM "PLATFORM_ARM64_GENERIC")
        message(STATUS ">>> DETECTED: Generic ARM64 Platform (Non-NVIDIA) <<<")
    endif()
```

- **`EXISTS "/usr/lib/aarch64-linux-gnu/tegra"`**：這是關鍵。NVIDIA Jetson 平台（Tegra 晶片）專有的多媒體與 GPU 驅動函式庫都會存放這個特定路徑下。如果該路徑存在，代表這台 ARM64 裝置是 **Jetson**。
    
- **`set(CURRENT_PLATFORM "...")`**：建立一個自訂變數 `CURRENT_PLATFORM`，並將判斷結果字串存進去。
    
    - 如果是 Jetson，設定為 `PLATFORM_JETSON`。
        
    - 如果不是 Jetson 的 ARM64（如樹梅派），設定為 `PLATFORM_ARM64_GENERIC`。
        

### 第三分支：x86_64 PC 平台

``` CMake
else()
    set(CURRENT_PLATFORM "PLATFORM_X86_64")
    message(STATUS ">>> DETECTED: x86_64 PC <<<")
endif()
```

- 如果前面第一層的 `aarch64` 不匹配，通常代表是我們一般開發用的 Intel 或 AMD 處理器桌機/筆電環境，此時變數會設定為 `PLATFORM_X86_64`。
    

## 3. 目標設定與編譯巨集綁定（Modern CMake 作法）

``` CMake
add_executable(StreamDecoder main.cpp)
```

- **`add_executable(...)`**：告訴 CMake 我要編譯出一個叫做 `StreamDecoder` 的執行檔，而它的原始碼來源是 `main.cpp`（如果有其他 `.cpp` 檔，依序寫在後面即可，例如 `main.cpp decoder.cpp`）。
    
``` CMake

target_compile_definitions(StreamDecoder PRIVATE ${CURRENT_PLATFORM})
```

- **`target_compile_definitions`**：這是現代 CMake 最推薦的作法。它的作用是**將前面判斷出來的平台變數，當作預處理器巨集（Preprocessor Macro）餵給特定的 Target**（也就是 `StreamDecoder`）。
    
- **`PRIVATE` 關鍵字**：代表這個巨集只有在編譯 `StreamDecoder` 本身時才會生效。如果以後這個專案被別人當成 Library 引用，這個巨集不會污染到別人的專案。
    

## 💡 這段程式碼在 C++ 中要怎麼用？

當你執行完 `cmake ..` 和 `make` 後，CMake 會自動在編譯時加上 `-DPLATFORM_XXX` 參數。

這意味著你可以在你的 `main.cpp` 裡面寫這樣的程式碼，達到完全的跨平台相容：

``` C++
#include <iostream>

int main() {
#ifdef PLATFORM_JETSON
    std::cout << "正在初始化 Jetson 硬體解碼器 (NVMM)..." << std::endl;
    // 呼叫 NVIDIA Jetson 專屬的 NvBufSurface 或 NvVideoDecoder API
#elif defined(PLATFORM_ARM64_GENERIC)
    std::cout << "正在初始化泛用 ARM64 解碼器..." << std::endl;
    // 呼叫一般 ARM64 的加速方案（例如 FFmpeg 或 樹梅派 MMAL/V4L2）
#elif defined(PLATFORM_X86_64)
    std::cout << "正在初始化 x86_64 PC 軟體解碼器..." << std::endl;
    // 呼叫標準 CPU 解碼器
#endif

    return 0;
}
```

這樣一來，同一套原始碼不需要修改，直接丟到 PC、Jetson 或一般 ARM 板子上，CMake 就會幫你搞定一切編譯設定！