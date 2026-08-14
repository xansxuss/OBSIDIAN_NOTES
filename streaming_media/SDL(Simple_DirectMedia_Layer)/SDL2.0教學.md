---
title: "SDL2.0教學"
source: "https://gemini.google.com/app/4d94b93db67c1697"
author:
published:
created: 2026-06-22
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
SDL2 (Simple DirectMedia Layer 2.0) 是一個跨平台的開發函式庫，專門用來處理低階的音訊、鍵盤、滑鼠、搖桿以及透過 OpenGL / Direct3D / Vulkan 的繪圖硬體加速。

它本身是用純 C 寫成的，但完美相容於 C++。既然你在 C++ 上傾向不使用標準函式庫 (STL)，SDL2 的設計剛好非常符合你的胃口，因為它有自己的記憶體管理與資料結構邏輯，完全不需要依賴 `std::vector` 或 `std::string` 。

以下是整理 SDL2 的核心架構與開發教學。

## 核心架構與生命週期

SDL2 的運作邏輯非常直覺，主要圍繞在 **初始化** 、 **主迴圈 (Main Loop)** 、 **事件處理 (Event Handling)** 與 **資源釋放** 。

### 1\. 初始化與系統啟動

在使用任何 SDL 功能之前，必須先初始化對應的子系統（例如視訊、音訊、計時器等）。

```
// 初始化視訊子系統
if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    // 錯誤處理：SDL_GetError() 會傳回 C 釋放的字串字面量 (const char*)
    // 不需要依賴 std::cerr
    return -1; 
}
```

### 2\. 建立視窗與渲染器

SDL2 將「視窗 (Window)」與「繪圖行為 (Renderer)」分離：

- **`SDL_Window`**: 代表作業系統視窗本身（處理大小、位置、邊框）。
- **`SDL_Renderer`**: 負責硬體加速繪圖的渲染上下文 (Context)。

## 實戰範例：純 C 風格的 C++ 視窗程式

以下是一個完全 **不使用 C++ 標準函式庫** 的標準 SDL2 範例，展示了如何建立視窗、處理關閉事件、並清除畫布。

```
#include <SDL2/SDL.h>

// 定義視窗寬高常數
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

int main(int argc, char* argv[]) {
    // 1. 初始化 SDL 視訊系統
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // 2. 建立視窗
    SDL_Window* window = SDL_CreateWindow(
        "SDL2 Taiwan Tutorial",                  // 視窗標題
        SDL_WINDOWPOS_UNDEFINED,                // 初始 X 座標
        SDL_WINDOWPOS_UNDEFINED,                // 初始 Y 座標
        SCREEN_WIDTH, SCREEN_HEIGHT,            // 寬高
        SDL_WINDOW_SHOWN                        // 顯示旗標
    );

    if (window == nullptr) {
        SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // 3. 建立硬體加速渲染器 (-1 代表自動選擇預設顯示卡驅動)
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        SDL_Log("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // 4. 主迴圈控制變數
    bool quit = false;
    SDL_Event e;

    // 主迴圈 (Main Loop)
    while (!quit) {
        // 5. 事件處理 (Event Polling)
        // 這裡會清空事件佇列，不需使用 std::queue
        while (SDL_PollEvent(&e) != 0) {
            // 使用者點擊視窗關閉按鈕
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            // 鍵盤事件處理
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    quit = true; // 按下 ESC 鍵退出
                }
            }
        }

        // 6. 渲染繪圖
        // 設定清除顏色為深藍色 (R, G, B, A)
        SDL_SetRenderDrawColor(renderer, 0x1A, 0x22, 0x38, 0xFF);
        SDL_RenderClear(renderer); // 用剛才設定的顏色填滿整個視窗

        // 在畫布正中央畫一個矩形
        SDL_Rect fillRect = { SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 };
        SDL_SetRenderDrawColor(renderer, 0xFF, 0x6B, 0x6B, 0xFF); // 紅色矩形
        SDL_RenderFillRect(renderer, &fillRect);

        // 7. 將快取區內容更新到螢幕上 (Double Buffering)
        SDL_RenderPresent(renderer);
    }

    // 8. 釋表資源與善後 (嚴格遵守建立的反向順序)
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
```

## 關鍵核心觀念解析

### 雙緩衝機制 (Double Buffering)

SDL2 預設開啟雙緩衝技術。所有的繪圖指令（例如 `SDL_RenderFillRect` ）都是在 **後台緩衝區 (Back Buffer)** 進行。當你呼叫 `SDL_RenderPresent` 時，後台緩衝區才會與 **前台緩衝區 (Front Buffer)** 交換，瞬間顯示在螢幕上。這樣可以完全避免畫面撕裂 (Tearing) 與閃爍。

### 事件驅動 (Event Driven)

`SDL_PollEvent(&e)` 是非阻塞的。如果事件佇列 (Event Queue) 裡面有資料，它會回傳 `1` 並把資料塞進 `SDL_Event` 結構體中；如果空了，它會立刻回傳 `0` ，讓程式繼續執行下一行的繪圖與邏輯更新，保證遊戲或應用程式不會卡死。

### 基本繪圖 API 對照表

| 功能 | SDL2 函式 | 說明 |
| --- | --- | --- |
| **設定顏色** | `SDL_SetRenderDrawColor` | 設定接下來所有繪圖動作的 RGBA 畫筆顏色 |
| **清空畫布** | `SDL_RenderClear` | 用當前畫筆顏色塗滿整個視窗 |
| **畫點** | `SDL_RenderDrawPoint` | 在指定座標 $(x, y)$ 畫一個像素 |
| **畫線** | `SDL_RenderDrawLine` | 連接兩點 $(x_1, y_1)$ 到 $(x_2, y_2)$ |
| **空心矩形** | `SDL_RenderDrawRect` | 根據 `SDL_Rect` 畫出矩形外框 |
| **實心矩形** | `SDL_RenderFillRect` | 根據 `SDL_Rect` 填滿矩形區域 |

## 常用擴充套件 (Extension Libraries)

SDL2 核心只支援基本圖形與 `.bmp` 格式。實務上開發通常會搭配以下官方維護的擴充套件（它們的設計風格與 SDL2 完全一致）：

1. **SDL\_image**: 支援 `.png`, `.jpg`, `.webp` 等圖片載入。
	- 關鍵函式： `IMG_LoadTexture(renderer, "path.png");`
2. **SDL\_ttf**: 支援 TrueType 字型轉成紋理繪製。
	- 關鍵函式： `TTF_OpenFont()`, `TTF_RenderText_Solid()`
3. **SDL\_mixer**: 專為遊戲設計的多聲道音訊混音庫。
	- 關鍵函式： `Mix_PlayMusic()`, `Mix_PlayChannel()`

## 繪圖部份與opencv比較

SDL2 與 OpenCV 在「繪圖與影像處理」的定位上，有著本質上的不同。簡單來說：**SDL2 是一個「多媒體視窗與渲染框架」，而 OpenCV 是一個「電腦視覺與影像處理演庫」**。

以下詳細拆解兩者在繪圖機制上的核心差異、架構對比，以及在不同情境下的選擇。

## 核心哲學與架構對比

|特性|SDL2 (Simple DirectMedia Layer)|OpenCV (Open Computer Vision)|
|---|---|---|
|**主要定位**|視窗管理、硬體加速控制（遊戲、多媒體播放器）|影像處理、矩陣運算、電腦視覺與 AI 模型前後處理|
|**運作核心**|**GPU 紋理 (Texture)** 導向，預設走硬體加速 (OpenGL/Direct3D)|**CPU 矩陣 (`cv::Mat`)** 導向，所有的繪圖都是在記憶體改寫像素|
|**效能重點**|著重於**高畫格率 (High FPS)** 與即時畫面更新、輸入零延遲|著重於**複雜演算法**的精確度（如濾鏡、邊緣偵測、特徵點）|
|**外部依賴**|輕量化，純 C 寫成，不依賴 C++ 標準函式庫|龐大，深度依賴 C++ STL（如 `std::vector`），且編譯時間長|

## 繪圖機制深入剖析

### 1. OpenCV 的繪圖方式：CPU 矩陣修改

當你在 OpenCV 中呼叫 `cv::rectangle` 或 `cv::circle` 時，它做的事情是：

- 在 CPU 端直接修改 `cv::Mat` 內部記憶體陣列（即像素的 BGR 數值）。
    
- 當你呼叫 `cv::imshow` 時，OpenCV 會透過作業系統的底層 API（如 Windows GDI 或 Linux X11）把這整塊記憶體從 **系統記憶體 (RAM)** 複製到 **顯示記憶體 (VRAM)**。
    
- **缺點：** 當解析度很高（例如 4K）或需要每秒更新 60 次以上時，頻繁的 CPU-to-GPU 記憶體拷貝會造成極大的效能瓶頸。
    

### 2. SDL2 的繪圖方式：GPU 頂點與紋理操作

在 SDL2 中，繪圖邏輯完全不同：

- `SDL_Texture` 是一塊直接存在於 VRAM（顯示記憶體）的圖形資料。
    
- 當你呼叫 `SDL_RenderCopy` 或 `SDL_RenderFillRect` 時，你實際上是向顯示卡發送了幾條繪圖指令（例如畫兩個三角形組成的矩形，並貼上紋理）。
    
- **優點：** 所有的幾何運算、縮放、旋轉與顏色混合（Alpha Blending）都由 GPU 的固定功能管線（Fixed-function pipeline）或著色器處理，CPU 完全不參與像素計算。因此能輕易達到 120 FPS 以上的流暢度。
    

## 程式碼實作對比 (C++ 風格)

假設我們要在一個 800×600 的視窗內，每幀將一個紅色矩形向右移動。

### OpenCV 版本 (CPU 密集型)

C++

```
#include <opencv2/opencv.h>

int main() {
    // 建立一張 800x600 的黑色畫布 (RAM)
    cv::Mat canvas = cv::Mat::zeros(600, 800, CV_8UC3);
    int x = 0;

    while (true) {
        // 必須先用黑色覆蓋上一幀的內容（修改記憶體）
        canvas = cv::Mat::zeros(600, 800, CV_8UC3);

        // 在 CPU 端改寫像素，畫出紅色矩形
        cv::rectangle(canvas, cv::Rect(x, 150, 200, 200), cv::Scalar(0, 0, 255), -1);

        cv::imshow("OpenCV Window", canvas); // 將整塊 RAM 拷貝到 VRAM 顯示
        
        x += 2;
        if (cv::waitKey(16) == 27) break; // 約 60 FPS，偵測 ESC
    }
    return 0;
}
```

### SDL2 版本 (GPU 加速型)

C++

```
#include <SDL2/SDL.h>

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* win = SDL_CreateWindow("SDL2 Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 600, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    int x = 0;
    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e)) { if (e.type == SDL_QUIT) quit = true; }

        // 指令 1：叫 GPU 用黑色清空 VRAM 緩衝區
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // 指令 2：叫 GPU 在指定 VRAM 位置畫一個紅色矩形
        SDL_Rect rect = { x, 150, 200, 200 };
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        SDL_RenderFillRect(renderer, &rect);

        // 指令 3：交換前後緩衝區 (VRAM 內部操作)
        SDL_RenderPresent(renderer);

        x += 2;
        SDL_Delay(16); // 約 60 FPS
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
```

## 總結：AI 工程師的實務抉擇

在實際的 AI 或視覺專案中，你通常不會讓這兩個工具「二選一」，而是讓它們**協同工作**：

1. **何時單獨使用 OpenCV？**
    
    - 撰寫純演算法測試、模型推論的前後處理（例如：將影像進行 Resize、Normalize、Bgr2Rgb）。
        
    - 需要繪製複雜的幾何圖形、文字標註（OpenCV 的 `cv::putText` 比 SDL2 的字型處理簡單非常多）。
        
    - **缺點：** 不要用 OpenCV 來做複雜的 UI 介面或互動式遊戲。
        
2. **何時引入 SDL2？**
    
    - 當你需要開發一個**即時的 AI 串流監控面板 (Real-time Dashboard)**，需要極高的 FPS 且不能卡頓。
        
    - 需要跨平台處理非影像的事務，例如接收遙控器搖桿 (Joystick) 訊號、同步播放音訊。
        
3. **最強大的黃金組合 (OpenCV + SDL2)：**
    
    - 使用 OpenCV/推理框架（如 ONNX Runtime）在 CPU/GPU 計算出影像與 Bounding Box 矩陣。
        
    - 將 OpenCV 的 `cv::Mat` 資料指標轉化為 `SDL_Texture`（僅做一次記憶體上傳）。
        
    - 後續所有的**畫面渲染、UI 按鈕點擊、Bounding Box 疊加、文字顯示**，全部交由 SDL2 透過 GPU 進行硬體加速繪製。