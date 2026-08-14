這是一份針對 SDL2 (Simple DirectMedia Layer 2) 的入門教學。
---

## 1. 什麼是 SDL2？

SDL2 是一個跨平台的開發函式庫，旨在提供對音訊、鍵盤、滑鼠、搖桿以及透過 OpenGL 或 Direct3D 進行繪圖硬體底層存取的管道。它被廣泛用於模擬器和遊戲開發。

---

## 2. 核心組件與生命週期

開發一個 SDL2 程式通常遵循以下流程：**初始化 → 建立視窗 → 事件迴圈 (Event Loop) → 渲染 → 清理**。

### A. 初始化與建立視窗

首先需要呼叫 `SDL_Init()`。這會初始化指定的子系統（如影片、音訊）。

``` C++
#include <SDL2/SDL.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    // 初始化影片子系統
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // 建立視窗
    SDL_Window* window = SDL_CreateWindow(
        "SDL2 Tutorial",                  // 視窗標題
        SDL_WINDOWPOS_UNDEFINED,           // 初始 X 位置
        SDL_WINDOWPOS_UNDEFINED,           // 初始 Y 位置
        800,                               // 寬度
        600,                               // 高度
        SDL_WINDOW_SHOWN                   // 旗標
    );

    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // 執行邏輯（見下文）...

    // 銷毀與離開
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
```

## 3. 渲染與繪圖 (Renderer)

在 SDL2 中，我們通常使用 `SDL_Renderer` 來進行 2D 繪圖。它支援硬體加速。

### 渲染流程：

1. **`SDL_RenderClear`**：清除目前的渲染目標。
    
2. **`SDL_SetRenderDrawColor`**：設定筆刷顏色。
    
3. **`SDL_RenderFillRect`** / **`SDL_RenderCopy`**：繪製幾何圖形或貼圖。
    
4. **`SDL_RenderPresent`**：將緩衝區的內容顯示到螢幕上（更新畫面）。

```C++
SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

// 設定背景為黑色
SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
SDL_RenderClear(renderer);

// 繪製一個紅色的矩形
SDL_Rect fillRect = { 200, 150, 400, 300 };
SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
SDL_RenderFillRect(renderer, &fillRect);

// 更新畫面
SDL_RenderPresent(renderer);
```

## 4. 事件處理 (Event Handling)

為了不讓程式卡死，你需要不斷檢查作業系統傳來的事件（如按下關閉鈕、鍵盤輸入）。

```C++
SDL_Event e;
bool quit = false;

while (!quit) {
    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            quit = true;
        }
        else if (e.type == SDL_KEYDOWN) {
            switch (e.key.keysym.sym) {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
            }
        }
    }
    // 更新遊戲邏輯與繪圖...
}
```

## 5. C++ 實作建議

既然你偏好不使用標準函式庫 (non-STL)，可以考慮將 SDL2 的指標封裝在自定義的結構或類別中，並利用 **RAII** 概念處理 `SDL_DestroyWindow` 與 `SDL_DestroyRenderer`。

### 編譯指令 (Linux/macOS 範例)

```bash
g++ main.cpp -o my_game -lSDL2
```

### 常見問題與建議：

- **影像格式**：原生的 SDL2 只支援 BMP。若要讀取 PNG 或 JPG，需要額外的 `SDL_image` 擴充包。
    
- **效能優化**：儘量減少 `SDL_RenderPresent` 的次數，通常維持在每秒 60 次 (VSync)。
    
- **語言學習**：在寫 code 時，變數命名建議使用英文慣用語，例如 `surface` (表面)、`texture` (貼圖) 等，這對你學習英文術語很有幫助。
    

> **小提醒：** SDL2 的 `SDL_GetError()` 會回傳 `const char*`，你可以直接輸出到 stderr 或寫入日誌檔案，不一定需要 `iostream`。


### 讀取 PNG 或 JPG，額外的 `SDL_image` 擴充包。

## 1. 安裝與連結 (Installation & Linking)

在開發環境中，你除了連結 `SDL2` 本身，還需要額外連結 `SDL2_image`。

- **Linux (Ubuntu/Debian):** `sudo apt-get install libsdl2-image-dev`
    
- **macOS (Homebrew):** `brew install sdl2_image`
    
- **編譯指令範例：**
```bash
g++ main.cpp -o app -lSDL2 -lSDL2_image
```

## 2. 初始化 SDL_image

與 SDL2 主程式類似，你需要呼叫 `IMG_Init` 並傳入你想要支援的格式標籤（Flags）。

```C++
#include <SDL2/SDL_image.h>

// 初始化支援 PNG 與 JPG
int imgFlags = IMG_INIT_PNG | IMG_INIT_JPG;
if (!(IMG_Init(imgFlags) & imgFlags)) {
    // 取得錯誤資訊：IMG_GetError()
    return 1;
}
```

## 3. 載入圖片並轉換為貼圖 (Texture)

在 SDL2 中，為了效能考慮，我們通常不直接在畫面繪製 `SDL_Surface`（在 CPU 中處理），而是將其轉換為 `SDL_Texture`（儲存在 GPU 顯存中）。

### 核心流程：

1. 使用 `IMG_Load` 讀取檔案至 `SDL_Surface`。
    
2. 使用 `SDL_CreateTextureFromSurface` 將其轉換成 GPU 可用的 `SDL_Texture`。
    
3. 釋放不再需要的 `SDL_Surface`。

```C++
SDL_Texture* loadTexture(const char* path, SDL_Renderer* renderer) {
    SDL_Texture* newTexture = NULL;

    // 直接將圖片載入為 Surface
    SDL_Surface* loadedSurface = IMG_Load(path);
    if (loadedSurface == NULL) {
        // 處理錯誤：IMG_GetError()
        return NULL;
    }

    // 將 Surface 轉換成 Texture
    newTexture = SDL_CreateTextureFromSurface(renderer, loadedSurface);
    
    // 轉換完畢後，Surface 就可以從記憶體釋放了 (不使用 STL，手動管理)
    SDL_FreeSurface(loadedSurface);

    return newTexture;
}
```

## 4. 繪製到螢幕 (Rendering)

一旦擁有了 Texture，你可以使用 `SDL_RenderCopy` 來指定要繪製的來源區域（Source Rect）與目標位置（Destination Rect）。

```C++
// 準備目標位置矩形
SDL_Rect destRect = { 100, 100, 200, 200 }; // x, y, width, height

// 將 Texture 複製到 Renderer
SDL_RenderCopy(renderer, myTexture, NULL, &destRect);

// 更新畫面
SDL_RenderPresent(renderer);
```

## 5. 清理 (Cleanup)

程式結束前，務必釋放資源，順序為：**Texture → Renderer → Window → IMG_Quit → SDL_Quit**。

```C++
SDL_DestroyTexture(myTexture);
IMG_Quit();
SDL_Quit();
```

