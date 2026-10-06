# SDL2 + Python 詳細教學

> 本篇使用 `PySDL2`（`pysdl2` 套件），這是 SDL2 C API 的 Python ctypes 綁定，
> 函式命名與參數幾乎與原生 C 的 SDL2 一致，方便未來對照 C/C++ 版本的 SDL2 程式碼。

---

## 1. 環境安裝

PySDL2 本身只是 Python 的綁定層，**還需要 SDL2 的原生動態函式庫**才能運作。

```bash
# 安裝 Python 綁定
pip install pysdl2

# 安裝 SDL2 動態函式庫（二選一）
pip install pysdl2-dll      # 方式一：直接裝預編譯好的 DLL/so（跨平台最省事）
# 或者用系統套件管理員安裝，例如 Ubuntu：
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev
```

驗證安裝：

```python
import sdl2
import sdl2.ext

print(sdl2.SDL_GetVersion)  # 能正常匯入且不報錯即代表安裝成功
```

---

## 2. 基本視窗建立

SDL2 的核心流程固定是：**初始化 → 建立視窗 → 建立渲染器 → 事件迴圈 → 清理資源**。
這個順序不能顛倒，因為後面的物件（渲染器）依賴前面的物件（視窗）存在。

```python
import sdl2
import sdl2.ext
import ctypes

def main():
    # 1. 初始化 SDL2 的影像子系統
    #    SDL_INIT_VIDEO 只啟動視窗/繪圖相關功能，若要用音效需另外加 SDL_INIT_AUDIO
    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

    # 2. 建立視窗
    #    參數依序為：標題、x座標、y座標、寬、高、旗標
    #    SDL_WINDOWPOS_CENTERED 讓視窗自動置中，不用自己算螢幕座標
    window = sdl2.SDL_CreateWindow(
        b"SDL2 Python Demo",
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        800, 600,
        sdl2.SDL_WINDOW_SHOWN
    )

    # 3. 建立渲染器（負責實際把圖形畫到視窗上）
    #    -1 代表讓 SDL2 自動挑選第一個支援的繪圖驅動（OpenGL / Direct3D 等）
    #    SDL_RENDERER_ACCELERATED 代表優先使用硬體加速
    renderer = sdl2.SDL_CreateRenderer(
        window, -1, sdl2.SDL_RENDERER_ACCELERATED
    )

    running = True
    event = sdl2.SDL_Event()  # 用來接收事件資料的緩衝區，迴圈中重複使用同一個物件

    while running:
        # 4. 事件迴圈：把所有排隊中的事件一次處理完
        #    SDL_PollEvent 是非阻塞的，沒有事件就馬上回傳 0，適合遊戲/即時渲染
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False

        # 5. 清除畫面（設定背景色並填滿），數值為 R, G, B, A
        sdl2.SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255)
        sdl2.SDL_RenderClear(renderer)

        # ↑↑↑ 這裡之後放每一幀要畫的內容 ↑↑↑

        # 6. 把繪製結果從後緩衝區交換到畫面上（雙緩衝技術，避免畫面閃爍/撕裂）
        sdl2.SDL_RenderPresent(renderer)

    # 7. 依建立順序反向釋放資源，避免記憶體洩漏或懸空指標
    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()

if __name__ == "__main__":
    main()
```

**重點說明：**
- SDL2 的字串參數必須是 `bytes`（前面加 `b"..."`），因為底層是 C 的 `char*`。
- `ctypes.byref(event)` 相當於 C 語言的 `&event`，把記憶體位址傳給 C 函式讓它寫入資料。
- 資源釋放的順序要跟建立順序相反：先建立的視窗最後才銷毀。

---

## 3. 繪製圖形（矩形、線條、點）

```python
# 畫一個實心矩形
rect = sdl2.SDL_Rect(100, 100, 200, 150)  # x, y, 寬, 高
sdl2.SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255)  # 設定畫筆顏色為紅色
sdl2.SDL_RenderFillRect(renderer, rect)   # 填滿矩形

# 畫一個空心矩形（只有外框）
rect2 = sdl2.SDL_Rect(350, 100, 200, 150)
sdl2.SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255)
sdl2.SDL_RenderDrawRect(renderer, rect2)

# 畫一條線
sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255)
sdl2.SDL_RenderDrawLine(renderer, 0, 0, 800, 600)

# 畫一個點
sdl2.SDL_RenderDrawPoint(renderer, 400, 300)
```

> **邏輯提醒**：SDL2 沒有「圖層」概念，畫面是依照程式呼叫繪圖函式的**先後順序**疊加，
> 後畫的會蓋在先畫的上面，所以繪圖順序等同於視覺上的 z-order。

---

## 4. 載入並顯示圖片（紋理 Texture）

需要額外安裝 `SDL2_image` 擴充模組來支援 PNG/JPG 等格式。

```bash
pip install pysdl2-dll  # 已包含 SDL2_image 的 dll，若用系統套件則要另裝 libsdl2-image-dev
```

```python
import sdl2.sdlimage as sdlimage

# 初始化 image 子系統，指定要支援的格式
sdlimage.IMG_Init(sdlimage.IMG_INIT_PNG | sdlimage.IMG_INIT_JPG)

# 從檔案載入圖片，直接轉成可繪製的 Texture（比先載入 Surface 再轉換更省一步）
texture = sdlimage.IMG_LoadTexture(renderer, b"player.png")

# 取得紋理的原始寬高，之後畫圖時才知道要給多大的目標範圍
w, h = ctypes.c_int(0), ctypes.c_int(0)
sdl2.SDL_QueryTexture(texture, None, None, ctypes.byref(w), ctypes.byref(h))

# 在事件迴圈中每一幀畫出這張紋理
dst_rect = sdl2.SDL_Rect(200, 150, w.value, h.value)  # 目標繪製位置與大小
sdl2.SDL_RenderCopy(renderer, texture, None, dst_rect)
# 第三個參數 None 代表使用整張圖片作為來源（不裁切）
# 若只想擷取圖片的一部分（例如 sprite sheet 切割單一幀），可傳入來源 SDL_Rect

# 用完記得銷毀紋理，釋放 GPU 記憶體
sdl2.SDL_DestroyTexture(texture)
sdlimage.IMG_Quit()
```

---

## 5. 鍵盤與滑鼠輸入處理

```python
while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
    if event.type == sdl2.SDL_QUIT:
        running = False

    elif event.type == sdl2.SDL_KEYDOWN:
        # event.key.keysym.sym 是按下的按鍵程式碼
        if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
            running = False
        elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
            player_x += 5

    elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
        # event.button.x / event.button.y 是滑鼠點擊當下的座標
        print(f"滑鼠點擊於: ({event.button.x}, {event.button.y})")

# 另一種方式：直接查詢「目前」鍵盤狀態（適合需要持續移動的情境，例如角色按住方向鍵移動）
keystate = sdl2.SDL_GetKeyboardState(None)
if keystate[sdl2.SDL_SCANCODE_LEFT]:
    player_x -= 5
```

**差異說明：**
- `SDL_KEYDOWN` 事件是「按下的瞬間」觸發一次，適合觸發單次動作（跳躍、開火）。
- `SDL_GetKeyboardState` 是查詢「當下」按鍵是否被壓著，適合連續移動這種需要每幀更新的邏輯。

---

## 6. 播放音效與音樂

需要 `SDL2_mixer` 擴充模組。

```python
import sdl2.sdlmixer as sdlmixer

# 開啟音訊裝置：取樣率、格式、聲道數（2=立體聲）、緩衝區大小
sdlmixer.Mix_OpenAudio(44100, sdlmixer.MIX_DEFAULT_FORMAT, 2, 2048)

# 播放短音效（載入到記憶體，適合音效類）
chunk = sdlmixer.Mix_LoadWAV(b"jump.wav")
sdlmixer.Mix_PlayChannel(-1, chunk, 0)  # -1 代表自動挑選空閒的播放聲道，0 代表不重複播放

# 播放背景音樂（串流播放，適合長音樂檔，不會整個載入記憶體）
music = sdlmixer.Mix_LoadMUS(b"bgm.mp3")
sdlmixer.Mix_PlayMusic(music, -1)  # -1 代表無限循環播放

# 結束時釋放資源
sdlmixer.Mix_FreeChunk(chunk)
sdlmixer.Mix_FreeMusic(music)
sdlmixer.Mix_CloseAudio()
```

---

## 7. 控制影格速率（Frame Rate）

SDL2 不會自動幫你限制 FPS，需要自己用延遲來控制，避免程式在高效能機器上跑到幾千 FPS 浪費資源。

```python
FPS = 60
frame_delay = 1000 // FPS  # 每一幀應該花費的毫秒數

while running:
    frame_start = sdl2.SDL_GetTicks()  # 記錄這一幀開始的時間戳（毫秒）

    # ...事件處理、繪圖邏輯...

    sdl2.SDL_RenderPresent(renderer)

    frame_time = sdl2.SDL_GetTicks() - frame_start  # 這一幀實際花費的時間
    if frame_delay > frame_time:
        sdl2.SDL_Delay(frame_delay - frame_time)  # 補足剩餘時間，讓每幀間隔穩定
```

---

## 8. 完整範例：可移動的方塊

整合上述所有觀念的小範例，按方向鍵可以移動一個紅色方塊。

```python
import sdl2
import sdl2.ext
import ctypes

def main():
    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

    window = sdl2.SDL_CreateWindow(
        b"Move the Box",
        sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
        800, 600, sdl2.SDL_WINDOW_SHOWN
    )
    renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.SDL_RENDERER_ACCELERATED)

    # 方塊的邏輯狀態獨立於繪圖之外，方便之後擴充碰撞偵測等邏輯
    box_x, box_y = 400, 300
    speed = 5

    running = True
    event = sdl2.SDL_Event()

    while running:
        frame_start = sdl2.SDL_GetTicks()

        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False

        # 用持續查詢鍵盤狀態的方式，讓移動更平滑（每幀都判斷一次）
        keystate = sdl2.SDL_GetKeyboardState(None)
        if keystate[sdl2.SDL_SCANCODE_LEFT]:
            box_x -= speed
        if keystate[sdl2.SDL_SCANCODE_RIGHT]:
            box_x += speed
        if keystate[sdl2.SDL_SCANCODE_UP]:
            box_y -= speed
        if keystate[sdl2.SDL_SCANCODE_DOWN]:
            box_y += speed

        # 清畫面
        sdl2.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255)
        sdl2.SDL_RenderClear(renderer)

        # 畫方塊
        rect = sdl2.SDL_Rect(box_x, box_y, 50, 50)
        sdl2.SDL_SetRenderDrawColor(renderer, 255, 80, 80, 255)
        sdl2.SDL_RenderFillRect(renderer, rect)

        sdl2.SDL_RenderPresent(renderer)

        # 限制 60 FPS
        frame_time = sdl2.SDL_GetTicks() - frame_start
        if frame_time < 16:
            sdl2.SDL_Delay(16 - frame_time)

    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()

if __name__ == "__main__":
    main()
```

---

## 9. 常見問題排查

| 問題 | 可能原因 |
|---|---|
| `OSError: could not find any library for SDL2` | 沒裝原生 dll/so，執行 `pip install pysdl2-dll` 或系統套件安裝 |
| 視窗建立後立刻閃退 | 忘記呼叫 `SDL_Init`，或初始化失敗但沒檢查回傳值 |
| 圖片載入回傳 `None` | 檔案路徑錯誤，或忘記先呼叫 `IMG_Init` |
| 畫面完全不更新 | 忘記呼叫 `SDL_RenderPresent`，只有 `SDL_RenderClear` 不會顯示到畫面上 |
| 字串型別錯誤 | SDL2 的字串參數要用 `bytes`（`b"text"`），不能直接傳 Python 的 `str` |

---

## 10. 與 C/C++ 版本 SDL2 的對應關係

因為 PySDL2 幾乎是原生 API 的直接映射，日後若要移植成 C/C++ 版本，函式名稱幾乎不需要更動，
只差在：
- C 版本要自行 `#include <SDL2/SDL.h>` 並連結 `-lSDL2`。
- Python 版本的字串是 `bytes`，C 版本直接用 `const char*`。
- Python 版本用 `ctypes.byref()` 傳址，C 版本用 `&變數` 即可。

這代表若之後要把原型（prototype）先用 Python 快速驗證邏輯，再改寫成 C/C++ 正式版本，
API 呼叫方式幾乎可以逐行對照翻譯，學習成本很低。
