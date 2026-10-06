---
title: C++_this
source: https://claude.ai/chat/b11002b7-f2e5-4939-9d7d-a506f653fcd4
author:
published: 2026-08-17
created: 2026-08-17
description: Claude conversation with 4 messages
tags:
---
### C++ 的 this

`this` 是一個 **指標** ，存在於類別的非靜態成員函式（non-static member function）內部，指向呼叫該函式的物件本身。

#### 基本概念

```cpp
class Point {
public:
    int x, y;

    Point(int x, int y) {
        this->x = x;  // this->x 是成員變數，x 是參數
        this->y = y;
    }

    void print() {
        // this 指向呼叫 print() 的那個物件
    }
};
```

編譯器實際上會把成員函式偷偷加一個隱藏參數，概念上像這樣：

```cpp
// 你寫的
void print() { ... }

// 編譯器實際處理成（概念示意，非真實語法）
void print(Point* this) { ... }
```

所以呼叫 `p.print()` 時，等同於把 `&p` 傳進去當作 `this` 。

#### 常見用途

**1\. 解決名稱遮蔽（shadowing）**

當參數名稱跟成員變數同名時，用 `this->` 明確指出你要的是成員變數。

```cpp
void setX(int x) {
    this->x = x;  // 沒有 this-> 的話，x = x 只是把參數指定給自己
}
```

**2\. 回傳自身參考，做成串接呼叫（method chaining）**

```cpp
class Builder {
public:
    Builder& setA(int v) { a = v; return *this; }
    Builder& setB(int v) { b = v; return *this; }
private:
    int a, b;
};

// 用法
Builder().setA(1).setB(2);
```

**3\. 把自己傳給其他函式**

```cpp
class Node {
public:
    void registerSelf(Container* c) {
        c->add(this);  // 把自己的位址交出去
    }
};
```

**4\. 在成員函式中判斷是否為同一個物件** （常見於 operator= 的自我指定檢查）

```cpp
MyClass& operator=(const MyClass& other) {
    if (this == &other) return *this;  // 避免自我指定造成問題
    // ... 複製邏輯
    return *this;
}
```

#### 需要注意的地方

- `this` 的型別是 `類別名稱*` （在 const 成員函式裡是 `const 類別名稱*` ），是個 **右值** （不能對它取址或指定新值）。
- 只能用在 **非靜態** 成員函式裡； `static` 成員函式沒有 `this` ，因為它不綁定特定物件。
- `this` 本身不是空的（除非你透過未定義行為硬塞 nullptr 進去），正常情況下呼叫成員函式時它一定指向一個有效物件。

因為你偏好不用標準函式庫，這邊範例也都沒用到， `this` 本身是語言核心機制，跟 STL 無關，純粹是類別/物件層級的東西。

---

**You**

``` C++ 
bool NvDecoder::Init(CUcontext cuCtx, cudaVideoCodec codecType) 
{ 
cuCtx\_ = cuCtx;
codecType\_ = codecType;
CUVIDPARSERPARAMS parserParams; 
memset(&parserParams, 0, sizeof(parserParams));
parserParams.CodecType = codecType\_;
parserParams.ulMaxNumDecodeSurfaces = 20;
parserParams.ulMaxDisplayDelay = 4;
parserParams.pUserData = this;
parserParams.pfnSequenceCallback = HandleVideoSequenceThunk; parserParams.pfnDecodePicture = HandlePictureDecodeThunk; parserParams.pfnDisplayPicture = HandlePictureDisplayThunk;
CUresult r = cuvidCreateVideoParser(&parser\_, &parserParams);
if (r!= CUDA\_SUCCESS)
	{
	fprintf(stderr, "cuvidCreateVideoParser failed: %d\\n", r);
	return false;
	} 
return true;
} 
```

這裡的 `this` 用法是一個很典型的 **C 風格 callback 搭配 C++ 物件** 的模式，常稱為「trampoline」或「thunk」技巧。因為 `cuvidCreateVideoParser` 這類 NVIDIA 的 C API 不懂 C++ 的成員函式（成員函式背後其實偷偷帶了 `this` 參數，簽名跟一般 C 函式指標對不上），所以要繞一手。

#### 逐步拆解

**1\. `parserParams.pUserData = this;`**

把「目前這個 `NvDecoder` 物件的位址」存進 `pUserData` （型別通常是 `void*` ）。這一步只是把 `this` 這個指標「寄放」在 NVIDIA driver 那邊，driver 自己不會去動它，只會在呼叫 callback 時原封不動傳回來。

**2\. `parserParams.pfnSequenceCallback = HandleVideoSequenceThunk;`**

這裡指定的必須是 **一般函式指標** （或 `static` 成員函式），不能是非靜態成員函式，因為非靜態成員函式隱含 `this` 參數，函式指標型別對不上。

`HandleVideoSequenceThunk` 通常長這樣：

```cpp
static int CUDAAPI HandleVideoSequenceThunk(void* pUserData, CUVIDEOFORMAT* pFormat) {
    // 把 driver 傳回來的 pUserData 轉型回 NvDecoder*
    return static_cast<NvDecoder*>(pUserData)->HandleVideoSequence(pFormat);
}
```

**3\. 呼叫流程**

之後每當 driver 內部有新的影像序列（sequence）需要處理，它就會呼叫：

```cpp
HandleVideoSequenceThunk(pUserData, pFormat);
```

這裡的 `pUserData` 就是當初存進去的 `this` （也就是那個 `NvDecoder` 物件的位址）。 `thunk` 函式再把 `void*` 轉型回 `NvDecoder*` ，然後呼叫真正的成員函式 `HandleVideoSequence` ，這樣就能在 callback 裡存取到物件的成員變數（例如 `codecType_` 、 `cuCtx_` 等）。

#### 為什麼要這樣繞

C 的函式指標只認得函式位址，沒有「物件」概念。C++ 的非靜態成員函式其實是：

```cpp
int HandleVideoSequence(CUVIDEOFORMAT* pFormat);
// 編譯器內部概念上是
int HandleVideoSequence(NvDecoder* this, CUVIDEOFORMAT* pFormat);
```

多了一個隱藏的 `this` 參數，跟 `int (*)(void*, CUVIDEOFORMAT*)` 這種純 C 函式指標簽名不相容，所以不能直接把 `&NvDecoder::HandleVideoSequence` 塞給 `pfnSequenceCallback` 。

解法就是：

- 用 `pUserData` 這個「使用者自訂資料欄位」把 `this` 偷渡進去
- 用一個 `static` 函式（thunk）當真正的 callback，在裡面把 `void*` 轉型回物件指標，再呼叫成員函式

這是跨語言介面（C API 包 C++ 物件）非常標準的手法，OpenCV、FFmpeg 的一些 callback、GLFW 的 callback 綁定 class 也都常這樣做。