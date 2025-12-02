## abstract class 跟 interface 的差別」，要分語言層面來看才會清楚

🧩 一、C++ 本身沒有「interface」這個關鍵字
在 C++ 裡，「interface」只是一種用法慣例，實際上還是靠「抽象類別（abstract class）」實現的。

換句話說：

在 C++ 中，interface ≒ 全部成員都是純虛函式的抽象類別。

✅ C++ 中的 interface 寫法

``` cpp
class IShape {
public:
    virtual void draw() = 0;
    virtual double area() const = 0;
    virtual ~IShape() = default;
};
```

這樣的 IShape 其實就是「介面（interface）」概念的具體實現。
它只定義規格，不包含任何邏輯或資料成員。

🧠 二、抽象類別 vs 介面 的差異總表

| 比較項目          | 抽象類別（Abstract Class）    | 介面（Interface）                          |
| ------------- | ----------------------- | -------------------------------------- |
| **語言層面**      | C++、Java、C# 都支援         | C++ 無關鍵字（用抽象類別模擬）                      |
| **成員內容**      | 可包含普通成員變數、普通方法、虛函式、純虛函式 | 只能包含純虛函式（規格定義），不可有資料成員                 |
| **繼承方式**      | 單繼承（但可搭配多重繼承多個介面）       | 常用多重繼承（多介面同時繼承）                        |
| **用途**        | 作為「部分實作」的基底類別           | 作為「純規格」的約定                             |
| **能否有成員變數**   | ✅ 可以                    | ❌ 不建議、有違語義                             |
| **能否提供預設實作**  | ✅ 可以                    | ❌ 不應該（否則就不是純介面）                        |
| **C++ 關鍵字實作** | 使用 `virtual ... = 0`    | 沒有 `interface` 關鍵字，用 abstract class 模擬 |

🧬 三、舉例比較
🔹 抽象類別（部分實作）

``` cpp
class AbstractLogger {
protected:
    int logLevel;
public:
    AbstractLogger(int level) : logLevel(level) {}
    virtual void log(const std::string &msg) = 0;
    void setLevel(int level) { logLevel = level; }
};
```

- 有成員變數 (logLevel)
- 有一般方法 (setLevel)
- 有純虛函式 (log)

✅ 可作為共用邏輯的基底類別

🔹 Interface（純規格）

``` cpp
class ILogger {
public:
    virtual void log(const std::string &msg) = 0;
    virtual ~ILogger() = default;
};
```

- ```❌``` 沒有任何資料成員
- ```❌``` 不含具體邏輯
- ✅ 只定義「行為介面」
- ✅ 子類別必須完整實作

⚙️ 四、實務設計建議

| 需求                                | 建議用法                                                     |
| --------------------------------- | -------------------------------------------------------- |
| 你要建立**一組行為規格**讓多類別實作              | 使用「介面風格」的抽象類別                                            |
| 你要建立**一個具有共用邏輯**的父類別              | 使用「抽象類別」並提供部分實作                                          |
| 你要在 C++ 裡模擬 Java/C# 的 `interface` | 就寫「純虛抽象類別」，命名上常以 `I` 開頭，如 `IShape`, `IStream`, `ILogger` |

🔧 小結一句話版：

在 C++ 裡：
💡 「interface」不是語法結構，而是一種設計風格。
它實際上是「沒有成員、沒有實作的抽象類別」。

### For examole 

TensorRT

TensorRT 的 API 架構大量使用了「介面（interface）」風格的抽象類別設計。

換句話說，它用 C++ 的 純虛抽象類別（pure virtual abstract class） 來模擬「interface」概念，讓底層引擎、plugin、builder、network 等模組都透過介面互動，而不是直接依賴實作。

🧩 一、設計哲學：Interface + Factory + Opaque Implementation

NVIDIA 在 TensorRT 的 C++ API 裡面幾乎所有「可操作」的物件，
例如：

- nvinfer1::INetworkDefinition
- nvinfer1::ILayer
- nvinfer1::IBuilder
- nvinfer1::ICudaEngine
- nvinfer1::IExecutionContext
- nvinfer1::IPluginV2
- nvinfer1::IPluginCreator

這些前面都有個大寫 I，其實就是：

「Interface」風格命名的純虛抽象類別。

🧠 二、實際範例

以 TensorRT 最常見的 IPluginV2 來說：

``` cppclass IPluginV2 : public virtual IPluginV2IOExt
{
public:
    // 取得 plugin 名稱
    virtual const char* getPluginType() const noexcept = 0;

    // 取得 plugin 版本
    virtual const char* getPluginVersion() const noexcept = 0;

    // 建立 plugin 的複製品
    virtual IPluginV2* clone() const noexcept = 0;

    // 計算輸出 tensor 的 shape
    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept = 0;

    // 執行 kernel
    virtual int enqueue(...) noexcept = 0;

    // 解構函式虛擬化
    virtual ~IPluginV2() noexcept {}
};
```

這是一個純虛類別（abstract class），裡面幾乎每個成員都是：

``` cpp
virtual ... = 0;
```

⚙️ 三、為什麼 TensorRT 要這樣設計？
✅ 1. 隱藏實作（Encapsulation / ABI 隔離）

TensorRT 的內部是封閉的（closed-source binary），
他們不希望開發者看到或依賴內部類別結構。
介面提供「穩定 API 層」，實作藏在 .so / .dll 裡。

✅ 2. 允許多型擴充（Polymorphic Extension）

像 plugin 系統就是靠這個機制：
你實作自己的 MyConvPlugin : public IPluginV2，
TensorRT runtime 會透過 IPluginCreator factory 動態建立。

這就是「多型 + 工廠模式 + 介面設計」的完美實例。

✅ 3. 版本相容與二進位穩定性（Binary Compatibility）

NVIDIA 改版時，只要不改動介面的函式簽名，
就不會破壞使用者 plugin 的二進位相容性。
→ 很像 COM（Component Object Model）或 Qt interface 的做法。

✅ 4. 支援跨語言綁定

因為 interface 沒有實作、不需要模板參數，
可以安全地包進 Python Binding、C API、Rust FFI 等。

TensorRT 的 Python 版本其實就是這些 interface 的封裝。

🧬 四、整體設計架構範例

``` bash
nvinfer1::IBuilder ------------------┐
                                    │ Factory Pattern
nvinfer1::INetworkDefinition -------┘
        │
        ▼
nvinfer1::ILayer (IConvolutionLayer, IActivationLayer ...)
        │
        ▼
nvinfer1::ICudaEngine
        │
        ▼
nvinfer1::IExecutionContext
```

每一層都是 interface，你不會 new 它，
而是透過 IBuilder::createNetwork()、IBuilder::buildEngine() 這些工廠方法取得具體實作。

💬 五、小結

| 項目           | TensorRT 的做法   | 對應 C++ 概念          |
| ------------ | -------------- | ------------------ |
| 類別開頭 `I`     | Interface 風格命名 | 純虛抽象類別             |
| 不能直接 new     | 工廠產生實例         | Factory Pattern    |
| 定義 API 規格    | 隱藏實作細節         | 封裝 (Encapsulation) |
| 支援 plugin 擴充 | 動態多型           | Polymorphism       |
| 穩定 ABI       | 透過介面層隔離        | Interface 隔離原則     |


💡一句話總結：

TensorRT 用「抽象類別模擬 interface」，
加上「工廠模式」與「多型」，
實現了一個封閉內核、可擴充外殼的架構。

### 兩層抽象設計範例

1. IShape：純粹介面（interface），只定義規格 → 純虛函式，沒有資料成員。
2. BaseShape：抽象基底（abstract base class），提供共用邏輯與部分實作，可包含資料成員。
3. Circle / Rectangle：具體類別，繼承 BaseShape，並實作 IShape 的規格。

🧩 範例程式

``` cpp
#include <iostream>
#include <vector>
#include <memory>

// ------------------------
// 1. Interface 層：定義規格
// ------------------------
class IShape {
public:
    virtual void draw() = 0;
    virtual double area() const = 0;
    virtual ~IShape() = default; // interface 也要虛析構
};

// ------------------------
// 2. Abstract Base 層：提供共用邏輯
// ------------------------
class BaseShape : public IShape {
protected:
    std::string name;  // 共用資料成員
public:
    BaseShape(const std::string& n) : name(n) {}

    // draw 先不實作，保持抽象
    virtual void draw() = 0;

    // area 先不實作，保持抽象
    virtual double area() const = 0;

    void printName() const { std::cout << "Shape: " << name << "\n"; }

    virtual ~BaseShape() = default;
};

// ------------------------
// 3. Concrete Class 層
// ------------------------
class Circle : public BaseShape {
    double radius;
public:
    Circle(double r) : BaseShape("Circle"), radius(r) {}

    void draw() override { std::cout << "畫圓\n"; }

    double area() const override { return 3.14159 * radius * radius; }
};

class Rectangle : public BaseShape {
    double width, height;
public:
    Rectangle(double w, double h) : BaseShape("Rectangle"), width(w), height(h) {}

    void draw() override { std::cout << "畫矩形\n"; }

    double area() const override { return width * height; }
};

// ------------------------
// 使用範例
// ------------------------
int main() {
    std::vector<std::unique_ptr<IShape>> shapes;

    shapes.push_back(std::make_unique<Circle>(3.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 2.0));

    for (auto& s : shapes) {
        // 多型呼叫 draw / area
        s->draw();
        std::cout << "面積: " << s->area() << "\n";

        // 轉 BaseShape 指標可以使用共用邏輯
        if (auto base = dynamic_cast<BaseShape*>(s.get())) {
            base->printName();
        }
    }
}
```

🔹 設計特色

1. IShape：
    - 純介面，只定義規格。
    - 不持有任何狀態。
2. BaseShape：
    - 提供共用資料 (name) 與共用方法 (printName())。
    - 保持抽象（draw()、area() 仍是純虛）。
3. Circle / Rectangle：
    - 具體實作。
    - 可以直接實例化。

💡 優勢

- 介面與抽象分離：清楚區分「規格」與「共用邏輯」。
- 可擴充：以後新增 Triangle 只要繼承 BaseShape 並實作規格即可。
- 多型使用：程式碼只依賴 IShape，不關心具體實作，符合依賴倒置原則 (DIP)。

#### 詳細拆解程式碼

1️⃣ Header 與 STL 引入

``` cpp
#include <iostream>
#include <vector>
#include <memory>
```

- iostream：用於輸出，例如 std::cout。
- vector：用於儲存多個 IShape 指標，方便多型操作。
- memory：用於智慧指標 std::unique_ptr，管理物件生命週期，避免手動 delete。

2️⃣ Interface 層：IShape

``` cpp
class IShape {
public:
    virtual void draw() = 0;
    virtual double area() const = 0;
    virtual ~IShape() = default; // interface 也要虛析構
};
```

- 純虛函式 (=0)：
    - draw() 與 area() 沒有實作。
    - 任何繼承 IShape 的類別都必須實作這兩個方法，否則該類別也會是抽象類別。
- 虛擬解構 (virtual ~IShape())：
    - 必須使用虛擬解構，以確保透過 IShape* 刪除衍生物件時，會呼叫正確的子類析構函式。
- 設計理念：
    - IShape 只是一個「介面」，定義行為規格（規範），不持有任何狀態。

3️⃣ Abstract Base 層：BaseShape

``` cpp
class BaseShape : public IShape {
protected:
    std::string name;  // 共用資料成員
public:
    BaseShape(const std::string& n) : name(n) {}
    
    virtual void draw() = 0;
    virtual double area() const = 0;

    void printName() const { std::cout << "Shape: " << name << "\n"; }

    virtual ~BaseShape() = default;
};
```

- 繼承 IShape：
    - BaseShape 本身是抽象類別，因為它沒有實作 draw() 和 area()。
    - 這層提供了「共用邏輯」，例如 name 與 printName()。
- 成員變數 name：
    - 用於記錄形狀名稱（Circle / Rectangle）。
    - 讓子類可以共用資料，而不用每個子類都自己寫成員變數。
- printName()：
    - 提供共用功能，不需要子類重寫。
- 設計理念：
    - BaseShape 是「抽象基底類別」：有共用資料與方法，但仍保留部分抽象接口，強制子類實作特定行為。

4️⃣ Concrete Class 層：Circle / Rectangle

``` cpp
class Circle : public BaseShape {
    double radius;
public:
    Circle(double r) : BaseShape("Circle"), radius(r) {}

    void draw() override { std::cout << "畫圓\n"; }
    double area() const override { return 3.14159 * radius * radius; }
};
```

- 建構子：
    - BaseShape("Circle")：呼叫父類建構子初始化名稱。
    - radius 初始化。
- 覆寫純虛函式 (override)：
    - draw() 與 area() 提供具體實作。
- Rectangle 同理：
    - 有 width、height，並實作 draw() 與 area()。

5️⃣ 使用範例：多型操作

``` cpp
std::vector<std::unique_ptr<IShape>> shapes;
shapes.push_back(std::make_unique<Circle>(3.0));
shapes.push_back(std::make_unique<Rectangle>(4.0, 2.0));
```


- 使用 unique_ptr<IShape>：
    - 智慧指標自動管理記憶體。
    - 可以存放不同衍生類別，實現「多型」。
- std::make_unique：
    - C++14 之後的標準方式，安全且簡潔。

6️⃣ 遍歷與多型呼叫

``` cpp
for (auto& s : shapes) {
    s->draw();
    std::cout << "面積: " << s->area() << "\n";

    if (auto base = dynamic_cast<BaseShape*>(s.get())) {
        base->printName();
    }
}
```

- s->draw() / s->area()：
    - 透過 IShape* 呼叫，**多型（polymorphism）**作用，執行實際子類函式。
- dynamic_cast<BaseShape*>：
    - 將 IShape* 轉回 BaseShape*，以使用共用功能 printName()。
    - dynamic_cast 安全：如果轉型失敗返回 nullptr。
- 設計理念：
    - 外層程式只依賴介面 IShape，不關心具體類型。
    - 內部可使用抽象基底提供的共用邏輯，減少重複程式碼。

7️⃣ 設計結論
層級設計

``` bash
IShape (interface)  <- 純規格
    │
BaseShape (abstract base)  <- 提供共用資料與方法
    │
Circle / Rectangle (concrete)  <- 具體實作
```

優點

1. 分層清楚：規格 vs 共用邏輯 vs 實作。
2. 支援多型：程式只依賴 IShape，易於擴充。
3. 共用程式碼：共用方法與資料放在 BaseShape，避免重複。
4. 安全管理記憶體：使用 unique_ptr，避免手動 delete。
5. 靈活擴充：新增 Triangle 只需繼承 BaseShape 並實作純虛函式。