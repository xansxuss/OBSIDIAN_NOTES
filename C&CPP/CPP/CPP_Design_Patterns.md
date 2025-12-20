C++_Design_Patterns

下面的分類方式偏「工程現場怎麼用」而不是「教科書式分類」。你會看到每個 pattern 我都會講：

⚙ 核心概念
- 🧩 什麼時候用
- 🧪 C++ 實作重點（含 modern C++ tips）
- ❗ 常見誤用坑

1️⃣ 建構與生命週期管理 (Creation / Resource Patterns)
■ RAII

- ⚙ 用 stack 物件管理所有資源生命週期（檔案、mutex、stream…）
- 🧩 Thread 管理 / GPU 資源 / CUDA Stream / OpenCV Mat / Socket 都靠它。
- 🧪 用 std::unique_ptr、std::shared_ptr。
- ❗ 千萬別用 new/delete（除非你在寫 allocator）。

``` cpp
class File {
public:
    File(const std::string& p) { fp = fopen(p.c_str(), "r"); }
    ~File() { if (fp) fclose(fp); }
private:
    FILE* fp;
};
```

■ Factory / Abstract Factory

- ⚙ 把「建立物件」這件事抽離程式碼，尤其物件結構超大。
- 🧩 常見於 plugin 系統、inference backend 切換 (TensorRT / OnnxRuntime)。
- 🧪 用 std::function + lambda 會比傳統的 class 更乾淨。

``` cpp
using Creator = std::function<std::unique_ptr<Model>()>;

std::map<std::string, Creator> registry;

registry["tensorrt"] = []{ return std::make_unique<TRTModel>(); };
```

■ Builder

- ⚙ 用來建構「參數很多」且「組裝步驟複雜」的類別。
- 🧩 例如：YOLO 推論引擎 config、GStreamer pipeline option。
- 🧪 Method chaining 是 C++ 標配：

``` cpp
auto config = InferConfig{}
    .setBatch(8)
    .setPrecision("fp16")
    .enableProfiler(true);
```

2️⃣ 結構型模式 (Structural Patterns)
■ Singleton（慎用！）

- ⚙ 全域唯一物件。
- 🧪 Modern C++ 版本：

``` cpp
class Global {
public:
    static Global& instance() {
        static Global inst;
        return inst;
    }
private:
    Global() = default;
};
```

- ❗ 請把它當作核廢料一樣小心使用，尤其 multithreading、unit test 會很臭。

■ Adapter

- ⚙ 讓「不同接口」的東西一起工作。
- 🧩 OpenCV Mat ↔ Nvidia NvBufSurface，或你的 GpuMat pipeline。

■ Facade

- ⚙ 給複雜系統一個超簡潔入口。
- 🧪 你的「多 RTSP 推論 pipeline」非常適合做一層 Facade：

``` cpp
class InferenceSystem {
public:
    void init();
    void start();
    std::vector<Object> infer(const cv::Mat&);
};
```

■ Pimpl（Pointer to Implementation）

- ⚙ 隱藏實作細節、加速編譯時間。
- 🧪 大型專案非常好用。

``` cpp
class Engine {
public:
    Engine();
    ~Engine();
    void run();
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};
```

3️⃣ 行為型模式 (Behavior Patterns)
■ Strategy

- ⚙ 把「可替換的行為」用 interface 抽象。
- 🧩 典型用於 preprocessor / postprocessor / anomaly scoring method 切換。

``` cpp
class NMS {
public:
    virtual std::vector<Box> run(const std::vector<Box>&) = 0;
};
```

■ Observer / Publisher–Subscriber

- ⚙ 事件觸發、推播消息。
- 🧩 RTSP frame arrives → preprocess → inference → postprocess
- 🧪 用 C++17 std::function 就好，不必太 OOP。

■ Command

- ⚙ 把「操作」封成物件。
- 🧩 比較常用在「工具列命令 / undo / log pipeline」。
- 🧪 CLI 子命令（你最近在做的）甚至也能套 Command pattern。

■ State Machine

- ⚙ 把狀態拆成獨立類別。
- 🧩 你的 Pixhawk + Jetson 自動飛行控制，強烈建議用 State pattern：
Idle / Tracking / Failsafe / Landing…

4️⃣ Concurrency Patterns（C++ 比較硬核的部分）
■ Thread Pool

- ⚙ 程式碼更乾淨、控制 worker 數量、避免爆掉。
- 🧩 你現在做的 stream pipeline 超適合。

■ Producer–Consumer

- ⚙ 典型串流處理架構（RTSP → frame buffer → inference）。
- 🧪 使用 lock-free queue（如 moodycamel CQ）會比 mutex 爽很多。

■ Active Object

- ⚙ 把工作送到獨立 thread 執行並回傳 future。
- 🧪 C++17/20 的 std::async / std::future 就能達成。

5️⃣ 針對「效能導向系統」的 Patterns（特別給你用的）
■ Zero-Copy Dataflow

不是 GoF，但你每天都在做。
核心思想：資料不被複製，只移交 ownership 或引用。

實作方式：

- GpuMat → cv::cuda::GpuMat → NvBufSurface
- shared_ptr + custom deleter
- move-only handle（CUDA stream / GPU context）

■ Handle–Body（配合 GPU 資源）

類似 Pimpl，但支援：

- custom deleter
- move-only（很像 unique_ptr）
- RAII + 資源移動

■ Pipeline / Filter Chain

- ⚙ 把處理步驟拆成 stage，像 YOLO preprocessing → inference → postprocess。

C++ 設計模式學習路線

- 最重要的三個：RAII、Pimpl、Strategy
- 跨模組大工程：Facade、Adapter、Builder、Factory
- 系統化架構：Observer、State、Command
- 效能與串流：Thread Pool、Producer-Consumer、Active Object
- 嵌入式/高效能特化：Zero-Copy、Handle-Body、Pipeline