### CUDA 的 loop unrolling
這東西聽起來好像很 fancy，但其實概念超直覺，就是「少用 for 迴圈，能展開就展開」，讓 GPU pipeline 更順、不被控制流卡住。

🔍 什麼是 loop unrolling(迴圈展開)？

簡單說：
原本的迴圈會在執行時「一圈一圈跑」，每圈都要判斷條件、跳回去、再執行下一次。
這些條件判斷和跳轉在 GPU 上其實很浪費，因為 CUDA 的 warp 是 SIMT (Single Instruction Multiple Thread)，所有 thread 要一起執行相同指令，控制流太多就會拖慢。

所以 loop unrolling 的目的就是：
👉「把迴圈攤平成多段連續運算」，減少分支判斷與跳躍，讓 compiler 能更好優化、把 pipeline 塞滿。

🧱 範例來看比較快
🔸 原始版本：

``` cpp
__global__ void sumKernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    for (int i = 0; i < 4; ++i) {
        int j = idx * 4 + i;
        if (j < N) sum += data[j];
    }

    data[idx] = sum;
}
```

這樣的寫法 GPU 每次都得判斷 i < 4，再決定跳回去或離開迴圈。
對 CPU 還好，但在 GPU warp 這種平行架構下就顯得笨重。

🔸 手動展開(manual unrolling)：

``` cpp
__global__ void sumKernel_unrolled(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    int j = idx * 4;
    if (j + 0 < N) sum += data[j + 0];
    if (j + 1 < N) sum += data[j + 1];
    if (j + 2 < N) sum += data[j + 2];
    if (j + 3 < N) sum += data[j + 3];

    data[idx] = sum;
}
```

這就是 loop unrolling 的概念：
把迴圈「攤開」，直接寫成多行，讓 compiler 不用猜測或生成條件分支。
通常能提升 10~30% 的效能，視情況而定。

🧠 CUDA 編譯器內建幫你偷懶

其實你不一定要手動展開，可以用這招：

``` cpp
#pragma unroll 4
for (int i = 0; i < 4; ++i) {
    ...
}
```

讓 nvcc 自己決定展開程度。
這樣比較乾淨，也比較不容易爆 code size。

⚠️ 小提醒：

#pragma unroll 只是「建議」編譯器，不保證真的展開。
若你的迴圈次數不是常數(像 for (int i = 0; i < N; ++i)，而 N 是變數)，nvcc 通常不會展開。
過度展開會造成 register 爆炸、L1 cache miss 變多，要 balance。

🧩 實戰建議

| 場景                                         | 建議                                            |
| -------------------------------------------- | ----------------------------------------------- |
| 小範圍固定次數(例如 RGB channel、3x3 filter) | ✅ 手動展開                                      |
| 大範圍(例如影像列掃描)                       | ⚠️ 適度使用 `#pragma unroll`，不要手動展太多     |
| 內部有 memory access(global memory)          | 🚀 展開通常能幫助合併存取(coalescing)與 pipeline |
| 有分支條件(if/else)                          | ⚠️ 小心 warp divergence，展開反而可能更慢        |

🧩 延伸話題：loop unrolling + CUDA warp

其實 loop unrolling 常搭配 warp-level primitive(像 shuffle、reduce)，
讓每個 warp 自己完成一整段任務，例如：

``` cpp
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

這裡每個 iteration 都有固定的 offset，
compiler 通常會直接展開成多條 shfl 指令，沒有分支開銷。

🏁 總結

| 項目      | 說明                                           |
| --------- | ---------------------------------------------- |
| 定義      | 把迴圈內容攤平，減少分支、提高 pipeline 利用率 |
| CUDA 工具 | `#pragma unroll` 或手動展開                    |
| 優點      | 提升效能、幫助 memory coalescing               |
| 缺點      | code size 膨脹、register 壓力上升              |
| 實務訣竅  | 展開固定次數的小迴圈最划算，動態的就別亂搞     |

### #pragma unroll控制區域

1️⃣ 基本語法

``` cpp
#pragma unroll N   // N 是展開次數，必須是常數
for (int i = 0; i < 4; ++i) {
    ...
}
```

或完全交給 compiler 自行判斷：

``` cpp
#pragma unroll
for (int i = 0; i < 4; ++i) {
    ...
}
```

重點：它只控制緊接著的單一迴圈。

2️⃣ 控制範圍
🔹 單一迴圈

``` cpp
#pragma unroll 4
for (int i = 0; i < 4; ++i) {
    doSomething(i);
}
```

- #pragma unroll 只作用於下一個 for。
- 如果沒有迴圈，#pragma unroll 就會被忽略。

🔹 多層迴圈

``` cpp
#pragma unroll 2
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        doSomething(i, j);
    }
}
```

- 這裡 #pragma unroll 2 只對 i 迴圈有效，內層 j 迴圈不會自動展開。
- 如果你想展開內層迴圈，也要另外加：

``` cpp
for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        doSomething(i, j);
    }
}
```

小提醒：CUDA compiler 並不保證一定展開，#pragma unroll 是「建議」，但對固定常數迴圈幾乎都會遵守。

🔹 條件使用
你也可以用 #pragma unroll 1 來取消展開：

``` cpp
#pragma unroll 1
for (int i = 0; i < N; ++i) {
    doSomething(i);
}
```

- 這等於告訴 compiler 不要展開，即便它本來想展開也不展開。
- 用在動態迴圈或 register 壓力大時很實用。

3️⃣ 小技巧
1. 緊接在 for 上方

``` cpp
#pragma unroll
for(...) {...}
```

不能放太遠，否則會被忽略。
2. 多層迴圈各自加 pragma
   - 如果你有 2~3 層小迴圈，想全部展開，每層迴圈都要加上。
3. 用常數迴圈 + pragma
    - 例如 RGB channel、kernel filter size（3x3、5x5）特別適合。

簡單說：
#pragma unroll 的控制區域就是它下面第一個 for 迴圈，不會自動延伸到內層迴圈或其他區塊，想展開多層就每層都加一個。
