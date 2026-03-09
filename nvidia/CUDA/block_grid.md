在 CUDA 裡，你寫的 kernel 不是「跑一次」，而是被大量複製同時跑。每一個複製體都有自己的身分證，靠的就是 grid、block、thread 這三層座標。

先給一個心智模型：
整個 GPU 是一座蜂巢
→ 蜂巢切成很多「街區」（block）
→ 每個街區裡有很多「工蜂」（thread）

三個層級在幹嘛

Thread
最小單位，真的在執行你寫的那幾行 C/CUDA code。
每個 thread 都有：

threadIdx：我在本 block 裡第幾個

自己的 register

Block
thread 的集合，是 CUDA 排程的基本單位。
每個 block 有：

blockIdx：我是第幾個 block

blockDim：我裡面有多少 threads

shared memory（block 內共享，跨 block 不行）

Grid
所有 block 的集合。
整個 kernel 啟動時，你其實是在說：

幫我開這麼多 block，每個 block 裡有這麼多 thread

最常見的一維範例（經典到不能再經典）

``` cpp
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}
```

啟動方式：

``` cpp
int threads = 256;
int blocks  = (N + threads - 1) / threads;
kernel<<<blocks, threads>>>(data, N);
```

這裡發生的事是：

- blockIdx.x：第幾個 block

- blockDim.x：每個 block 256 threads

- threadIdx.x：block 裡第幾條 thread

- idx：全域唯一 thread 編號

一句話總結：
global_idx = blockIdx * blockDim + threadIdx

這行你會背一輩子，像工程師版九九乘法表。

二維 / 三維不是炫技，是現實需求

影像、feature map、tensor，天生就是多維。

2D image kernel

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

啟動：

``` cpp
dim3 block(16, 16);
dim3 grid(
    (width  + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
kernel<<<grid, block>>>(...);
```

這跟 OpenCV 的 (x, y) 幾乎一模一樣，只是每個 pixel 都是平行宇宙分身。

為什麼 block 不能太大也不能太小

這裡開始進入「硬體現實主義」。

- Warp = 32 threads（硬體事實）

- thread 數最好是 32 的倍數

- 常見 block size：128 / 256 / 512

太小：

- GPU 吃不飽

- latency 蓋不起來

太大：

- register / shared memory 爆掉

- occupancy 掉下來

所以 256 threads/block 會變成一種「工程師民俗信仰」，不是迷信，是經驗統計學。

一個你現在就該有的直覺

- grid 決定「做多少工作」

- block 決定「怎麼分工」

- thread 決定「我只負責這一小格」

而 GPU 真正的排程單位不是 thread，是 warp。


