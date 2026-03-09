DAM-buf由淺入深(一)--最簡單的dma-buf驅動程式

歷史:
dma-buf 最初的原型為 shrbuf，由 Marek Szyprowski（Samsung）於 2011 年 8 月 2 日首次提出，他實現了 “Buffer Sharing” 的概念驗證（Proof-of-Concept），並在三星平台的 V4L2 驅動中實現了 camera 與 display 的 buffer 共享問題。該 patch 發佈後，在內核社群引起了巨大反響，因為當時關於 buffer 共享問題很早就開始討論，但由於內核沒有現成的框架支持，導致各個廠商實作的驅動五花八門，此時急需一個統一的框架來解決 buffer 共享問題。

LWN:Buffer sharing proof-of-concept

LWN:Sharing buffers between devices

於是 Sumit Semwal (Linaro) 基於 Marek Szyprowski 的 patch 重構了一套新的框架，也就是我們今天看到的 dma-buf 核心程式碼，它經歷了社群開發者給出的重重考驗，並最終於 2012 年 2 月 merge 到 Linux-3.3 主線版本中，這也是 dma-buf 的第一個正式版本。此後 dma-buf 被廣泛應用於內核多媒體驅動開發中，尤其在 V4L2、DRM 子系統中得到了充分應用。

LWN:DMA buffer sharing in 3.3

Patch:dma-buf: Documentation for buffer sharing framework

Patch:dma-buf: Introduce dma buffer sharing mechanism

第一個使用 dma-buf 的 DRM 分支:drm-prime-dmabuf

概念:
dma-buf 的出現就是為了解決各個驅動之間 buffer 共享的問題，因此它本質上是 buffer 與 file 的結合，即 dma-buf 既是塊物理 buffer，又是個 Linux file。buffer 是內容，file 是媒介，只有透過 file 這個媒介才能實現同一 buffer 在不同驅動之間的流轉。
一個典型的 dma-buf 應用框圖如下:
![[DAM-buf由淺入深(一)--最簡單的dma-buf驅動程式_典型dma-buf應用框架圖.png]]
通常，我們將分配 buffer 的模組稱為 exporter，將使用該 buffer 的模組稱為 importer 或 user。但在本系列文章中，importer 特指內核空間的使用者，user 特指使用者空間的使用者。

有的人習慣將 exporter 說成是生產者，importer 說成是消費者，我個人認為這樣的說法並不嚴謹。舉例來說，Android 系統中，graphic buffer 都是由 ION 來分配的，GPU 負責填充該 buffer，DPU 負責顯示該 buffer。那麼在這裡，ION 則是 exporter，GPU 和 DPU 則都是 importer。但是從生產者/消費者模型來講，GPU 則是生產者，DPU 是消費者，因此不能片面的認為 exporter 就是生產者。

最簡單的 dma-buf 驅動程式
如下程式碼演示了如何編寫一個最簡單的 dma-buf 驅動程式，我將其稱為 dummy 驅動，因為它什麼事情也不做。
注意：該程式碼已經是精簡到不能再精簡，少一行程式碼都不行！
exporter-dummy.c

``` cpp
#include<linux/dma-buf.h>
#include<linux/module.h>

static struct sg_table *exporter_map_dma_buf(struct dma_buf_attachment *attachment, enum dma_data_direction dir)
{
  return <span class="token constant">NULL;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment *attachment, struct sg_table *table, enum dma_data_direction dir)
{
}

static void exporter_release(struct dma_buf *dmabuf)
{
}

static void *exporter_kmap_atomic(struct dma_buf *dmabuf, unsigned long page_num)
{
return <span class="token constant">NULL;
}

static void *exporter_kmap(struct dma_buf *dmabuf, unsigned long page_num)
{return <span class="token constant">NULL;
}

static int exporter_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{return -ENODEV;
}

static const struct dma_buf_ops exp_dmabuf_ops = 
{
 .map_dma_buf = exporter_map_dma_buf,
 .unmap_dma_buf = exporter_unmap_dma_buf,
 .release = exporter_release,
 .map_atomic = exporter_kmap_atomic,
 .map = exporter_kmap,
 .mmap = exporter_mmap,
};

static int __init exporter_init(void)
{
 DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
 struct dma_buf *dmabuf;

 exp_info.ops = &exp_dmabuf_ops;
 exp_info.size = PAGE_SIZE;
 exp_info.flags = O_CLOEXEC;
 exp_info.priv = <span class="token string">&#34;null&#34;;

 dmabuf = dma_buf_export(&exp_info);

 return <span class="token number">0;
}

module_init(exporter_init);
```
從上面的程式碼來看，要實作一個 dma-buf exporter 驅動，需要執行三個步驟：

dma_buf_ops

DEFINE_DMA_BUF_EXPORT_INFO

dma_buf_export()

注意：其中 dma_buf_ops 的回呼接口中，如下接口又是必須要實作的，缺少任何一個都將導致 dma_buf_export() 函數呼叫失敗！

map_dma_buf

unmap_dma_buf

map

map_atomic

mmap

release

從 Linux-4.19 開始，map_atomic 接口被廢棄，map 和 mmap 接口不再被強制要求。
dma-buf: make map_atomic and map function pointers optional

dma-buf: remove kmap_atomic interface

dma-buf: Remove requirement for ops->map() from dma_buf_export

dma-buf: Make mmap callback actually optional

開發環境:

|內核原始碼 | 4.14.143|
|---|---|
|範例程式碼 | [hexiaolong2008-GitHub/sample-code/dma-buf/01](https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/01)|
|開發平台 | Ubuntu14.04/16.04|
|執行平台 | my-qemu模擬環境|

編譯:
dma-buf 的核心程式碼由 CONFIG_DMA_SHARED_BUFFER 宏來控制是否參與編譯，而該 config 並不是一個顯式的選單項，我們無法直接在 menuconfig 選單中找到它，因此這裡直接簡單粗暴地修改 Kconfig 檔案，設定 default y 來實現 dma-buf.c 的強制編譯：

``` bash
linux-4.14.43/drivers/base/Kconfig:
```

``` bash
config DMA_SHARED_BUFFER
<span class="token builtin">bool
default y
```

或者也可以透過 menuconfig 選單選擇那些依賴 dma-buf 的設備驅動，如 DRM VGEM。

然後編譯 exporter_dummy.ko 檔案，並打包到 my-qemu 環境中。

執行:
在 my-qemu 模擬環境中執行如下命令：

``` bash
insmod /lib/modules/4.14.143/kernel/drivers/dma-buf/exporter-dummy.ko
lsmod
```

``` bash
exporter_dummy 16384 1 - Live 0x7f000000
```

透過如下命令來查看 dma-buf 的相關資訊:
cat /sys/kernel/debug/dma_buf/bufinfo

Dma-buf Objects:

|size | flags | mode | count | exp_name|
|---|---|---|---|---|
|00004096 | 00000000 | 00000005 | 00000001 | exporter_dummy|

Attached Devices:
Total 0 devices attached
Total 1 objects, 4096 bytes

執行截圖:
![[DAM-buf由淺入深(一)--最簡單的dma-buf驅動程式_運行截圖.png]]
在實際執行的過程中，細心的小夥伴可能會發現，該 exporter_dummy.ko 只能被 insmod，無法被 rmmod。關於該問題的原因，在 [[dma-buf 由淺入深(五)—— File_dma-buf fd 傳輸.md]] 中說明。

總結:
dma-buf 本質上是 buffer + file 的結合。
編寫 dma-buf 驅動的三個步驟:
(1) dma_buf_ops
(2) DEFINE_DMA_BUF_EXPORT_INFO
(3) dma_buf_export()

透過本篇我們學習了如何編寫一個最簡單的 dma-buf 驅動程式。但是該驅動什麼事情也做不了，因為它的 dma_buf_ops 回呼函數都是空的。從下一篇起，我們將一步步實作 dma_buf_ops 的回呼函數，讓大家逐步掌握 dma-buf 的使用技巧。

參考資料:
敬葉: [Linux DMA-BUF](http://lotleaf.com/kernel/linux-dmabuf.html)