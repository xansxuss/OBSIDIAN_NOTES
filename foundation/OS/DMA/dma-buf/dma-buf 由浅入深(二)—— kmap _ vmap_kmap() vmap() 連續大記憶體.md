[dma-buf 由浅入深(二)—— kmap _ vmap_kmap() vmap() 連續大記憶體](https://blog.csdn.net/hexiaolong2009/article/details/102596761)

前言
撰寫 dma-buf 驅動程式的三個基本步驟，即 dma_buf_ops、dma_buf_export_info、dma_buf_export()。在本篇中，我們將在 exporter-dummy 驅動的基礎上，對其 dma_buf_ops 的 kmap / vmap 介面進行擴充，藉此示範這兩個介面的使用方式。

dma-buf 只能用於 DMA 硬體存取嗎？
在內核程式碼中，我們最常見到的 dma-buf API 莫過於 dma_buf_attach()、dma_buf_map_attachment()，dma-buf 難道只能給 DMA 硬體來存取嗎？當然不是！dma-buf 本質上是 buffer 與 file 的結合，因此它仍然是一塊 buffer。不要因為它帶有 dma 字樣就被誤導了，dma-buf 不僅能用於 DMA 硬體存取，也同樣適用於 CPU 軟體存取，這也是 dma-buf 在內核中大受歡迎的一個重要原因。

正因如此，我才決定將 dma_buf_kmap() / dma_buf_vmap() 作為 dma-buf 系列教學的第二篇文章來說明，因為這兩個介面用起來實在比 DMA 操作介面簡單太多了！

dma-buf 只能分配離散 buffer 嗎？
當然不是！就和內核中的 dma-mapping 介面一樣，dma-buf 既可以是物理連續的 buffer，也可以是離散的 buffer，這最終取決於 exporter 驅動採用何種方式來分配 buffer。
因此，為了讓讀者更容易理解，本篇特別使用了內核中最簡單、最常見的 kzalloc() 函式來分配 dma-buf，自然，這塊 buffer 就是物理連續的。

CPU Access？
從 Linux-3.4 開始，dma-buf 引入了 CPU 操作介面，使得開發人員可以在內核空間中直接使用 CPU 來存取 dma-buf 的實體記憶體。

dma-buf: add support for kernel cpu access

以下 dma-buf API 實現了 CPU 在內核空間中對 dma-buf 記憶體的存取：

dma_buf_kmap()

dma_buf_kmap_atomic()

dma_buf_vmap()

（它們的反向操作分別對應各自的 unmap 介面）

透過 dma_buf_kmap() / dma_buf_vmap() 操作，就可以把實際的實體記憶體映射到 kernel 空間，並轉換成 CPU 可以連續存取的虛擬位址，方便後續軟體直接讀寫這塊實體記憶體。因此，無論這塊 buffer 在物理上是否連續，在經過 kmap / vmap 映射後，其虛擬位址一定是連續的。

上述三個介面分別與 Linux 記憶體管理子系統（MM）中的 kmap()、kmap_atomic() 與 vmap() 函式一一對應，其差異如下：

| 函式            | 說明                                            |
| ------------- | --------------------------------------------- |
| kmap()        | 一次只能映射 1 個 page，可能會睡眠，只能在行程（process）上下文中呼叫    |
| kmap_atomic() | 一次只能映射 1 個 page，不會睡眠，可在中斷（interrupt）上下文中呼叫    |
| vmap()        | 一次可映射多個 pages，且這些 pages 在物理上可以不連續，只能在行程上下文中呼叫 |


1. 從 Linux-4.19 開始，dma_buf_kmap_atomic() 不再被支援。

2. dma_buf_ops 中的 map / map_atomic 介面名稱，其實原本就叫 kmap / kmap_atomic，只是後來發現與 highmem.h 中的巨集定義同名，為了避免開發人員在自己的驅動中引用 highmem.h 時產生命名衝突問題，因此移除了前面的「k」字。

想更深入了解 kmap()、vmap() 的相關資訊，推薦閱讀參考資料中的〈Linux 內核記憶體管理架構〉一文。

範例程式？
本範例分為 exporter 與 importer 兩個驅動。

首先是 exporter 驅動，我們基於上一篇的 exporter-dummy.c，對其 exporter_kmap() 與 exporter_vmap() 進行擴充，具體如下：
exporter-kmap.c

``` cpp
#include <linux/dma-buf.h>
#include <linux/module.h>
#include <linux/slab.h>

struct dma_buf *dmabuf_exported;
EXPORT_SYMBOL(dmabuf_exported);

static struct sg_table *exporter_map_dma_buf(struct dma_buf_attachment *attachment,
      enum dma_data_direction dir)
{
 return NULL;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment *attachment,
          struct sg_table *table,
          enum dma_data_direction dir)
{
}

static void exporter_release(struct dma_buf *dmabuf)
{
 kfree(dmabuf->priv);
}

static void *exporter_vmap(struct dma_buf *dmabuf)
{
 return dmabuf->priv;
}

static void *exporter_kmap_atomic(struct dma_buf *dmabuf, unsigned long page_num)
{
 return dmabuf->priv;
}

static void *exporter_kmap(struct dma_buf *dmabuf, unsigned long page_num)
{
 return dmabuf->priv;
}

static int exporter_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
 return -ENODEV;
}

static const struct dma_buf_ops exp_dmabuf_ops = {
 .map_dma_buf = exporter_map_dma_buf,
 .unmap_dma_buf = exporter_unmap_dma_buf,
 .release = exporter_release,
 .map = exporter_kmap,
 .map_atomic = exporter_kmap_atomic,
 .vmap = exporter_vmap,
 .mmap = exporter_mmap,
};

static struct dma_buf *exporter_alloc_page(void)
{
 DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
 struct dma_buf *dmabuf;
 void *vaddr;

 vaddr = kzalloc(PAGE_SIZE, GFP_KERNEL);
 if (!vaddr)
  return NULL;

 exp_info.ops = &exp_dmabuf_ops;
 exp_info.size = PAGE_SIZE;
 exp_info.flags = O_CLOEXEC;
 exp_info.priv = vaddr;

 dmabuf = dma_buf_export(&exp_info);
 if (IS_ERR(dmabuf)) {
  kfree(vaddr);
  return NULL;
 }

 sprintf(vaddr, "hello world!");

 return dmabuf;
}

static int __init exporter_init(void)
{
 dmabuf_exported = exporter_alloc_page();
 if (!dmabuf_exported) {
  pr_err("error: exporter alloc page failed\n");
  return -ENOMEM;
 }

 return 0;
}

static void __exit exporter_exit(void)
{
 pr_info("exporter exit\n");
}

module_init(exporter_init);
module_exit(exporter_exit);

MODULE_AUTHOR("Leon He <343005384@qq.com>");
MODULE_DESCRIPTION("DMA-BUF Exporter example for cpu-access (kmap/vmap)");
MODULE_LICENSE("GPL v2");
```

接著我們再撰寫一個 importer 驅動，用來示範如何在 kernel 空間中，透過 dma_buf_kmap() / dma_buf_vmap() 介面操作 exporter 驅動所匯出的 dma-buf。
importer-kmap.c

``` cpp
#include <linux/dma-buf.h>
#include <linux/module.h>
#include <linux/slab.h>

extern struct dma_buf *dmabuf_exported;

static int importer_test(struct dma_buf *dmabuf)
{
 void *vaddr;

 if (!dmabuf) {
  pr_err("dmabuf_exported is null\n");
  return -EINVAL;
 }

 vaddr = dma_buf_kmap(dmabuf, 0);
 pr_info("read from dmabuf kmap: %s\n", (char *)vaddr);
 dma_buf_kunmap(dmabuf, 0, vaddr);

 vaddr = dma_buf_vmap(dmabuf);
 pr_info("read from dmabuf vmap: %s\n", (char *)vaddr);
 dma_buf_vunmap(dmabuf, vaddr);

        return 0;
}

static int __init importer_init(void)
{
 return importer_test(dmabuf_exported);
}

static void __exit importer_exit(void)
{
 pr_info("importer exit\n");
}

module_init(importer_init);
module_exit(importer_exit);

MODULE_AUTHOR("Leon He <343005384@qq.com>");
MODULE_DESCRIPTION("DMA-BUF Importer example for cpu-access (kmap/vmap)");
MODULE_LICENSE("GPL v2");
```

範例說明：

exporter 透過 kzalloc 分配了一個 PAGE 大小的物理連續 buffer，並向該 buffer 寫入了「hello world!」字串；

importer 驅動透過 extern 關鍵字匯入 exporter 的 dma-buf，並透過 dma_buf_kmap()、dma_buf_vmap() 函式讀取該 buffer 的內容並輸出到終端顯示。

開發環境

| 內核原始碼 | [4.14.143](https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.14.143.tar.xz)                              |
| ----- | -------------------------------------------------------------------------------------------------------------------- |
| 範例原始碼 | [hexiaolong2008-GitHub/sample-code/dma-buf/02](https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/02) |
| 開發平台  | Ubuntu14.04/16.04                                                                                                    |
| 執行平台  | [my-qemu 模擬環境](https://github.com/hexiaolong2008/my-qemu)                                                            |


執行
在 my-qemu 模擬環境中執行以下指令：

``` bash
insmod /lib/modules/4.14.143/kernel/drivers/dma-buf/exporter-kmap.ko
insmod /lib/modules/4.14143/kernel/drivers/dma-buf/importer-kmap.ko

將會看到如下的輸出結果：

``` bash
read from dmabuf kmap: hello world!
read from dmabuf vmap: hello world!
```
注意：執行 insmod 指令時，必須先載入 exporter-kmap.ko，再載入 importer-kmap.ko，否則會出現符號相依錯誤。
或者也可以直接使用「modprobe importer_kmap」指令來自動解決模組相依問題。

結語
dma_buf_kmap()、dma_buf_vmap() 函式的底層實作，以及如何使用這兩個 API，是 CPU 在 kernel 空間存取 dma-buf 的典型代表。在下一篇中，我們將一起學習如何透過 DMA 硬體來存取 dma-buf 的實體記憶體。

參考資料
[Linux 內核記憶體管理架構](https://www.cnblogs.com/wahaha02/p/9392088.html)
[i915 drm selftests](https://elixir.bootlin.com/linux/v4.14.143/source/drivers/gpu/drm/i915/selftests/mock_dmabuf.c)
