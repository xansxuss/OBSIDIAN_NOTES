[dma-buf 由浅入深(三)—— map attachment_dma-buf_ api](https://blog.csdn.net/hexiaolong2009/article/details/102596772)

前言
前面已介紹如何使用 CPU 在 kernel 空間中存取 dma-buf 的實體記憶體，但通常這種操作方式在內核中出現的頻率並不高，因為 dma-buf 在設計之初就是為了滿足那些「大記憶體存取需求」的硬體而生，例如 GPU / DPU。在這種情境下，如果使用 CPU 直接存取 memory，效能會大幅下降。因此，dma-buf 在內核中出現頻率最高的，仍然是 dma_buf_attach() 與 dma_buf_map_attachment() 這兩個介面。本篇我們就一起來學習，如何透過這兩個 API 來實現 DMA 硬體對 dma-buf 實體記憶體的存取。

DMA Access
dma-buf 提供給 DMA 硬體存取的 API 主要就兩個：
dma_buf_attach()
dma_buf_map_attachment()

這兩個介面的呼叫有嚴格的先後順序，必須先 attach，再 map attachment，因為後者的參數是由前者所提供，因此這兩個介面通常形影不離。

與上述兩個 API 相對應的反向操作介面為：dma_buf_detach() 與 dma_buf_unmap_attachment()，此處不再贅述。

sg_table
由於 DMA 操作涉及到內核中 dma-mapping 的諸多介面與概念，本篇刻意避重就輕，不打算深入說明。但 sg_table 這個概念必須特別提一下，因為它是 dma-buf 提供給 DMA 硬體存取的最終目標，也是 DMA 硬體存取「離散記憶體」的唯一途徑。

sg_table 本質上是一個由多個「單一物理連續 buffer」所組成的鏈結串列，但整體來看卻是離散的，因此它可以很好地描述從高端記憶體中分配出的離散 buffer；當然，它同樣也可以用來描述從低端記憶體中分配出的物理連續 buffer，如下圖所示：

sg_table 代表整個鏈結串列，而其中的每一個節點則由 scatterlist 表示。因此，一個 scatterlist 就對應著一塊「物理連續」的 buffer。我們可以透過以下介面取得一個 scatterlist 對應 buffer 的實體位址與長度：

sg_dma_address(sgl)
sg_dma_len(sgl)

取得 buffer 的實體位址與長度後，就可以將這兩個參數設定到 DMA 硬體暫存器中，進而實現 DMA 硬體對這一小塊 buffer 的存取。那要如何存取整塊離散 buffer 呢？答案很直觀：使用 for 迴圈，不斷解析 scatterlist，不斷設定 DMA 硬體暫存器即可。

對於現代多媒體硬體而言，IOMMU 的出現解決了程式設計師撰寫 for 迴圈的煩惱。因為在 for 迴圈中，每次設定完 DMA 硬體暫存器後，都必須等待本次 DMA 傳輸完成，才能進行下一次迴圈，這會大幅降低軟體執行效率。而 IOMMU 的功能正是用來解析 sg_table，它會將 sg_table 內部一塊塊離散的小 buffer，映射到自身的裝置位址空間中，使得整塊 buffer 在裝置位址空間中呈現為連續的。

如此一來，在存取離散 buffer 時，只需將 IOMMU 映射後的裝置位址（注意，這與 MMU 映射後的 CPU 虛擬位址並不是同一概念）以及整塊 buffer 的 size 設定到 DMA 硬體暫存器中即可，中途無需反覆設定，便能完成 DMA 硬體對整塊離散 buffer 的存取，大幅提升軟體效率。

dma_buf_attach()
此函式實際上是「dma-buf attach device」的縮寫，用於建立 dma-buf 與 device 之間的連結關係。這個連結關係會被存放在新建立的 dma_buf_attachment 物件中，供後續呼叫 dma_buf_map_attachment() 使用。

此函式對應 dma_buf_ops 中的 attach 回呼介面，如果 device 對後續的 map attachment 操作沒有任何特殊需求，則可以不實作。

dma_buf_map_attachment()
此函式實際上是「dma-buf map attachment into sg_table」的縮寫，主要完成兩件事情：

產生 sg_table

同步 Cache

之所以回傳 sg_table 而不是直接回傳實體位址，是為了相容所有 DMA 硬體（無論是否支援 IOMMU），因為 sg_table 既可以表示連續的實體記憶體，也可以表示非連續的實體記憶體。

Cache 同步的目的，是為了避免該 buffer 先前曾被 CPU 填寫過，而資料仍暫存在 Cache 中而非 DDR，導致 DMA 存取到的不是最新、有效的資料。透過將 Cache 中的資料回寫到 DDR，可以避免此類問題。同樣地，在 DMA 存取完成後，需要將 Cache 設為無效，確保後續 CPU 是直接從 DDR（而非 Cache）讀取資料。

通常會使用以下的串流 DMA 映射介面來完成 Cache 同步：

dma_map_single() / dma_unmap_single()
dma_map_page() / dma_unmap_page()
dma_map_sg() / dma_unmap_sg()

關於 dma_map_*() 函式的更多說明，可參考〈Linux 記憶體管理 —— DMA 與一致性快取〉。

dma_buf_map_attachment() 對應 dma_buf_ops 中的 map_dma_buf 回呼介面，該回呼介面（包含 unmap_dma_buf）是被強制要求實作的，否則 dma_buf_export() 會執行失敗。

為什麼需要 attach 操作？
同一個 dma-buf 可能會被多個 DMA 硬體存取，而每個 DMA 硬體可能因自身能力限制，對 buffer 有不同的要求。例如，硬體 A 的定址能力僅有 0x0 ~ 0x10000000，而硬體 B 的定址能力為 0x0 ~ 0x80000000，那麼在分配 dma-buf 的實體記憶體時，就必須以硬體 A 的能力為基準，這樣 A 與 B 才都能存取這段記憶體。否則，若只滿足 B 的需求，A 就可能無法存取超過 0x10000000 的位址空間，道理其實就像木桶理論。

因此，attach 操作可以讓 exporter 驅動依據不同 device 的硬體能力，選擇最合適的實體記憶體配置方式。

透過設定 device->dma_params 參數，可以告知 exporter 驅動該 DMA 硬體的能力限制。

dma-buf 的實體記憶體通常是在 dma_buf_export() 階段就已分配完成，而 attach 操作只能在 export 之後執行，那要如何確保已分配的記憶體符合硬體能力需求呢？這就引出了下一個問題。

何時分配記憶體？
答案是：既可以在 export 階段分配，也可以在 map attachment 階段分配，甚至兩個階段都分配，這完全取決於 DMA 硬體的能力。

一般流程如下：
首先，驅動開發者需要統計系統中有哪些 DMA 硬體會存取 dma-buf；接著，依據各個 DMA 硬體的能力，決定在何時以及如何分配實體記憶體。

常見策略如下（假設只有 A、B 兩個硬體）：
如果硬體 A 與 B 的定址空間有交集，則在 export 階段分配記憶體，並以 A / B 的交集為準；
如果硬體 A 與 B 的定址空間沒有交集，則只能在 map attachment 階段分配記憶體。

對於第二種策略，由於 A 與 B 的定址空間完全沒有交集，實際上無法直接共享記憶體。此時的解法是：A 與 B 在 map attachment 階段各自分配實體記憶體，再透過 CPU 或通用 DMA 硬體，將 A 的 buffer 內容複製到 B 的 buffer 中，以此間接實現 buffer 的「共享」。

還有另一種策略是：不論條件如何，先在 export 階段分配記憶體，然後在第一次 map attachment 階段，透過 dma_buf->attachments 鏈結串列，逐一比對所有 device 的能力；若符合則直接回傳 sg_table，若不符合則重新分配符合所有 device 需求的實體記憶體，再回傳新的 sg_table。

關於 dma_buf_map_attachment() 的更多說明，可參考
ELCE-DMABUF.pdf

範例程式碼
本範例基於第一篇的 exporter-dummy.c 修改，實作 dma_buf_ops 中的 attach 與 map_dma_buf 回呼介面。為了方便示範，仍然和前面一樣，在 exporter_alloc_page() 中預先分配好 dma-buf 的實體記憶體。

[exporter-sg.c](https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/03/exporter-sg.c) 

``` cpp
#include <linux/dma-buf.h>
#include <linux/module.h>
#include <linux/slab.h>

struct dma_buf *dmabuf_exported;
EXPORT_SYMBOL(dmabuf_exported);

static int exporter_attach(struct dma_buf *dmabuf, struct device *dev,
			struct dma_buf_attachment *attachment)
{
	pr_info("dmabuf attach device: %s\n", dev_name(dev));
	return 0;
}

static void exporter_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment)
{
	pr_info("dmabuf detach device: %s\n", dev_name(attachment->dev));
}

static struct sg_table *exporter_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction dir)
{
	void *vaddr = attachment->dmabuf->priv;
	struct sg_table *table;
	int err;

	table = kmalloc(sizeof(*table), GFP_KERNEL);
	if (!table)
		return ERR_PTR(-ENOMEM);

	err = sg_alloc_table(table, 1, GFP_KERNEL);
	if (err) {
		kfree(table);
		return ERR_PTR(err);
	}

	sg_dma_len(table->sgl) = PAGE_SIZE;
	sg_dma_address(table->sgl) = dma_map_single(NULL, vaddr, PAGE_SIZE, dir);

	return table;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment *attachment,
			       struct sg_table *table,
			       enum dma_data_direction dir)
{
	dma_unmap_single(NULL, sg_dma_address(table->sgl), PAGE_SIZE, dir);
	sg_free_table(table);
	kfree(table);
}

static void exporter_release(struct dma_buf *dmabuf)
{
	kfree(dmabuf->priv);
}

static void *exporter_kmap_atomic(struct dma_buf *dmabuf, unsigned long page_num)
{
	return NULL;
}

static void *exporter_kmap(struct dma_buf *dmabuf, unsigned long page_num)
{
	return NULL;
}

static int exporter_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
	return -ENODEV;
}

static const struct dma_buf_ops exp_dmabuf_ops = {
	.attach = exporter_attach,
	.detach = exporter_detach,
	.map_dma_buf = exporter_map_dma_buf,
	.unmap_dma_buf = exporter_unmap_dma_buf,
	.release = exporter_release,
	.map = exporter_kmap,
	.map_atomic = exporter_kmap_atomic,
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
MODULE_DESCRIPTION("DMA-BUF Exporter example for dma-access");
MODULE_LICENSE("GPL v2");
```

在上面的 attach 實現中,我們僅僅只是輸出了一句 log,其他什麼事情也不做。在 map_dma_buf 實作中,我們建構了一個 sg_table 物件,並透過呼叫 dma_map_single() 來取得 dma_addr 以及實作 Cache 同步操作。
[importer-sg.c](https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/03/importer-sg.c) 
``` cpp
#include <linux/device.h>
#include <linux/dma-buf.h>
#include <linux/module.h>
#include <linux/slab.h>

extern struct dma_buf *dmabuf_exported;

static int importer_test(struct dma_buf *dmabuf)
{
        struct dma_buf_attachment *attachment;
        struct sg_table *table;
	struct device *dev;
	unsigned int reg_addr, reg_size;

	if (!dmabuf)
		return -EINVAL;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;
	dev_set_name(dev, "importer");

        attachment = dma_buf_attach(dmabuf, dev);
        if (IS_ERR(attachment)) {
		pr_err("dma_buf_attach() failed\n");
                return PTR_ERR(attachment);
	}

        table = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);
        if (IS_ERR(table)) {
		pr_err("dma_buf_map_attachment() failed\n");
		dma_buf_detach(dmabuf, attachment);
                return PTR_ERR(table);
        }

	reg_addr = sg_dma_address(table->sgl);
	reg_size = sg_dma_len(table->sgl);
	pr_info("reg_addr = 0x%08x, reg_size = 0x%08x\n", reg_addr, reg_size);

	dma_buf_unmap_attachment(attachment, table, DMA_BIDIRECTIONAL);
	dma_buf_detach(dmabuf, attachment);

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
MODULE_DESCRIPTION("DMA-BUF Importer example for dma-access");
MODULE_LICENSE("GPL v2");
``` 

示例說明：
exporter 透過 kzalloc 分配了一個 PAGE 大小的物理連續 buffer；importer 驅動透過 extern 關鍵字匯入 exporter 的 dma-buf，並透過 dma_buf_map_attachment() 介面取得該實體記憶體對應的 sg_table，接著將 sg_table 中的 address 與 size 解析到 reg_addr 與 reg_size 這兩個虛擬暫存器中。

|內核原始碼 | 4.14.143|
|---|---|
|範例程式碼 | [hexiaolong2008-GitHub/sample-code/dma-buf/01](https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/01)|
|開發平台 | Ubuntu14.04/16.04|
|執行平台 | my-qemu模擬環境|

執行
在 my-qemu 模擬環境中執行以下指令：

``` bash
insmod /lib/modules/4.14.143/kernel/drivers/dma-buf/exporter-sg.ko
insmod /lib/modules/4.14.143/kernel/drivers/dma-buf/importer-sg.ko
```

將看到如下輸出結果：

``` bash
dmabuf attach device: importer
reg_addr = 0x7f6ee000, reg_size = 0x00001000
dmabuf detach device: importer
``` 

注意：執行 insmod 時，必須先載入 exporter-sg.ko，再載入 importer-sg.ko，否則會出現符號相依錯誤。

總結
sg_table 是 DMA 硬體操作的關鍵；
attach 的目的，是讓後續的 map attachment 操作更具彈性；
map attachment 主要完成兩件事：產生 sg_table 與 Cache 同步；
DMA 硬體能力決定了 dma-buf 實體記憶體的分配時機。

透過本篇，我們學習了如何使用 dma_buf_attach() 與 dma_buf_map_attachment() 來實現 DMA 硬體對 dma-buf 的存取。下一篇，我們將一起學習如何在 userspace 中存取 dma-buf 的實體記憶體。

參考資料：
ELCE-DMABUF.pdf

DMA Buffer Sharing – An Introduction

Linux kernel scatterlist API 介紹

Linux 記憶體管理 —— DMA 與一致性快取
