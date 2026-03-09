## [TLDR](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#tldr)

DMA-BUF 是使用者程序以檔案（FD）形式從核心引用並可以傳遞給其他程序或核心元件的、不一定位於記憶體（RAM）上的儲存區域。

其主要被用於在使用者空間零複製（Zero-copy）地向某一硬體傳遞引用自相同或另一硬體並可被兩硬體直接訪問的儲存區域，如將 Vulkan 儲存引用的視訊記憶體區域以 DMA-BUF 匯出並匯入為 EGL/OpenGL 儲存。

## [前言](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E5%89%8D%E8%A8%80)

如果你經常關注 Linux 新聞或開源專案動態，你或許曾經不止一次在如下門類中瞥見過“DMA-BUF”。

- V4L2, DRM
- Mesa
- Vulkan
- Wayland
- GNOME, KDE

這些門類往往相連相交，而連線它們的是圖形/影象的渲染、顯示又或是接收。而 DMA-BUF 在其中最大的用處即是以更高的效能進行影象資料的共享、傳遞，這就涉及到了零複製的概念。

## [零複製](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E9%9B%B6%E8%A4%87%E8%A3%BD)

引自維基百科[零複製](https://zh.wikipedia.org/zh-cn/%E9%9B%B6%E5%A4%8D%E5%88%B6)條目

> 零複製（英語：Zero-copy；也譯零複製）技術是指計算機執行操作時，CPU 不需要先將資料從某處記憶體複製到另一個特定區域。

設想現在有 A、B 兩者者，

- A 根據請求生成影象資料
- B 根據請求顯示影象
- 需要將 A 產生的影象傳遞給 B

在影象傳輸的情景中，頻寬動輒 Gbit/s，比如傳輸 1920x1080 32bit(RGBA) 60hz 的未壓縮影片資料，頻寬將達到近 475 MB/s。若是直接使用 CPU 對 A 的影象資料進行復制（memcpy）並傳遞給 B，即使對於現代桌面 CPU 這也是一個資源大戶。更別說日益成為主流的 4K 解析度（2160p）所需的 4 倍帶寬了。

### [memfd](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#memfd)

若 A、B 同屬一程序，即它們共享記憶體地址空間，其實只需將 A 的影象資料地址傳遞給 B 並保證 A、B 不同時使用此資料，與其複製不如共享同一塊儲存區域。

但如果 A、B 是兩個不同的程序呢？這就需要檔案作為中介。

memfd 是由核心建立的以記憶體為後端的匿名檔案（FD），並可以使用 `mmap` 對映到當前程序的虛擬地址空間來對檔案進行讀寫。因此透過在 A、B 之間傳遞引用影象資料的 memfd 即可實現資料零複製共享。

FD 可透過 socket 的 SCM_RIGHTS 訊息在程序間傳遞或使用 `pidfd_getfd` 間接傳遞，也可以使用 D-Bus 等 IPC 通道，網上已有足夠文件故本文不再贅述

#### [示例](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E7%A4%BA%E4%BE%8B)

如下為簡化版的 memfd 傳遞示例。

exporter 建立 memfd 並列印 PID 和 FD 資訊以供其他程式使用

exporter.c

``` cpp
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#define LEN 1024
int main(void)
{
int fd = memfd_create("image", 0);
ftruncate(fd, LEN);
void *data = mmap(NULL, LEN, PROT_WRITE, MAP_SHARED, fd, 0);    // 寫入字串資料
sprintf(data, "Hello World!\n");
printf("/proc/%d/fd/%d\n", getpid(), fd);
pause();
return 0;
}
```

importer 使用 Linux 核心提供的 `/proc/[PID]/fd/[FD]` 開啟其他程序的 FD

importer.c

``` cpp
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#define LEN 1024
int main(int argc, char *argv[])
{
int fd = open(argv[1], O_RDWR);
void *data = mmap(NULL, LEN, PROT_READ, MAP_SHARED, fd, 0);    // 輸出字串資料
printf("%s", (char *)data);
return 0;
}
```

執行 exporter，其列印 fd 檔案路徑，可用 Ctrl + C 結束程式

```$ gcc -o exporter exporter.c && ./exporter  /proc/123456/fd/3 ```

以 exporter 輸出的 fd 路徑為引數執行 importer，成功打印出 exporter 之前寫入 memfd 的內容

`$ gcc -o importer importer.c  $ ./importer /proc/123456/fd/3  Hello World!`

### [DMA-BUF FD](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#dma-buf-fd)

在使用者空間，DMA-BUF 的形式就是 FD，故而它和 memfd 有相似的屬性，它也可以和其他 FD 一樣被傳遞，取決於實現也可能執行 `mmap` 操作。

不過有一點最大的不同就是 DMA-BUF 不能被使用者建立，它是核心模組對儲存區域的抽象引用，而你只能透過核心模組提供的 `ioctl` 介面或檔案介面等匯出、匯入 DMA-BUF FD。

通常只有 DRM 模組、V4L2 模組等圖形相關核心模組會提供 DMA-BUF 匯出或匯入介面以滿足高效能零複製傳遞影象資料需求。使用者可將模組 A 匯出的 DMA-BUF FD 匯入至模組 B 或是模組 A 自身，使用者在此過程中扮演的角色只是路由。

而對於 DMA-BUF 在核心空間的儲存後端和傳遞則是對使用者是不透明的，它的儲存可以位於視訊記憶體、可以位於記憶體、也可以位於硬體獨有儲存，資料的傳遞也不一定透過 DMA 而可以透過 CPU。

## [圖形 API 中的 DMA-BUF](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E5%9C%96%E5%BD%A2-api-%E4%B8%AD%E7%9A%84-dma-buf)

伴隨著 Wayland 對在使用者空間共享螢幕或視窗內容的需求，DMA-BUF 在桌面端軟體的應用也逐漸展開。而這其中的底層，也是 DMA-BUF 的最終來源和去向，就是基於圖形驅動的使用者空間的圖形 API。

其實本文的出發點正是我之前寫的 [pw-capture](https://github.com/EHfive/pw-capture)，一個使用 Vulkan API 和 EGL/GLX API 匯出 DMA-BUF FD 影象流並傳遞給其他程式（PipeWire）的影象擷取層。

對於一張 Vulkan 圖片 `VkImage`，它會有寬、高、層數、畫素格式等屬性，而它又會連結 `VkMemory` 作為儲存。這其中的儲存 `VkMemory` 即可選地能被匯出為 DMA-BUF FD，又或是從 DMA-BUF FD 匯入為 `VkMemory` 從而作為 `VkImage` 的儲存。而 OpenGL 與之類似，只不過是從 Texture 匯出匯入且需要如 EGL 之類的中間層中介。

進而即可利用 DMA-BUF FD 可分享的屬性實現影象儲存的跨程序、跨 API 的傳輸與共享。

### [DRM format modifier](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#drm-format-modifier)

在 Vulkan 和 OpenGL 等圖形應用中，你通常不會使用核心提供介面而是廠商實現的圖形 API 進行 DMA-BUF 操作，故除了純粹的匯出、匯入外，你還會獲取或需要提供如畫素格式、尺寸等圖形相關的元資訊。

而 DRM format modifier 是更底層的描述 DMA-BUF 資料的畫素結構的 64-bit 整數描述符。

對於一張影象畫素在記憶體中的排列，最自然的有線性排列，即按順序排放從第一行到最後一行的資料，資料可按 `x + y * 列寬` 定址。這種線性排列就可以用 `DRM_FORMAT_MOD_LINEAR`（`0x0`）描述。

而如 Intel、AMD 等廠商通常會有私有定義的或許效能更佳的畫素排列，這時他們會選擇一個新的 64-bit 整數作為此排列的描述符以區分其他排列。

而其他私有但不需要區分的排列通常會使用 `DRM_FORMAT_MOD_INVALID` 描述。

通常只有 `DRM_FORMAT_MOD_LINEAR` 格式描述的資料可以被使用者直接解析使用，而其他格式的資料只能被使用者中轉匯出匯入。

### [Vulkan](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#vulkan)

在 Vulkan 中，如果驅動實現支援 [`VK_EXT_external_memory_dma_buf`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html) 擴充套件，則使用者可以利用它匯入和匯出 DMA-BUF FD。

而利用 [`VK_EXT_image_drm_format_modifier`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_drm_format_modifier.html) 則可進行對特定影象格式所支援的 DRM format modifier 進行查詢，以便在匯入 DMA-BUF FD 時使用正確格式。

### [OpenGL](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#opengl)

OpenGL 是平臺無關的，但 OpenGL 上下文的建立是平臺相關的，這就需要中間曾的介入以建立上下文、視窗、實現影象儲存後端，對於 X11 有 GLX，對於 Wayland 有 EGL，對於 Windows 有 WGL。

其中 EGL 類似 Vulkan 以擴充套件的方式提供 X11、Wayland 等平臺的支援。

而 EGL 提供了 [`EGL_EXT_image_dma_buf_import`](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_image_dma_buf_import.txt) 和 [`EGL_EXT_image_dma_buf_export`](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_image_dma_buf_export.txt) 擴充套件以供驅動實現 DMA-BUF FD 匯入匯出支援。

## [DMA-BUF 的利用情況](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#dma-buf-%E7%9A%84%E5%88%A9%E7%94%A8%E6%83%85%E6%B3%81)

### [桌面環境](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E6%A1%8C%E9%9D%A2%E7%92%B0%E5%A2%83)

在 KDE、GNOME 等 Wayland 桌面實現中，DMA-BUF 被用於螢幕/視窗共享，其中最通行的即是桌面環境實現 XDG Desktop Portal，而使用者程式呼叫其中的螢幕共享介面 [`org.freedesktop.portal.ScreenCast`](https://flatpak.github.io/xdg-desktop-portal/#gdbus-org.freedesktop.portal.ScreenCast) 以獲取螢幕/視窗資料的 DMA-BUF FD。

### [PipeWire](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#pipewire)

PipeWire 是影片和音訊資料的交換中心，其中的影片資料就支援依賴 DRM format modifier 的 DMA-BUF 共享機制。

XDG Desktop Portal 的螢幕共享介面和攝像頭介面其實就是以 PipeWire 為後端的。

### [VA-API](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#va-api)

VA-API 是硬體影片編解碼加速 API，它可選的支援使用 DMA-BUF 作為編碼輸入。故而對於螢幕錄製可以先將螢幕匯出為 DMA-BUF FD 再匯入編碼以減少傳輸損失。

### [GStreamer](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#gstreamer)

GStreamer 支援 DMA-BUF 作為影片儲存格式（`video/x-raw(memory:DMABuf)`），故只要外掛支援 DMABuf 就可以使用 DMA-BUF。比如使用 `pipewiresrc` 和 `glimagesink` 實現 EGL/OpenGL 零複製顯示 PipeWire 中支援 DMA-BUF 的影片源，又或是配合 VA-API 外掛進行高效能錄製編碼。

## [總結](https://eh5.me/zh-tw/blog/dmabuf-in-instance/#%E7%B8%BD%E7%B5%90)

DMA-BUF 已然是使用者空間 Linux 影象共享的通行券，現在的、可見未來的影象共享應用都應該且需要使用它。