memcpy 的實作，對以下幾個原理的理解：

- 記憶體對齊（memory alignment）
- 記憶體存取粒度與效率的關係
- 記憶體重疊問題（memory overlap）

### 一、基本實作

``` cpp
#include <stdio.h>

void *memcpy(void *dst, void const *src, size_t size)
{
    assert((dst != NULL) && (src != NULL));
    unsigned char *pdst = (char*)dst;
    unsigned char const *psrc = src;

    while (size--)
    {
        *pdst++ = *psrc++;
    }
    return dst;
}

```

這是最基本的 memcpy 實作。
assert 斷言的加入，能讓面試官看到你有「邊界條件檢查」的意識。
雖然許多標準函式庫的官方實作要求呼叫端自己確保不傳入 NULL 指標，但這樣寫至少顯示出安全意識。

### 二、進一步完善 —— 處理記憶體重疊問題

下面是一個常見的「錯誤示範」，你會在許多文章裡看到類似的寫法：

``` cpp
#include <stdio.h>

void *memcpy(void *dst, const void *src, size_t size)
{
    assert((dst != NULL) && (src != NULL));
    unsigned char *pdst = dst;
    const unsigned char *psrc = src;

    if (psrc < pdst)
    {
        psrc = psrc + size - 1;
        pdst = pdst + size - 1;
        while (size--) // 從後往前複製
        {
            *pdst-- = *psrc--;
        }

    }
    else
    {
        while (size--) // 從前往後複製
        {
            *pdst++ = *psrc++;
        }
    }

    return dst;
}
```

這段程式碼顯示出作者有意識到「記憶體重疊」的情況，並試圖解決。
但它有個潛在錯誤點：psrc < pdst 的比較。

根據 C 標準（參考 CLC-Wiki《the Standard》6.5.9），
只有在兩個指標都指向「同一個陣列」的情況下，才允許做 <、<=、>、>= 等關係運算。
如果兩個指標指向不相關的記憶體區域，這種比較的結果是未定義行為（undefined behavior）。

不過實際上，大部分平台上這樣寫仍「看似能正常運作」，
因為不論 psrc < pdst 的結果如何，只要能避免破壞資料，就能達到預期效果。
因此，這種寫法雖不嚴謹，但在一般應用中可能還能接受。
但如果你要開發像 libc 這類標準函式庫，就不能這樣寫。
這也是為什麼官方的 memcpy 實作不處理重疊區域的原因。

### 三、允許重疊的版本：memmove

memmove 是允許重疊的版本。
但它並不是透過判斷是否重疊來處理，而是使用「暫存區」的方式：

``` cpp
void *memmove(void *dst, const void *src, size_t size)
{
    unsigned char temp[size];
    memcpy(temp, src, size);
    memcpy(dst, temp, size);
    return dst;
}
```

### 四、再進一步完善 —— 存取效率與記憶體對齊

面試官如果更進一步，可能會考你記憶體存取效率的最佳化。
例如 Stack Overflow 上這個問題：
[Implementing own memcpy (size in bytes?)](https://stackoverflow.com/questions/11876361/implementing-own-memcpy-size-in-bytes)

我查閱過 glibc-2.28 中的 memcpy，那實作相當複雜，但明顯考慮了存取效率與記憶體對齊。

``` CPP
void * memcpy (void *dstpp, const void *srcpp, size_t len)
{
  unsigned long int dstp = (long int) dstpp;
  unsigned long int srcp = (long int) srcpp;

  /* 從前往後複製 */

  if (len >= OP_T_THRES)
    {
      /* 先複製幾個位元組讓 DSTP 對齊 */
      len -= (-dstp) % OPSIZ;
      BYTE_COPY_FWD (dstp, srcp, (-dstp) % OPSIZ);

      PAGE_COPY_FWD_MAYBE (dstp, srcp, len, len);
      WORD_COPY_FWD (dstp, srcp, len, len);

      /* 最後複製尾端 */
    }

  /* 剩下的部分使用位元組複製 */
  BYTE_COPY_FWD (dstp, srcp, len);

  return dstpp;
}
```

再看另一個版本

``` cpp
00018 void *memcpy(void *dst, const void *src, size_t len)
00019 {
00020         size_t i;
00021 
00022         /*
00023          * memcpy does not support overlapping buffers, so always do it
00024          * forwards. (Don&#39;t change this without adjusting memmove.)
00025          *
00026          * For speedy copying, optimize the common case where both pointers
00027          * and the length are word-aligned, and copy word-at-a-time instead
00028          * of byte-at-a-time. Otherwise, copy by bytes.
00029          *
00030          * The alignment logic below should be portable. We rely on
00031          * the compiler to be reasonably intelligent about optimizing
00032          * the divides and modulos out. Fortunately, it is.
00033          */
00034 
00035         if ((uintptr_t)dst % sizeof(long) == 0 &&
00036             (uintptr_t)src % sizeof(long) == 0 &&
00037             len % sizeof(long) == 0) {
00038 
00039                 long *d = dst;
00040                 const long *s = src;
00041 
00042                 for (i=0; i < len / sizeof(long); i++) {
00043                         d[i] = s[i];
00044                 }
00045         }
00046         else {
00047                 char *d = dst;
00048                 const char *s = src;
00049 
00050                 for (i=0; i < len; i++) {
00051                         d[i] = s[i];
00052                 }
00053         }
00054 
00055         return dst;
00056 }
```

第 35～36 行檢查目標與來源指標是否「以字長（sizeof(long)）」對齊。
第 37 行則檢查長度是否是 sizeof(long) 的整數倍。
若三個條件都成立，就用 long 為單位複製，效能會比逐位元組高得多。
若不滿足，則退回到單位元組的複製。

這裡牽涉的知識點是：

記憶體對齊（alignment）

存取粒度（access granularity）

存取效率的差異

可以參考這篇延伸閱讀：
👉 記憶體對齊相關問題的簡要總結

### 五、再再進一步：混合對齊與餘數處理

如果我們假設系統是 4 位元組對齊（sizeof(unsigned int)），
那就可以把 n（要複製的長度）分成兩部分：

整數倍的 4 位元組（n / 4）

不足 4 位元組的尾巴（n % 4）

當來源與目標都對齊時，用 4 位元組存取；否則就退回到逐位元組。

``` cpp
#include <stdio.h>

// 假設記憶體存取粒度 align = sizeof(unsigned int)

void *mymemcpy(void *dst, void const *src, size_t n)
{
   size_t div = n / sizeof(unsigned int); // 有多少個完整的 align 區塊
   size_t rem = n % sizeof(unsigned int); // 剩下不足 align 的部分

   unsigned char *pdst = dst;
   unsigned char const *psrc = src;

   if ((unsigned int)dst % sizeof(unsigned int) == 0 &&
       (unsigned int)src % sizeof(unsigned int) == 0)
   {
       // 對齊情況，使用 align 粒度複製
       for (size_t i = 0; i < div; ++i)
       {
           *((unsigned int *)pdst) = *((unsigned int*)psrc);
           pdst += sizeof(unsigned int);
           psrc += sizeof(unsigned int);
       }

       // 處理尾巴部分（不足 align）
       for (size_t i = 0; i < rem; ++i)
           *pdst++ = *psrc++;
   }
   else 
   {
       // 未對齊，逐位元組複製
       for (size_t i = 0; i < n; ++i)
       {
           *pdst++ = *psrc++;
       }
   }

   return dst;
}
```
