# 一、字串拷貝 strcpy

`strcpy` 函式的原型是：

```cpp
char* strcpy(char* des , const char* src)
```
des 和 src 所指的記憶體區域 不可以重疊，且 des 必須有足夠的空間來容納 src 的字串。

``` cpp
#include <assert.h>
#include <stdio.h>

char* strcpy(char* des, const char* src)
{
    assert((des != NULL) && (src != NULL)); 
    char *address = des;  
    while((*des++ = *src++) != '\0')  
        ;  
    return address;
}
```

要注意 `strcpy` 會拷貝 `'\0'`，此外還有幾個重點：

- **來源指標所指的字串內容不可修改**，因此應該宣告為 `const` 型別。
- 要檢查來源與目的指標是否為空指標（`NULL`），這裡使用 `assert`來驗證。
- 需使用一個暫存變數保存目的字串的首位址，最後回傳這個首位址。
- 函式返回 `char*` 的原因是為了支援**鏈式呼叫**（chained expression），例如可以將 `strcpy` 作為其他函式的引數。
<p>函数 strlen 的原型是 size_t strlen(const char *s)&#xff0c;其中 size_t 就是 unsigned int。</p>

## 二、字串長度 strlen

`strlen` 函式的原型為：

``` cpp
size_t strlen(const char *s)
```

其中 `size_t` 是 `unsigned int` 的別名。

``` cpp
#include <assert.h>
#include <stdio.h>

int strlen(const char* str)
{
    assert(str != NULL);
    int len = 0;
    while((*str++) != '\0')
        ++len;
    return len;
}
```

### strlen 與 sizeof 的差異：

- `sizeof` 是**運算子**，`strlen` 是**函式**。
- `sizeof` 可以用在**型別**或**變數**上；`strlen` 只能用 `char*` 型態的變數作為引數，且必須以 `'\0'` 結尾。
- `sizeof` 在**編譯期**計算所佔記憶體大小；`strlen` 則在**執行期**運算字串長度。
- 當陣列作為 `sizeof` 的參數時，不會退化為指標；但傳遞給 `strlen` 時會退化成指標。

---

## 三、字串連接 strcat

`strcat` 函式的原型是：

``` cpp
char* strcat(char* des, const char* src)
```

`des` 和 `src` 所指的記憶體區域 **不可以重疊**，且 `des` 必須有足夠的空間來容納 `src` 的字串。

```cpp
#include <assert.h>
#include <stdio.h>

char* strcat(char* des, const char* src)   // const 表示輸入參數 
{  
    assert((des != NULL) && (src != NULL));
    char* address = des;
    while(*des != '\0')  // 移動到字串末尾
        ++des;
    while(*des++ = *src++)
        ;
    return address;
}
```

---
## 四、字串比較 strcmp

`strcmp` 函式的原型為：

```cpp
int strcmp(const char *s1, const char *s2)
```

返回值規則如下：

- 若 `s1 == s2`，返回 **0**
- 若 `s1 > s2`，返回 **正數**
- 若 `s1 < s2`，返回 **負數**

比較方式是：兩個字串自左至右逐字比較（依 **ASCII** 值大小），直到出現不同字元或遇到 `'\0'` 為止。

``` cpp
#include <assert.h>
#include <stdio.h>

int strcmp(const char *s1, const char *s2)
{
    assert((s1 != NULL) && (s2 != NULL));
    while(*s1 == *s2)
    {
        if(*s1 == '\0')
            return 0;
        ++s1;
        ++s2;
    }
    return *s1 - *s2;
}
```

## 附：`assert()` 斷言

`assert` 是**巨集（macro）**，而不是函式。  
其原型定義在標頭檔 `assert.h` 中：

``` cpp
void assert(int expression);
```

`assert` 通常用於函式開頭處檢查輸入參數的合法性，可視為一種簡易的「例外處理機制」。

### `assert` 的行為：

- 若 **表達式為假（false）**，會先將錯誤訊息輸出至 `stderr`，然後呼叫 `abort()` 終止程式執行。
- 若 **表達式為真（true）**，則繼續執行後續程式。

### 注意事項：

`assert` 僅在 **DEBUG 模式** 下生效。  
若要在釋出（release）版本中停用，可以在：

``` cpp
#include <assert.h>
```

之前加上：

``` cpp
#define NDEBUG
```

即可關閉所有 `assert` 檢查。