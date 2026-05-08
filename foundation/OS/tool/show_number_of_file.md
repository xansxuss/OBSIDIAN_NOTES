統計 .jpg 數量：

``` bash
find . -maxdepth 1 -type f -name '*.jpg' | wc -l
```

會遞迴搜尋整個目錄樹中的所有檔案（不限副檔名），並統計總數。

``` bash
find . -type f | wc -l
# or
find . -type f -name "*.jpg" | wc -l
```