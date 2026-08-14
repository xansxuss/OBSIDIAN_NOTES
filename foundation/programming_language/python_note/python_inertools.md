

### 1. cycle

有時會遇到需要環狀走訪一個 list 的情況，譬如以下情況

A -> B -> C -> D -> A -> B -> C …

這時候能夠利用 itertools 中的 [cycle](https://docs.python.org/3/library/itertools.html#itertools.cycle) ，例如以下範例走訪 A, B, C, D ，到 D 之後又從 A 開始走訪起，走訪 2 次後結束執行：

``` python
from itertools import cycle   
count = 1 
for x in cycle(['A', 'B', 'C', 'D']):     
	print(count, x)     
		if count == 8:         
			break     
		count += 1
```
PS 值得注意的是 cycle 並沒有設置結束條件的選項，所以得自己控制何時結束才行，否則會進入無窮迴圈。

### 2. ncycles

如果要設置環狀走訪幾次，除了像前述範例自行控制之外，可以利用 more-itertools 的 [ncycles](https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles) 。

例如以下範例，只會走訪 A, B, C, D 2 次，不像 cycle 需要自行控制何時結束，相較於 cycle 更加簡潔：
```python
from more_itertools import ncycles

for x in ncycles(['A', 'B', 'C', 'D'], 2):
    print(x)
```

### 3. filterfalse

[filterfalse](https://docs.python.org/3/library/itertools.html#itertools.filterfalse) 很適合用來找出 iterable 中不符合條件的元素。
```python
from itertools import filterfalse


input = [
   {'user_id': 1, 'habits': ['fishing', 'hiking']},
   {'user_id': 2, 'habits': []},
   {'user_id': 3, 'habits': ['drawing']},
   {'user_id': 4, 'habits': ['swimming']},
]
for x in filterfalse(lambda d: d['habits'], input):
    print(x)
```

第 10 行的 `lambda d: d['habits']` 被稱為 **predicate** ， 該函式只會回傳 True/False 2 種情況，回傳 false 時，就會被 `filterfalse` 捕捉，進而將結果 yield ，因此該 predicate 判斷情況為 False 時，就是沒有 habits 的使用者。

上述範例用以找出沒有 `habits` 的使用者，其執行結果如下。

{'user_id': 2, 'habits': []},

### 4. groupby

Itertools 還提供 [groupby](https://docs.python.org/3/library/itertools.html#itertools.groupby) 讓我們能夠為 iterable 進行分組，雖然其函式名稱 `groupby` 會讓人直覺認為與 SQL 的 groupby 一樣方便，但其實並非如此，要能夠順利使用 itertools 的 `groupby` ，得先將 iterable 按照分組依據排序過才行。例如以下資料 `input` 已經先按照 `group` 排序過一次，才能夠順利執行：
```python
from itertools import groupby


input = [
    {'id': 1, 'group': 'A'},
    {'id': 4, 'group': 'A'},
    {'id': 5, 'group': 'B'},
    {'id': 2, 'group': 'B'},
    {'id': 3, 'group': 'C'},
    {'id': 6, 'group': 'C'},
]
for group, members in groupby(input, lambda x: x['group']):
    print(group, list(members))
```

### 5. product

[product](https://docs.python.org/3/library/itertools.html#itertools.product) 的完美應用場景是處理像 9 * 9 乘法表，需要多層迴圈的情況，譬如以下是 9 * 9 乘法表最直覺的實作方式 － 使用雙層迴圈：

```python
for x in range(1, 10):
    for y in range(1, 10):
        print(x, y, x*y)
```

然而用 [product](https://docs.python.org/3/library/itertools.html#itertools.product) 就只需要 1 個迴圈即可：

```python
from itertools import product


for x, y in product(range(1, 10), range(1, 10)):
    print(x, y, x*y)
```

product 也可以處理多層迴圈，因為 product 可以接受不定長度的 iterable ，例如 `product(itertable1, iterable2, iterable3, ...)` ，所以我們同樣可以將 2 * 3 * 4 乘法表濃縮為 1 個迴圈：

```python
from itertools import product


for x, y, z in product(range(1, 3), range(1, 4), range(1, 5)):
    print(x, y, z, x*y*z)
```

### 5. flatten
處理 2 維陣列時，有一種情況也經常會遇到 － 走訪 2 維陣列中的所有元素。

例如以下 2 維陣列，如果我們想要走訪所有元素，直覺上應該也是使用 2 層迴圈：
```python
input = [
   [1, 2, 3],
   [4, 5, 6],
   [7, 8, 9],
]
```

不過實際上還可以透過 [flatten](https://myapollo.com.tw/blog/python-itertools-more-itertools/flatten) 將 2 維陣列中的值全部扁平化到 1 維陣列後進行走訪，程式看起來也相對簡潔些：

```python
from more_itertools import flatten

input = [
   [1, 2, 3],
   [4, 5, 6],
   [7, 8, 9],
]

for x in flatten(input):
    print(x)
```

### 6. islice

使用 Python 的人都知道 slice 的用法，譬如 `x = [1, 2, 3, 4]` ，我們可以用 `x[1:3]` 取得 `[2, 3]` 2 個元素，這種用法就被稱為切片(slice) ，而 [islice](https://docs.python.org/3/library/itertools.html#itertools.islice) 與 slice 用法相似，差別在於 islice 會回傳 1 個 generator ，而不是直接回傳切片後的結果。

以下是 islice 的範例，該範例利用 islice 試圖取得 `input[2:4]` 的結果：
```python
from itertools import islice


input = [1, 2, 3, 4, 5, 6]

generator = islice(input, 2, 4)
for i in generator:
    print(i)
```

### 7.  grouper

接著談談很高機率會遇到的批次(batch)處理，譬如將 list 中的資料每 500 個為一組進行處理，這種需求可以透過 [grouper](https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.grouper) 達成，例如以下範例以 2 個為一組取得資料：
```python
from more_itertools import grouper


for group in grouper([1, 2, 3, 4, 5, 6, 7], 2):
    print(group)
```

如果想改變預設值 None ，可以多加個參數，例如以下：
```python
from more_itertools import grouper


for group in grouper([1, 2, 3, 4, 5, 6, 7], 2, fillvalue=-1):
    print(group)
```

### 8. ichunked

[ichunked](https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ichunked) 與 grouper 作用類似，但是 ichunked 並不會將長度補滿，而且其回傳的值是 islice 的 generator ，如果不需要將長度補滿的情況，可以選擇用 ichunked ：
```python
from more_itertools import ichunked for chunk in ichunked([1, 2, 3, 4, 5, 6, 7], 2): print(chunk, list(chunk))
```
