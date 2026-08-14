在 Linux 中使用 find 指令要排除特定路徑（資料夾），最標準且有效率的方法是使用 -prune 參數。

-prune 的邏輯是：「如果遇到這個路徑，就剪枝（不要走進去）」。

以下為您整理幾種最常見的寫法與情境：

1. 排除單一特定路徑
假設您要在目前目錄（.）下尋找所有的 .py 檔案，但想排除 ./vendor 這個資料夾：

```
Bash
find . -path "./vendor" -prune -o -name "*.py" -print
```


💡 關鍵邏輯拆解：

-path "./vendor" -prune：如果路徑符合 ./vendor，就直接跳過不搜尋。

-o：代表 OR（或）。

-name "*.py" -print：如果不是上述路徑，才尋找檔名符合 .py 的檔案並印出來。

⚠️ 重要細節：使用 -prune 時，後面一定要加上 -o 以及最後的 -print（或 -print0），否則 find 的預設行為會把被排除的路徑本身也印出來。

2. 排除多個指定路徑
如果您想同時排除多個資料夾（例如 vendor 和 .git），可以用 -o 把排除條件括起來。
注意：括號 () 在 Shell 中有特殊意義，所以前面必須加反斜線 \ 進行跳脫。

```Bash
find . -type d \( -path "./vendor" -o -path "./.git" \) -prune -o -name "*.py" -print
```

3. 只針對「資料夾名稱」排除（不看完整路徑）
如果您想排除所有叫做 build 或 node_modules 的資料夾，不論它們在哪個層級，可以使用 -name 搭配 -prune：

```Bash
find . -type d \( -name "build" -o -name "node_modules" \) -prune -o -name "*.cpp" -print
```


### 附加規則
### 1. 限制搜尋深度 (`-maxdepth` / `-mindepth`)

- **`-maxdepth <層級>`**：限制向下搜尋的最大深度。`-maxdepth 1` 代表只搜尋當前目錄，不進入任何子目錄。
    
- **`-mindepth <層級>`**：限制開始搜尋的最小深度。例如 `-mindepth 2` 代表跳過當前目錄下的檔案，從第一層子目錄內部才開始找。
    

> ⚠️ **重要語法規則**：在較新版本的 Linux 中，`-maxdepth` 和 `-mindepth` 必須放在**最前面**（也就是路徑之後、其他條件之前），否則 `find` 會發出警告。

**混合範例：** 在當前目錄下尋找最大深度 3 層內的 `.cpp` 檔案，排除 `./vendor` 資料夾：

Bash

```
find . -maxdepth 3 -path "./vendor" -prune -o -name "*.cpp" -print
```

### 2. 篩選特定權限 (`-perm`)

您可以透過 `-perm` 參數，利用「八進位數字」（如 `755`, `644`）或「符號」（如 `u+x`）來篩選檔案權限。常見的用法有以下幾種：

- **精準匹配**：`-perm 644`（權限剛好是 `644` 的檔案）
    
- **至少包含**：`-perm -200`（代表擁有者「至少」要有寫入權限，不管其他權限為何）
    
- **任一符合**：`-perm /111`（擁有者、群組或其他人中，「任一個」有執行權限即可。舊版 Linux 寫法為 `+111`）
    

**混合範例：** 尋找目前目錄下，擁有者具有「執行權限」（`u+x`）的 Python 檔案，且排除 `.git` 資料夾：

Bash

```
find . -path "./.git" -prune -o -type f -perm -u+x -name "*.py" -print
```

### 3. 篩選特定擁有者或群組 (`-user` / `-group`)

如果您要找的是屬於特定 User 或 Group 的檔案，可以使用 `-user` 和 `-group`：

Bash

```
# 尋找屬於使用者 "root" 且排除 "logs" 目錄的所有檔案
find . -path "./logs" -prune -o -user root -print
```