`git submodule` 是一個讓你在 Git 儲存庫（Repository）中嵌入另一個 Git 儲存庫的機制。這對於管理大型專案中的外部函式庫或模組非常有用，能確保專案引用的是特定版本的程式碼。

以下是針對 `git submodule` 的核心概念與常用指令：

---

### 1. 基本操作

- **新增子模組：**
    
    將外部專案加入目前的儲存庫中。
    
    Bash
    
    ``` bash
    git submodule add <repository_url> <path/to/folder>
    ```
    
    這會建立一個 `.gitmodules` 檔案，記錄子模組的 URL 與路徑。
    
- **複製含有子模組的專案：**
    
    單純的 `git clone` 不會自動下載子模組的內容。你需要執行：
    
    Bash
    
    ``` bash
    git clone --recursive <repository_url>
    ```
    
    或是克隆後手動初始化：
    
    Bash
    
    ``` bash
    git submodule update --init --recursive
    ```
    

---

### 2. 更新與維護

- **從遠端更新子模組：**
    
    如果子模組的遠端有新提交，你可以進入該目錄執行 `git pull`，或者在主專案目錄執行：
    
    Bash
    
    ``` bash
    git submodule update --remote
    ```
    
- **修改子模組：**
    
    子模組本質上是一個獨立的 Git 儲存庫。若要在其中修改程式碼，必須先切換到某個分支（通常子模組處於「斷頭」（Detached HEAD）狀態），修改後進行 `commit` 與 `push`，最後回到主專案提交該子模組指針（Pointer）的更新。
    

---

### 3. 注意事項

- **指針機制：** 主專案並不儲存子模組的程式碼，而是儲存一個 **Commit ID**。當你更新子模組後，主專案會偵測到變更，你需要執行 `git add` 與 `git commit` 來紀錄新的版本指針。
    
- **刪除子模組：** 步驟較為繁瑣，通常建議使用：
    
    Bash
    
    ``` bash
    git rm <path/to/submodule>
    ```
    
    並手動清理 `.git/modules` 資料夾。
    

---

### 4. 與 `git subtree` 的差異

雖然兩者都能達成模組化，但邏輯不同：

|**特性**|**Git Submodule**|**Git Subtree**|
|---|---|---|
|**程式碼儲存**|僅儲存 Commit ID（指針）|直接將程式碼併入主專案|
|**依賴性**|強烈依賴遠端儲存庫|較為獨立|
|**操作難度**|較高，需處理指針更新|較低，操作類似普通分支|

---

### 常用指令表

|**指令**|**說明**|
|---|---|
|`git submodule status`|查看子模組目前的 Commit ID 與狀態。|
|`git submodule init`|初始化 `.gitmodules` 中的設定檔。|
|`git submodule update`|根據主專案紀錄的指針更新子模組內容。|
|`git submodule foreach <command>`|對所有子模組批次執行指令（例如 `git checkout master`）。|
