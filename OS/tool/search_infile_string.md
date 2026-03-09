### 檢索指定路徑下的檔案有無指定字串

用 grep 就好，讓工具替你當顯微鏡。思路是：只掃 header 檔、遞迴、顯示位置，避免在雜訊裡迷路。

假設你要檢索的路徑是 /some/path，字串是 importbuffer_virtualaddr。

最實用版本長這樣：

bash
￼複製程式碼
grep -RIn --include='*.h' 'importbuffer_virtualaddr' /some/path
在做什麼，一行拆解給你：

-R：遞迴掃整個目錄樹

-I：忽略 binary 檔（避免奇怪亂碼）

-n：顯示行號（debug 時超重要）

--include='*.h'：只看 header file

'importbuffer_virtualaddr'：你要找的關鍵字

/some/path：指定路徑

如果你想順便看一下是哪個 header 被 hit，但不想看到內容，只要檔名：

bash
￼複製程式碼
grep -RIl --include='*.h' 'importbuffer_virtualaddr' /some/path
如果你在 cross-compile 或 vendor SDK 裡，header 很多、grep 有點慢，可以再加一層懷疑精神，只掃 include 相關目錄：

bash
￼複製程式碼
grep -RIn --include='*.h' 'importbuffer_virtualaddr' /some/path/include