GDB（GNU Debugger）是Linux/Unix下非常強大且廣泛使用的程式偵錯工具，幾乎是C/C++開發的必備利器。要用好GDB，光知道它能停在斷點不夠，得真正理解它怎麼觀察和操控程式流程，還能動態檢查變數、記憶體、呼叫堆疊，甚至修改執行路徑。

GDB 基本教學：從入門到實務
1. 編譯程式時要加上 -g
GDB需要符號資訊才能對應原始碼。
bash
```
gcc -g -o myprogram myprogram.c
```

或者C++：
bash
```
g++ -g -o myprogram myprogram.cpp
```
這樣編譯出的執行檔才能在GDB看到原始碼、變數名稱。

2. 啟動 GDB
bash
```
gdb ./myprogram
```

或直接：

bash
```
gdb --args ./myprogram arg1 arg2
```
後者允許你直接帶程式參數。

3. 常用指令
| 指令               | 功能說明                                |
| ---------------- | ----------------------------------- |
| `run` (r)        | 開始執行程式                              |
| `break` (b)      | 設定斷點 (ex: `b main`、`b myfile.c:25`) |
| `next` (n)       | 執行下一行（不進入函式內部）                      |
| `step` (s)       | 執行下一行（會進入函式內部）                      |
| `continue` (c)   | 繼續執行直到下一斷點或程式結束                     |
| `print` (p)      | 印出變數值 (`p var`、`p mystruct.field`)  |
| `info locals`    | 列出當前函式的區域變數                         |
| `backtrace` (bt) | 顯示呼叫堆疊                              |
| `list` (l)       | 顯示程式碼，預設從當前行附近顯示10行                 |
| `quit` (q)       | 退出GDB                               |


4. 實務流程示範
假設程式裡有bug，我們想從main函式開始偵錯。

bash
```
(gdb) break main
Breakpoint 1 at 0x4005d6: file myprogram.c, line 10.
(gdb) run
Starting program: ./myprogram
Breakpoint 1, main () at myprogram.c:10
10      int a = 5;
(gdb) next
11      int b = 0;
(gdb) print a
$1 = 5
(gdb) step
```
用step可以進入函式，next則跳過函式內部。
用print能查詢任何變數或表達式的值。

5. 進階技巧
條件斷點

bash
```
(gdb) break myfile.c:50 if x > 10
```
當 x > 10 才會在第50行停下。

修改變數

bash
```
(gdb) set var x = 100
```
直接改變變數的值，方便測試。

觀察點 (watchpoint)
監控變數的值改變：

bash
```
(gdb) watch x
```
一旦x變了，程式就會暫停。

反覆執行直到特定條件
用until執行直到某行：

bash
```
(gdb) until 40
```
6. 多執行緒程式調試
bash
```
(gdb) info threads
(gdb) thread 2
(gdb) bt
```
切換不同執行緒，查看呼叫堆疊。

7. 其他有用指令
info breakpoints：列出所有斷點。

disable / enable：停用/啟用斷點。

disassemble：反組譯目前函式，查看機器碼。

set pagination off：關閉分頁，讓輸出不會卡住。

8. GDB Script自動化
將指令寫在檔案，執行時用：

bash
```
gdb -x myscript.gdb ./myprogram
```
方便重複調試。

小結
GDB就像給你一把程式世界的「顯微鏡」和「遙控器」，能精準定位問題、動態觀察變數狀態、甚至即時改變程式狀態。使用起來雖然有學習曲線，但掌握關鍵指令後效率驚人。

你可以告訴我你用什麼語言、想debug的程式狀況，我可以幫你定製一套更細節的GDB操作流程。或者你有什麼想挑戰的GDB操作，都可以丟過來，咱們一起攻克！








