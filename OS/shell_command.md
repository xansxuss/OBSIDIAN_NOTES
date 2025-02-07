搜尋包含特定字串的檔案
若已知目標檔案的特定類型或名稱模式，可使用 find 的其他選項來縮小搜尋範圍，從而提高搜尋效率。例如，僅搜尋以 .txt 結尾的檔案：

``` find / -type f -name "*.txt" -print0 | xargs -0 grep -l "example_string" ```

pgrep：比如，你可以使用 pgrep -u root 來代替 ps -ef | egrep 『^root 『 | awk 『{print $2}』，以便抓取屬於 root 的 PID。

pstree：我覺得這個命令很酷，它可以直接列出進程樹，或者換句話說是按照樹狀結構來列出進程。

bc：這個命令在我的系統中沒有找到，可能需要安裝。這是用來執行計算的一個命令，如使用它來開平方根。

split：這是一個很有用的命令，它可以將一個大文件分割成幾個小的部分。比如：split -b 2m largefile LF_ 會將 largefile 分割成帶有 LF 文件名前綴且大小為 2 MB 的小文件。

nl：能夠顯示行號的命令。在閱讀腳本或代碼時，這個命令應該非常有用。如：nl wireless.h | head。 mkfifo：作者說這是他最喜歡的命令。該命令使得其他命令能夠通過一個命名的管道進行通信。嗯，聽起來有點空洞。舉例說明，先創建一個管道並寫入內容： mkfifo ive-been-piped ls -al split/* | head > ive-been-piped

然後就可以讀取了：head ive-been-piped。

ldd：其作用是輸出指定文件依賴的動態鏈接庫。比如，通過 ldd /usr/java/jre1.5.0_11/bin/java 可以瞭解哪些線程庫鏈接到了 java 依賴（動態鏈接）了哪些庫。（感謝 NetSnail 的指正。）

col：可以將 man 手冊頁保存為無格式的文本文件。如： PAGER=cat man less | col -b > less.txt

xmlwf：能夠檢測 XML 文檔是否良好。比如： curl -s 『http://bashcurescancer.com' > bcc.html xmlwf bcc.html perl -i -pe 『s@ @ @g' bcc.html xmlwf bcc.html bcc.html:104:2: mismatched tag

lsof：列出打開的文件。如：通過 lsof | grep TCP 可以找到打開的端口。