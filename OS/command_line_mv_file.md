你要「指定範圍移動」這批檔案（例如從 frame_228516.jpg 到 frame_228533.jpg）到另一個資料夾，
假設：

檔案都在當前目錄

你想移到 /mnt/storage/output

可以直接用 bash 一行完成：

```bash
mv C01F_AI004_2025-10-15_10_10_03_802_frame_{228516..228533}.jpg /mnt/storage/output/
```

💡解釋：

{228516..228533} 是 bash 的序列展開語法，會展開成 228516～228533。

mv 會依序移動所有符合的檔案。

若想「複製」而非移動，改成 cp。

📦 如果要批次移多段範圍：

```bash
mv C01F_AI004_2025-10-15_10_10_03_802_frame_{228400..228450,228516..228533}.jpg /mnt/storage/output/
```

這個錯誤👇

``` bash
bash: /usr/bin/mv: Argument list too long
```

代表展開後的檔案太多（超過 Linux ARG_MAX 限制，大約幾萬～十萬個字元）。

✅ 解法 1：用 find + xargs

這是最穩定的方式（處理成千上萬檔案也沒問題）：

``` bash
find . -maxdepth 1 -type f -name 'C01F_AI004_2025-10-15_10_10_03_802_frame_*.jpg' \
  | awk -F'[_.]' '{n=$(NF-1); if(n>=000000 && n<=250000) print $0}' \
  | xargs -I{} mv {} ../C01F_AI004_2025-10-15_10_10_03_802_2/
```

🔍說明：
find 列出目前資料夾下的所有符合 .jpg 的檔案。
awk 解析檔名中的 frame 編號，篩出在 [114266, 228533] 範圍內的。
xargs 批次移動，不會超出 ARG_MAX 限制。

``` bash
| awk -v RS='\0' -F'[_.]' '{n=$(NF-1); if(n>=114266 && n<=228533) printf "%s\0",$0}'
```

這段其實是 為了從 find -print0 接收 null-delimited 檔名，解析其中的 frame 編號，並只輸出在指定範圍內的檔名。

🔍 逐段解析
1️⃣ -v RS='\0'
- AWK 預設用「換行」當 record separator。
- 改成用 NULL (\0) 當 record 結束符號。
- 這是為了跟 find -print0 搭配，避免檔名有空白、tab、奇怪字元炸掉。
👉 每個 record = 一個完整的檔案路徑

2️⃣ -F'[_.]'
設定欄位分隔符（field separator）為：
- _
- .
表示每遇到 _ 或 . 就切成一個欄位。
例如：
` C01F_AI004_2025-10-15_10_10_03_802_frame_114300.jpg `
會被分成：

|NF | 欄位內容|
| --- | --- |
| 1 | C01F |
| 2 | AI004 |
| 3 | 2025-10-15 |
| 4 | 10 |
| 5 | 10 |
| 6 | 03 |
| 7 | 802 |
| 8 | frame |
| 9 | 114300 |
| 10 | jpg |

3️⃣ {n=$(NF-1); ... }
NF = 欄位總數
$(NF-1) = 倒數第二個欄位 → 就是 frame number
在上例中：
- NF = 10
- NF-1 = 9
- $(NF-1) = 第 9 欄 = 114300
👉 n = 那個 frame 數字
4️⃣ if(n>=114266 && n<=228533)
這是 frame 範圍過濾。
只輸出介於 114266 到 228533 的檔案。
5️⃣ printf "%s\0",$0
- $0 = 完整原始 line（也就是檔案路徑）
- 用 \0 結尾，保持 null-delimited 形式
- 讓後面的 xargs -0 正確接收
👉 這是 piping 的「零損失安全輸出」。

✅ 解法 2：用迴圈（安全但慢）

如果你想明確看到每個搬動過程，可以用：

``` bash
for f in C01F_AI004_2025-10-15_10_10_03_802_frame_*.jpg; do
  num=$(echo "$f" | grep -oP '(?<=frame_)\d+(?=\.jpg)')
  if (( num >= 114266 && num <= 228533 )); then
    mv "$f" ../C01F_AI004_2025-10-15_10_10_03_802_2/
  fi
done
```

這段 bash 會逐一解析檔名中的 frame 編號，並只搬符合範圍的。

✅ 解法 3：用 parallel（最快）

如果你有 GNU parallel：

``` bash
ls C01F_AI004_2025-10-15_10_10_03_802_frame_*.jpg | \
awk -F'[_.]' '{n=$(NF-1); if(n>=114266 && n<=228533) print $0}' | \
parallel mv {} ../C01F_AI004_2025-10-15_10_10_03_802_2/
```

``` bash
find . -maxdepth 1 -type f -name 'C01F_AI004_2025-10-15_10_10_03_802_frame_*.json' \
  | awk -F'[_.]' '{n=$(NF-1); if(n>=114266 && n<=228533) print $0}' \
  | xargs rm -f
```

📝 指令說明（快速 recap）

find：找所有符合 frame pattern 的 .json

awk：從檔名中抓 frame number（倒數第二段）

rm -f：刪掉符合範圍的檔案

完全沿用你的處理邏輯，只是把 mv 換成 rm -f。

🧪 如果你想先看會刪哪些，請先跑這個：

``` bash
find . -maxdepth 1 -type f -name 'C01F_AI004_2025-10-15_10_10_03_802_frame_*.json' \
  | awk -F'[_.]' '{n=$(NF-1); if(n>=114266 && n<=228533) print $0}'
```

確認沒問題再刪。

計算數量
find . -maxdepth 1 -type f -name 'C01F_AI004_2025-10-15_10_10_03_802_frame_*.json' \
  | awk -F'[_.]' '{n=$(NF-1); if(n>=114266 && n<=228533) print $0}' \
  | wc -l
