一、Vim 的核心世界觀：模式(Mode)

Vim 的靈魂只有一句話：

你大部分時間不在打字，而是在「操作文字」

因此它有模式切換，而不是滑鼠。

最重要的三個模式

Normal mode：預設狀態，發號施令(移動、刪除、複製)

Insert mode：真的在輸入字元

Visual mode：選取文字(像拖曳，但用鍵盤)

Esc 是宇宙的 reset 鍵。迷路就按。

二、移動：比方向鍵快一個維度
基本移動(Normal mode)

``` markdown
h  ←
j  ↓
k  ↑
l  →
```


方向鍵是給新手用的，Vim 直接把手留在主鍵盤。

進階移動(效率暴擊)

``` markdown
w   下一個單字開頭
b   上一個單字
e   單字結尾

0   行首
^   第一個非空白
$   行尾
```

大尺度跳躍(空間摺疊)

``` markdown
gg  檔案開頭
G   檔案結尾
```

三、編輯的數學公式(Vim 的精髓)

Vim 指令其實長這樣：

``` markdown
[次數] + 動作 + 範圍
```

動作(Action)

``` markdown
d  delete
y  yank (copy)
c  change = delete + insert
```

範圍(Motion)

``` markdwon
w   一個單字
$   到行尾
j   下一行
```

組合起來就很科學

``` markdown
dw    刪一個單字
d$    刪到行尾
2dw   刪兩個單字
c$    改到行尾(進入 insert)
```

這不是快捷鍵，是語言。

四、插入模式：只在必要時才打字

進入 Insert mode 的正確姿勢：

``` markdown
i   在游標前插入
a   在游標後插入
o   下一行新增一行
O   上一行新增一行
```

打完字，Esc 回 Normal。這個節奏很重要。

五、Visual 模式：精準選取，不靠滑鼠

``` markdown
v   字元選取
V   整行選取
Ctrl + v   區塊選取(矩形)
```

選好後直接：

``` markdown
d   刪除
y   複製
```

工程師會愛上 Ctrl+v，改 table / 多行對齊時很像在寫 CUDA kernel。

六、複製、貼上、反悔人生

``` markdown
yy    複製整行
p     貼在後面
P     貼在前面

u     undo
Ctrl+r redo
```

Vim 的 undo 是「樹狀歷史」，不是線性的，這點很反直覺也很強。

七、搜尋與取代(編輯器瞬間升級)

``` markdown
/word     向下搜尋
?word     向上搜尋
n         下一個
N         上一個
```

取代(工程師常用)

``` markdown
:%s/old/new/g
```

翻譯：

``` markdown
% 全檔案

s substitute

g 全部取代
````

八、檔案操作(最後還是得存檔)

``` markdown
:w      存檔
:q      離開
:wq     存檔離開
:q!     強制離開(放棄人生)
```

九、給理工腦的記憶方式

把 Vim 想成：

- Normal mode：控制平面(control plane)

- Insert mode：資料平面(data plane)

- 指令 = operator + operand

- 移動不是副作用，是主要操作

十、最小生存指令集(先背這 12 個)

``` markdown
Esc  i  a  o
h j k l
w b
d y p
u
```