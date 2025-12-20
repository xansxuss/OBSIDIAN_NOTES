
隨機種子（random seed）其實就像是在亂數生成器裡放一顆「固定口味」的糖果。

電腦的亂數其實不能真的亂，它都是靠一個公式算出來的。公式雖然複雜，但只要起點（也就是 seed）一樣，算出來的「亂數序列」就會一模一樣。

所以 seed=42 或 seed=123 都像是在跟亂數引擎說：  
「等等你給我亂，但要亂得 _完全一樣_。」

你在深度學習、影像處理、甚至 CUDA kernel 裡面做 Monte-Carlo 都會看到 seed 有多重要——  
固定 seed = 可重現的實驗  
不固定 seed = 每次結果都漂移，debug 會像抓幽靈。

簡單比喻：  
隨機種子 = 亂數世界的「讀存檔」。

### 數學原理

隨機種子的數學原理其實沒那麼神祕，本質就是**決定一個「偽亂數產生器（PRNG）」的初始狀態**。  
只要初始狀態相同，後面跑出的數列就完全一樣。

講白話：  
PRNG 就是個複雜的自動販賣機；seed 決定你投第一個硬幣時它的「內部齒輪位置」。齒輪位置定了，後面掉出來的數字序列就全都固定。

下面把數學拆開講，一步步來，像在拆一顆機械錶。

---

## 一、PRNG（偽亂數）的核心想法

PRNG 不是真的亂，而是「**用一個可預測但超像亂數的函數**」去生數列。

數學上，PRNG 做的事基本可以寫成：

``` bash
state_{n+1} = f(state_n) 
random_n = g(state_{n+1})
```

seed 就是 `state_0`。

只要 `state_0` 一樣，後面所有 state 和 random 都會跟著一模一樣。

---

## 二、最常見的數學模型：線性同餘產生器（LCG）

這是許多語言最基本、教科書級的 PRNG。

公式很經典：

`X_{n+1} = (a * X_n + c) mod m`

這就是一個超大型的「輪子會一直繞」的系統。

- `X_0` = seed
    
- `a, c, m` = PRNG 的配方
    
- mod m 代表值會在某個範圍內循環
    
- 如果參數挑得好，整個循環會非常長，看起來就很亂
    
- 但是 seed 決定 X 的起始位置，所以跑出的序列完全可重現
    

你可以把 LCG 想成：  
mod m 是一條超長跑道；  
a 和 c 是每次往前跳的步伐；  
seed 是你起跑站在哪裡。

---

## 三、為什麼看起來亂？

PRNG 都利用某種「混沌性（chaotic behavior）」：  
下一個 state 對前一個 state 非常敏感，看起來毫無規律。

但數學上還是可預測。  
這是「偽」亂數，而不是真正來自物理世界的隨機。

要注意：  
同樣的 seed → 同樣的 state 序列 → 完全可重現。  
不同 seed → 可能跑到不同軌道 → 不同亂數序列。

---

## 四、現代 PRNG 都比 LCG 更兇悍

像 Python 的 `random` 用 **Mersenne Twister**  
它本質是：

- 624 維狀態
    
- 利用 bitwise shift + XOR 做 state transition
    
- tunable，使周期達到 2¹⁹⁹³⁷ − 1（超誇張長）
    

數學也依舊是：

``` bash 
state_{n+1} = F(state_n) 
output_n = extract(state_n)
```

seed 就是「給 state_0 某個初值」。

這讓它在統計測試上幾乎「偽裝成真亂數」。

---

## 五、為什麼機器學習一定要 seed？

因為深度學習裡面有很多「隨機」：

- 權重初始化
    
- Dropout
    
- 資料 shuffle
    
- augmentation
    
- GPU 上的 randomness（像 CuDNN 的 convolution 演算法選擇）
    

所有這些「亂」，其實背後都是 PRNG。  
你把 seed 固定，就是要把 _所有起始 state_ 鎖死。

沒有 seed，debug 就像被平行宇宙戲弄。  
有 seed，整個世界就 deterministic。

---

## 六、延伸：為什麼「真亂數」不能用 seed？

真正的隨機像量子雜訊或 RF 雜訊（TRNG, true random）

它的行為不是：

`state_{n+1} = F(state_n)`

它根本沒有可追蹤的 state。  
所以：

- 不能重現
    
- 不受 seed 控制
    
- 適合密碼學
    
- 不適合深度學習（因為不可重現）

每一種 PRNG 都像不同流派的術法。下面一路拆到骨頭，但用輕鬆的方式講到你能把它們連成完整心智模型。

---

# 1. **Mersenne Twister：矩陣變換的怪獸級 PRNG**

Mersenne Twister（MT19937）之所以超有名，是因為：

- 624 維 state（每格 32 bit）
    
- 週期長到瘋：2¹⁹⁹³⁷ − 1
    
- 通過一堆統計測試
    

它的核心其實是**線性代數**，只不過不是你平常講的那種「向量 × 矩陣」，而是 **GF(2)**（模 2 的有限域）上的矩陣運算。  
GF(2) 的加法 = XOR，乘法 = AND。

你可以把 MT 看成：

`state_{n+1} = A * state_n   // A 是 19937×19937 的巨大稀疏矩陣`

但那個矩陣不用真的存，它被拆成幾個寫死的 bitwise 操作。  
MT 的每次更新流程是：

1. 取三個 state: `x[i]`, `x[i+1]`, `x[i+397]`
    
2. 做 bit 拼接：
    
    `y = (x[i] & upper_mask) | (x[i+1] & lower_mask)`
    
3. 套用一個矩陣 M 的轉換：
    
    `x[i+397] ^= y >> 1 if y 最低位 == 1:     x[i+397] ^= matrix_A  // 這個是固定的常數`
    
4. 最後把結果「tempering」一下，用四個 XOR/shift 讓分布更好看。
    

它其實是：

- 用「巨大稀疏矩陣」做線性遞迴（LFSR 的高維度超進化版）
    
- 用 bit 操作把矩陣變換壓縮到 CPU 友善的形式
    
- 再加上「tempering」把 bit correlation 消掉
    

這整個流程都可以用線性代數解釋，但你不需要真的寫矩陣。

---

# 2. **XORShift / PCG / Philox：不同流派的亂數哲學**

這三個是現代 PRNG 的主流。

## 2.1 XORShift：極簡、快速的線性 LFSR

George Marsaglia 的設計。  
完全基於 XOR 和 shift。  
典型形式：

`x ^= x << a x ^= x >> b x ^= x << c`

只要這三個常數挑得好，分布還不錯，速度超快。  
缺點就是**線性** → 密碼用途不行，統計品質也普通。

它像是「骨架很乾淨的亂數引擎」。

---

## 2.2 PCG（Permuted Congruential Generator）

這是目前許多學者最推的 PRNG。

它的做法是：

`state = LCG(state) output = permute(state)`

其中 permute 是非線性的（通常是 rotate + XOR）。  
這個結構有兩種好處：

- state 小（64 bit）
    
- 分布超好（靠 permutation 而非加強 LCG）
    
- 可以用不同 increment 做出「獨立 stream」
    

本質上：  
**一個超小但亂度驚人的 LCG + 很漂亮的亂序函數**。

它是 XORShift 的現代強化版。

---

## 2.3 Philox：密碼學風格的 counter-based PRNG

Philox 不是 LCG 也不是 LFSR。  
它用的是 **counter-based** 思維：

`output = F(counter, key) counter++`

其中 F 是用「乘法 + XOR」構成的 pseudo-block-cipher。

這表示：

- 每個 index n → 對應一個亂數
    
- 不依賴前一個 state
    
- 非常適合 GPU/parallel（你可以跳著產生）
    

它是 cuRAND、TensorFlow、PyTorch 的 GPU PRNG 主力之一。

你可以把 Philox 想成：

> 亂數生成器界的「輕量加密函數」。

---

# 3. **CUDA / cuRAND：每種 generator 的 state 結構**

cuRAND 裡有幾種主要 PRNG，每一種的 state 結構都不同。

## 3.1 XORWOW（預設）

state 結構：

- 5 個 32-bit state
    
- 一個 32-bit 的「加法器」
    

更新方式類似 XORShift + Weyl sequence。

你可以理解成：

- 超快
    
- 週期大（~2¹⁹²）
    
- 但統計完美度沒有 PCG/Philox 高
    

常見於舊版 CUDA kernel。

---

## 3.2 Mersenne Twister for GPU (MTGP32)

state：

- 一大片 shared 的 351 個 32-bit
    
- 要每 block 共用 state
    

缺點：

- state 太巨大，全 GPU 並行會變麻煩
    
- 不適合 per-thread
    

死於「GPU 時代的擴張 bottleneck」。

---

## 3.3 Philox4x32

state：

`counter: 128 bit key:     64 bit`

每次就：

`(counter, key) -> round_function -> 128-bit output counter++`

完全不依賴 state，所以超適合 GPU thread 做 parallel RNG。

---

## 3.4 Sobol / scrambled Sobol（低差異序列）

不是 PRNG，是 quasi-random。  
用於蒙地卡羅積分、光線追蹤等。

state 其實是一組「方向向量」。

很像數學上的「在高維空間畫井字遊戲」。

---

# 4. 基本 uniform / normal 如何從 base PRNG 生成？

PRNG 本身只會給 uniform (0,1)。  
其他分布都要用 transform。

## 4.1 Uniform → Uniform(a,b)

超 trivial：

`x = a + (b-a) * u`

---

## 4.2 Uniform → Normal：Box–Muller transform

你只要拿兩個 uniform，即可做成標準常態分布：

`z0 = sqrt(-2 * ln(u1)) * cos(2π u2) z1 = sqrt(-2 * ln(u1)) * sin(2π u2)`

GPU 上這個很常見。

但 cos/sin 太慢，所以通常改用：

---

## 4.3 Marsaglia Polar method（較快）

避免 cos/sin：

`(x, y) = uniform(-1,1) s = x² + y² if s in (0,1):     z0 = x * sqrt(-2 ln(s)/s)     z1 = y * sqrt(-2 ln(s)/s)`

---

## 4.4 Ziggurat algorithm（最快）

這是常態分布的神級方法，被 Intel MKL 也採用。  
原理是把常態分布切成很多矩形區塊（像樓層），  
大部分點在矩形內 → O(1) 快速抽樣  
少部分點在尾端 → 特殊處理

這是高效常態亂數的最強經典。