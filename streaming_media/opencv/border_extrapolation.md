### 邊界延伸（border extrapolation）」型別定義，用於像是濾波、卷積、影像金字塔等操作時，當 kernel 超出影像邊界該怎麼處理的策略。

Enum：cv::BorderTypes

| Enum 名稱              |          數值           | 視覺範例（假設影像內容是 `abcdefgh`） | 延伸邏輯                                       | 常見用途               |                                                 |                              |
| :--------------------- | :---------------------: | :------------------------------------ | :--------------------------------------------- | :--------------------- | ----------------------------------------------- | ---------------------------- |
| **BORDER_CONSTANT**    |           `0`           | `iiiiii                               | abcdefgh                                       | iiiiiii`               | 超出邊界的值以「指定常數 `i`」填滿              | 手動填充背景色、zero-padding |
| **BORDER_REPLICATE**   |           `1`           | `aaaaaa                               | abcdefgh                                       | hhhhhhh`               | 邊界外的值重複最近的像素                        | 模糊、濾波時常用             |
| **BORDER_REFLECT**     |           `2`           | `fedcba                               | abcdefgh                                       | hgfedcb`               | 超出邊界時鏡射，但不重複邊界像素                | 平滑過渡、不產生明顯邊緣     |
| **BORDER_WRAP**        |           `3`           | `cdefgh                               | abcdefgh                                       | abcdefg`               | 從另一邊循環取樣                                | 週期性影像、傅立葉環狀處理   |
| **BORDER_REFLECT_101** |           `4`           | `gfedcb                               | abcdefgh                                       | gfedcba`               | 鏡射但重複邊界外一層像素（注意比 REFLECT 不同） | 預設選項（BORDER_DEFAULT）   |
| **BORDER_TRANSPARENT** |           `5`           | `uvwxyz                               | abcdefgh                                       | ijklmno`               | 超出區域的像素不影響結果（當作透明）            | 特殊 blending / ROI 運算     |
| **BORDER_REFLECT101**  | 同 `BORDER_REFLECT_101` | —                                     | 同上                                           | 同上                   |                                                 |                              |
| **BORDER_DEFAULT**     | 同 `BORDER_REFLECT_101` | —                                     | 同上                                           | 預設值                 |                                                 |                              |
| **BORDER_ISOLATED**    |          `16`           | —                                     | 限制內插僅在 ROI 內，不會去取鄰近 ROI 外部像素 | ROI 區域運算時避免越界 |                                                 |                              |

對比重點：REFLECT vs REFLECT_101

| 類型            | 範例輸出（假設影像 `a b c d e f g h`） |                 |                |
| --------------- | -------------------------------------- | --------------- | -------------- |
| **REFLECT**     | `f e d c b a                           | a b c d e f g h | h g f e d c b` |
| **REFLECT_101** | `g f e d c b                           | a b c d e f g h | g f e d c b a` |
