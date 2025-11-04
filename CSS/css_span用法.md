CSS `<span>`用法表格，幫你快速掌握各種實務用法與常見屬性 👇

| 用法類型                | 範例                                                                                                                       | 說明                                                | 範例效果（文字描述）         |
| :------------------ | :----------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------ | :----------------- |
| **基本結構**            | `<span>文字</span>`                                                                                                        | `<span>` 是行內元素（inline element），不會自動換行，通常用於局部樣式設定。 | 無樣式，純文字。           |
| **設定文字顏色**          | `<span style="color: red;">警告</span>`                                                                                    | 改變文字顏色。                                           | 顯示紅色文字「警告」。        |
| **設定字型大小**          | `<span style="font-size: 20px;">大字</span>`                                                                               | 改變字體大小。                                           | 顯示較大的文字「大字」。       |
| **設定字型樣式**          | `<span style="font-style: italic;">斜體</span>`                                                                            | 改變字型樣式。                                           | 顯示斜體文字「斜體」。        |
| **設定字重（粗體）**        | `<span style="font-weight: bold;">強調</span>`                                                                             | 改變字重。                                             | 顯示粗體「強調」。          |
| **設定背景顏色**          | `<span style="background-color: yellow;">標示</span>`                                                                      | 加上背景底色。                                           | 顯示黃色底的「標示」。        |
| **多屬性組合**           | `<span style="color: blue; font-weight: bold; text-decoration: underline;">連結風格</span>`                                  | 一次設定多個樣式屬性。                                       | 藍色、粗體、有底線的「連結風格」。  |
| **CSS 類別套用**        | `<span class="highlight">文字</span>`<br>`<style>.highlight { color: orange; background: black; }</style>`                 | 透過 class 管理樣式（推薦用法）。                              | 黑底橘字的「文字」。         |
| **ID 指定樣式**         | `<span id="note">備註</span>`<br>`<style>#note { color: gray; font-size: 12px; }</style>`                                  | 適用於唯一元素。                                          | 灰色小字「備註」。          |
| **滑鼠懸停效果**          | `<style>span:hover { color: red; }</style>`<br>`<span>滑過我</span>`                                                        | 使用偽類控制互動樣式。                                       | 滑鼠移上去時變紅。          |
| **內距與邊框**           | `<span style="padding: 3px; border: 1px solid #ccc;">框線文字</span>`                                                        | 雖為行內元素，但可加內距與邊框。                                  | 顯示有細灰框的「框線文字」。     |
| **改為區塊顯示**          | `<span style="display: block;">變成區塊</span>`                                                                              | 讓 span 變成 block 元素。                               | 獨立一行顯示。            |
| **inline-block 排版** | `<span style="display: inline-block; width: 100px;">A</span><span style="display: inline-block; width: 100px;">B</span>` | 可並排顯示但有固定寬度。                                      | A、B 各佔 100px 並排顯示。 |
| **動畫或過渡效果**         | `<style>span { transition: color 0.3s; } span:hover { color: red; }</style>`                                             | 加入 CSS transition 動畫效果。                           | 滑過時顏色會平滑變化。        |
