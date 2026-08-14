**ABA 問題**是==多執行緒與並行程式設計（Concurrent Programming）中，使用無鎖結構（Lock-free）與 **CAS（Compare-And-Swap）** 機制時常見的隱藏錯誤==。當執行緒讀取變數值為 `A`、被暫停，此時其他執行緒將值改為 `B` 又改回 `A`，原執行緒恢復後會誤以為數值「從未改變」而操作成功，導致邏輯出錯。 [[1](https://zh.wikipedia.org/zh-tw/%E6%AF%94%E8%BE%83%E5%B9%B6%E4%BA%A4%E6%8D%A2), [2](https://zhuanlan.zhihu.com/p/110301415), [3](https://www.zhihu.com/question/23281499)]

詳細解析與常見應對方式整理如下：

發生情境與過程

- 執行緒 1 從記憶體讀取變數值為 `A`。

- 執行緒 1 暫停（或換到其他時間片）。

- 執行緒 2 將該記憶體值從 `A` 改為 `B`，接著又改回 `A`。

- 執行緒 1 恢復執行，進行 CAS 檢查，發現值仍是 `A`。

- 執行緒 1 判定「數值沒有改變」並繼續執行，但背後的資料結構可能已被其他執行緒替換或刪除。 [[1](https://translate.google.com/translate?u=https://en.wikipedia.org/wiki/ABA_problem&hl=zh&sl=en&tl=zh&client=sge), [2](https://jovanaeducation.com/zh-hant/glossary/aba-problem), [3](https://zh.wikipedia.org/zh-tw/%E6%AF%94%E8%BE%83%E5%B9%B6%E4%BA%A4%E6%8D%A2)]

常見解決方法

- **版本號或時間戳（Stamp / Version）**：在比較數值時，除了比對資料本身，也同步比對修改次數或版本號（例如 Java 的 `AtomicStampedReference`），只要數值被改過，版本號就會增加，使 CAS 檢查失敗。 [[1](https://translate.google.com/translate?u=https://lumian2015.github.io/lockFreeProgramming/aba-problem.html&hl=zh&sl=en&tl=zh&client=sge), [2](https://translate.google.com/translate?u=https://www.baeldung.com/cs/aba-concurrency&hl=zh&sl=en&tl=zh&client=sge)]

- **安全記憶體回收與物件標識**：利用行程垃圾回收機制（GC）、**Hazard Pointers**（危險指標）或延遲釋放，避免記憶體位址被其他新物件重複快速重用。 [[1](https://translate.google.com/translate?u=https://www.baeldung.com/cs/aba-concurrency&hl=zh&sl=en&tl=zh&client=sge), [2](https://jovanaeducation.com/zh-hant/glossary/aba-problem)]

- **使用傳統互斥鎖（Lock）**：如果效能允許，改用有鎖的同步機制（如 Mutex / Synchronized）能直接從根本阻斷並行交錯修改的問題。 [[1](https://zhuanlan.zhihu.com/p/110301415)]