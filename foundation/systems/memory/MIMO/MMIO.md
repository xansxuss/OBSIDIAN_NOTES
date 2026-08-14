MMIO（記憶體映射輸入輸出）暫存器是**將硬體裝置的控制與資料暫存器對應到主記憶體位址空間**的一種設計，包含**記憶體共用**、**一般讀寫指令**、**裝置控制**等核心特徵。這讓 CPU 能用存取一般記憶體的方式來操作周邊設備。 [1](https://www.cnblogs.com/suv789/p/18536526), [2](https://translate.google.com/translate?u=https://swiftpackageindex.com/apple/swift-mmio/0.1.1/documentation/mmio/understanding-mmio&hl=zh&sl=en&tl=zh&client=sge)

運作原理

- **位址對應**：硬體裝置（如計時器、序列埠、GPIO）被分配到 CPU 記憶體空間裡的特定數值位址。

- **一般指令**：CPU 讀寫這些裝置時，不必使用特別的指令，直接用平常讀寫 RAM 的載入（load）和儲存（store）指令即可。

- **硬體回應**：裝置會一直看著位址匯流排，只要發現 CPU 訪問了分配給自己的位址就會立刻做出回應。 [1](https://zh.wikipedia.org/zh-tw/%E5%AD%98%E5%82%A8%E5%99%A8%E6%98%A0%E5%B0%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA) [2](https://translate.google.com/translate?u=https://swiftpackageindex.com/apple/swift-mmio/0.1.1/documentation/mmio/understanding-mmio&hl=zh&sl=en&tl=zh&client=sge)  [3](https://www.cnblogs.com/suv789/p/18536526)

特點與注意事項

- **記憶體屬性**：因為對應到外部裝置，編譯器或 CPU 不能隨便亂序執行或把它當成普通 RAM 快取，通常需要設定為揮發性（volatile）或不可快取。

- **相比 PMIO**：與使用獨立埠和專用指令的端口映射輸入輸出（Port-Mapped I/O）不同，MMIO 直接共用記憶體空間。[1](https://ithelp.ithome.com.tw/m/articles/10364246), [2](https://www.cnblogs.com/suv789/p/18536526), [3](https://translate.google.com/translate?u=https://swiftpackageindex.com/apple/swift-mmio/0.1.1/documentation/mmio/understanding-mmio&hl=zh&sl=en&tl=zh&client=sge)