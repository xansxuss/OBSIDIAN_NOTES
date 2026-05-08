### Understanding Memory Management on Hardware-Coherent Platforms

這篇由 NVIDIA 技術部落格文章，探討在「硬體相干記憶體平台」（hardware-coherent platforms）上，兩種主要記憶體管理模式：NUMA 模式與 Coherent Driver‑based Memory Management (CDMM) 模式。以下是重點：

1. 什麼是 NUMA 模式？
   - 在 NUMA 模式下，CPU（主機）記憶體與 GPU（裝置）記憶體都會被暴露給作業系統 (OS) 管理。 
   - 意即 OS 可以用 malloc、mmap、CUDA API 等方式同時分配 CPU 與 GPU 記憶體。 
   - 但這也帶來副作用：GPU 記憶體被視作一般記憶體池，可能被 OS 用來做檔案快取、或系統記憶體不足時「spill」到 GPU，對某些應用來說不理想。

2. 什麼是「硬體相干平台」（hardware-coherent platforms）？
   - 指像 GH200、GB200、GB300 這類平台，CPU 與 GPU 透過 NVLink C2C (chip-to-chip) 連線，可實現記憶體的硬體級相干 (即 CPU/GPU 可直接存取彼此記憶體)。 
   - 這雖好，但對一些假設 OS 無法／不該存取 GPU 記憶體的應用與叢集平台（例如 Kubernetes）會造成意外行為。 

3. CDMM 模式（Coherent Driver-based Memory Management）
   - CDMM 模式下，GPU 記憶體 不再 被暴露給 OS 作為一個可被管理的 NUMA 節點；而是由 NVIDIA 的驅動程式來直接管理 GPU 記憶體。 
   - 在 CDMM 模式中：
     - 系統分配記憶體（system-allocated memory）不會被遷移到 GPU。雖然 GPU 仍可透過 C2C 連線訪問這些記憶體，但頁面不會自動移過去。 
     - 工具如 numactl、mbind 對 GPU 記憶體不再有效，它們只能用於系統記憶體。 
   - 為什麼要用？對於需要更嚴格控制 GPU 記憶體、不想被 OS「動手腳」的應用／叢集環境（例如 Kubernetes 管理 GPU 的情境）特別適合。

4. NUMA vs CDMM：何時用哪種？

| 模式   | 適用場景                                  | 特性摘要                                                     |
| ---- | ------------------------------------- | -------------------------------------------------------- |
| NUMA | 傳統依賴 OS 管理記憶體的應用                      | CPU＋GPU 記憶體被合併為大池，系統可做動態遷移。 ([NVIDIA Developer][1])      |
| CDMM | 想對 GPU 記憶體做精細控制／使用 Kubernetes 等容器叢集環境 | GPU 記憶體隔離、OS 不可動、GPU 記憶體用量視覺化較佳。 ([NVIDIA Developer][1]) |

5. 為什麼在 Kubernetes 環境要特別注意？
   - 在 NUMA 模式下，Kubernetes 可能會：
     - 錯誤把 GPU 記憶體算進節點的系統記憶體（memory over-reporting），導致 Pod 要的記憶體比實際可用的還多。 
     - Pod 的記憶體上限 (memory limits) 可能會同時作用在系統＋GPU 記憶體，違背原本想只限系統記憶體的意圖。 
     - 記憶體隔離失效：容器可能存取它原本不應該存取的 GPU 記憶體區塊。 
   - 所以在硬體相干平台＋Kubernetes 場景下，建議使用 CDMM 模式。 

6. 小結
   - 如果你的平台是硬體相干（如 GH200／GB200／GB300），而且你關心 GPU 記憶體管理、容器化、需強隔離／最佳效能，那 CDMM 是值得啟用的模式。
   - 若你的是傳統平台、或應用依賴 OS 管理整個記憶體池，那麼 NUMA 模式可能繼續也沒問題。
   - 關鍵就是：**誰來管 GPU 記憶體？OS 還是驅動？**這決定了記憶體遷移、可視化、隔離、效能的差異。


**如果你的機器是那種 CPU 跟 GPU 有夠貼（透過 NVLink 那種硬體直通），意思就是 GPU 記憶體其實跟系統那塊記憶體有點像「同桌吃飯」的關係。這種「硬體相干平台」其實帶來好處但也有坑。**

在一般的 NUMA 模式下：作業系統看到 GPU 記憶體也當成池子裡一塊，系統、程式都可能把 GPU 記憶體當成「普通記憶體」來用，比如快取資料、搬頁面什麼的。但你如果是做 AI／大規模叢集／GPU 很重要的應用，你可能不想系統這樣亂用／亂乖乖。
所以 NVIDIA 提出 CDMM 模式：讓 GPU 記憶體從 OS 的視野中「抽離」出來，專由 NVIDIA 驅動管理。你就能比較清楚控制：好，這塊 GPU 記憶體就是給 GPU 用，不用怕 OS 偷用、記憶體被亂分配。
實作上：在 CDMM 模式，系統分配的記憶體雖然 GPU 還看得到，但不會被自動「搬」進 GPU。也就是說，搬記憶體給 GPU 的那條路被關了一條。
再來，一些常見工具（numactl、mbind）原本如果你想針對 NUMA 節點做記憶體綁定，在 CDMM 裡面對 GPU 那塊其實沒作用了。
如果你是在 Kubernetes 叢集裡面做 GPU 分配：若還用 NUMA 模式，可能會看到奇怪的狀況──Pod 要記憶體看到比實際少、記憶體上限被 GPU＋系統合算、記憶體隔離怪怪的。這時候切成 CDMM，就比較「乾淨」。
簡單一句話：
**「你要 GPU 記憶體當成獨立、你自己管，那就 CDMM；你要系統跟 GPU記憶體當成一大池子，那就 NUMA。」**

7. reference
[Understanding Memory Management on Hardware-Coherent Platforms](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)