一、為什麼建議硬啃這條路
學習型專案的價值就在於搞懂底層原理,GStreamer 那條路雖然省事,但它會把 output plane/capture plane、DMA buffer、EGL interop 這整套邏輯都包在黑盒子裡,你學到的只是「怎麼串 pipeline 字串」,不是「Jetson 硬解到底怎麼運作」。既然你的目標是學習,直接碰底層 API 才能真正搞懂:

V4L2 M2M 的 buffer queue 機制(這是很多嵌入式影像系統的共通概念,不只 Jetson 用)
DMA buffer 跟一般記憶體的差異、為什麼要 EGL 這一層才能讓 CUDA 讀到
已經寫過 NPP 色彩轉換那段,這條路可以讓你把「硬解」跟「GPU 後處理」串成完整一條龍,體會完整的 zero-copy pipeline 概念
二、循序漸進的建議走法,不要一次到位
直接從 EGL interop 開始寫容易卡死很久,除錯也難,建議分階段:

第一階段:先讓 NvVideoDecoder 能吐畫面出來就好
不管 CUDA,先用 dqBuffer 拿到 capture plane 的資料後,直接 memcpy 到 CPU、丟給 OpenCV imshow 看得到畫面即可。這階段的重點是搞懂 output/capture plane 的 queue/dequeue 節奏、V4L2 event 訂閱(V4L2_EVENT_RESOLUTION_CHANGE 這類）,這部分不弄懂,後面接 CUDA 只會更混亂。

第二階段:接上 EGL/CUDA interop
確認畫面正確後,再把 memcpy 那段換成 NvBufSurfaceGetEGLImage + cudaGraphicsEGLRegisterImage,這時候你已經很清楚 buffer 的生命週期,比較不會在 interop 這層又同時除錯兩種問題。

第三階段:接你原本的 NPP 後處理
把 CUeglFrame 拿到的指標接回你原本 nppiNV12ToBGR_8u_P2C3R_Ctx 那段邏輯,整條 pipeline 就打通了。