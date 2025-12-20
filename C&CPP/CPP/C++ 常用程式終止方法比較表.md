#### C++ 常用程式終止方法比較表

| 方法 | 所屬標頭 | 行為	| 是否呼叫局部/全域析構函式	| 返回碼 / 狀態碼 | 適合場合 | 備註 |
|----|----|----|----|----|----|----|
| return (main)  | — | 結束 main 函式 → 程式終止 | 會呼叫所有局部及全域/靜態物件析構函式 | return 值即程式退出碼 | 正常結束程式 | 最安全、最常用 |
| std::exit(status) | <cstdlib> | 立即終止程式 | 只呼叫全域/靜態物件析構函式 + atexit() 註冊函式，局部物件不呼叫 | EXIT_SUCCESS / EXIT_FAILURE 或自訂整數 | 異常情況下需要快速退出，但想保留全域/靜態清理	 | 不會執行 try/catch；不會返回 main |
std::_Exit(status) / _exit(status)	<cstdlib> / <unistd.h>	立即終止程式	完全不呼叫任何析構函式	自訂整數	當程式在 forked child 或非常危險狀態下直接退出	更低層次，跳過一切清理
| abort() | <cstdlib> | 異常終止程式，會產生 core dump（通常 134） | 不呼叫任何析構函式或 atexit() | 平台依賴（通常 134） | 偵測到不可恢復錯誤、程式崩潰 | 適合做 assert 失敗時使用 |
| throw（未捕捉異常） | — | 如果異常沒被捕捉 → 呼叫 std::terminate() → 預設行為為 abort() | 會呼叫正在 unwind 的局部物件析構函式 | 平台依賴 | 異常處理機制 | 捕捉異常後可以恢復程式流程 |

1. 析構函式呼叫範圍
    - return main → 全部析構
    - std::exit → 全域/靜態析構 + atexit()，局部的不會
    - abort/_Exit → 一概不呼叫
2. 程式返回碼
    - 0 或 EXIT_SUCCESS → 表示成功
    - 非 0 或 EXIT_FAILURE → 表示異常
    - shell 可以透過 $? 或 echo $? 讀取
3. 適用場景簡單判斷
    - 正常結束 → return main
    - 非預期錯誤，但想做部分清理 → std::exit(EXIT_FAILURE)
    - 危險狀態或 fork child → _Exit() / std::_Exit()
    - 程式崩潰或 assert → abort()