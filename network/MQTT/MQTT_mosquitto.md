## libmosquitto C API 精選整理

> 以下內容根據官方手冊與 mosquitto.h／man 頁整理。參考源包括 “libmosquitto — MQTT version 5.0/3.1.1 client library”. [Eclipse Mosquitto](https://mosquitto.org/man/libmosquitto-3.html?utm_source=chatgpt.com)

### 1. 版本與初始/清理函式

``` c
int mosquitto_lib_version(int *major, int *minor, int *revision);
int mosquitto_lib_init(void);
int mosquitto_lib_cleanup(void);
```

- `mosquitto_lib_version(...)`：回傳庫版本（如 major/minor/revision）以及整數表示。
- `mosquitto_lib_init()`：在使用任何 client 實例前必呼叫。
- `mosquitto_lib_cleanup()`：結束後釋放資源。 [# mosquitto.h](https://mosquitto.org/api/files/mosquitto-h.html)
- **提醒**：在你嵌入式場景中，保證此呼叫僅一次、且避免重覆初始化，比較安全。

### 2. 建構與釋放 Client 實例

``` c
struct mosquitto *mosquitto_new(const char *id, bool clean_session, void *userdata);
int mosquitto_reinitialise(struct mosquitto *mosq, const char *id, bool clean_session, void *userdata);
void mosquitto_destroy(struct mosquitto *mosq);
```

- `mosquitto_new(...)`：建立一個 mosquitto client 實例。參數：
    - `id`：client ID（字串）
    - `clean_session`：是否為乾淨會話
    - `userdata`：你可傳入任意 void* 作為上下文指標，在 callback 中回傳。 [Stack Overflow](https://stackoverflow.com/questions/75006187/how-to-use-a-username-and-password-for-mosquitto-new-c-c-mqtt?utm_source=chatgpt.com)
- `mosquitto_reinitialise(...)`：重新設定已存在 mosq 實例（少用）。
- `mosquitto_destroy(...)`：銷毀 client 實例，釋放資源。
- **提醒**：你若在多串流／多 thread 環境，需確保 destroy 前完成 loop ＆ disconnect 流程，以避免 race 或資源洩漏。

### 3. 認證、用戶名/密碼、TLS／安全設定

``` c
int mosquitto_username_pw_set(struct mosquitto *mosq, const char *username, const char *password);
int mosquitto_tls_set(struct mosquitto *mosq, const char *cafile, const char *capath, const char *certfile, const char *keyfile, int (*pw_callback)(char *buf, int size, int rwflag, void *userdata));
int mosquitto_tls_opts_set(struct mosquitto *mosq, int cert_reqs, const char *tls_version, const char *ciphers);
int mosquitto_tls_insecure_set(struct mosquitto *mosq, bool value);
```

- `mosquitto_username_pw_set(...)`：設定 MQTT broker 連線時的用戶名與密碼。[how to use a username and password for mosquitto_new c/c++ mqtt](https://stackoverflow.com/questions/75006187/how-to-use-a-username-and-password-for-mosquitto-new-c-c-mqtt)
- `mosquitto_tls_set(...)`：設定 TLS 所需：CA 憑證、路徑、用戶端憑證與金鑰、以及密碼回呼函式。
- `mosquitto_tls_opts_set(...)`：設定 TLS 選項，例如憑證驗證需求、TLS 版本、加密套件。
- `mosquitto_tls_insecure_set(...)`：設定是否允許不檢查伺服器的 TLS 憑證（測試用，不建議生產使用）。
- **提醒**：在你重視零拷貝／高效能／嵌入式網路的場景中，TLS 設定若不慎可能成為 latency 瓶頸。建議預先在 boot 階段建立憑證／上下文，而不要於主資料流程進行動態設定。

### 4. 連線、斷線、回呼設定

``` c
int mosquitto_connect(struct mosquitto *mosq, const char *host, int port, int keepalive);
int mosquitto_connect_async(struct mosquitto *mosq, const char *host, int port, int keepalive);
int mosquitto_reconnect(struct mosquitto *mosq);
int mosquitto_reconnect_async(struct mosquitto *mosq);
int mosquitto_disconnect(struct mosquitto *mosq);
int mosquitto_disconnect_async(struct mosquitto *mosq);
```

- `mosquitto_connect(...)`：同步方式與 broker 建立連線。
- `mosquitto_connect_async(...)`：非同步方式。
- `mosquitto_reconnect(...)`／`*_async`：重新連線。
- `mosquitto_disconnect(...)`／`_async`：斷線。
- **提醒**：對於你多串流／GPU／DMA 系統，建議使用非同步版本（_async）以避免阻塞主 thread 或干擾影像處理流程。

### 5. 訂閱與發佈

``` c
int mosquitto_subscribe(struct mosquitto *mosq, int *mid, const char *sub, int qos);
int mosquitto_unsubscribe(struct mosquitto *mosq, int *mid, const char *sub);
int mosquitto_publish(struct mosquitto *mosq, int *mid, const char *topic, int payloadlen, const void *payload, int qos, bool retain);
```

- `mosquitto_subscribe(...)`：訂閱主題 `sub`，可取得 mid (msg ID)。
- `mosquitto_unsubscribe(...)`：取消訂閱。
- `mosquitto_publish(...)`：發佈資料至 topic。可指定 payloadlen、payload（可為 binary）、qos、retain。
- **提醒**：在 零拷貝／高效能系統中，若 payload 為大型資料（例如影像摘要或異常訊號），注意是否需要封包分段或避免同步發佈造成 frame 率下降。

### 6. 事件處理迴圈／背景執行

``` c
int mosquitto_loop(struct mosquitto *mosq, int timeout, int max_packets);
int mosquitto_loop_forever(struct mosquitto *mosq, int timeout, int max_packets);
int mosquitto_loop_start(struct mosquitto *mosq);
int mosquitto_loop_stop(struct mosquitto *mosq, bool force);
```

- `mosquitto_loop(...)`：非阻塞式事件迴圈呼叫，timeout 毫秒，最多處理 max_packets 封包。
- `mosquitto_loop_forever(...)`：阻塞式，直到連線中斷或錯誤。
- `mosquitto_loop_start(...)`：啟動背景執行緒處理迴圈。
- `mosquitto_loop_stop(...)`：停止背景 thread，可選是否強制 force。 [Eclipse Foundation](https://www.eclipse.org/lists/mosquitto-dev/msg01032.html?utm_source=chatgpt.com)
- **提醒**：你在嵌入式多任務環境中可能傾向使用 loop_start 搭配 callback，或者在你自訂的 thread pool 中呼叫 loop 以整合網路事件與影像流程。注意 thread 安全、避免 100% CPU 忙等（有使用者回報 loop 造成高 CPU 負載）。 [Stack Overflow](https://stackoverflow.com/questions/tagged/libmosquitto?utm_source=chatgpt.com)

### 7. 錯誤碼定義

- 錯誤與返回值以 `int` 型別表示。
- 常見錯誤碼：
    - `MOSQ_ERR_SUCCESS` = 0：成功。
    - `MOSQ_ERR_INVAL`：參數不合法。
    - `MOSQ_ERR_NO_CONN`：尚未連線。
    - `MOSQ_ERR_PROTOCOL`：協議錯誤。 [docs.rs](https://docs.rs/crate/libmosquitto-sys/latest/source/src/lib.rs?utm_source=chatgpt.com)
- **提醒**：在你高可靠／監控系統中，必須檢查每一個呼叫的返回値、登錄錯誤碼、並根據 error 做適當重連或錯誤處理。