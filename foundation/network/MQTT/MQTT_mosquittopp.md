---
title: ""
source: ""
author: ""
published: ""
created: 2026-08-27
description: ""
tags:
  - foundation
  - network
  - MQTT
process: true
note-link: "[[LLM_Wiki/wiki/foundation/network/mqtt/mosquittopp.md]]"
status: draft
aliases: []
related: []
---

## 📋 API 整理(精華版)

| 函式 / 方法                                                                                             | 用途                   | 備註                               |
| --------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------- |
| `mosqpp::mosquittopp(const char *id=NULL, bool clean_session=true)`                                 | 建構函式。建立一個 client 實例。 | client ID、clean session 的預設參數。   |
| `~mosquittopp()`                                                                                    | 解構函式。清理／釋放資源。        | 若呼叫 loop_start 等非同步模式，要注意停止。     |
| `int connect(const char *host, int port=1883, int keepalive=60)`                                    | 與 broker 建立連線。       | 回傳值需檢查是否成功連線。                    |
| `int disconnect()`                                                                                  | 與 broker 斷線。         | 在退出程式或切換狀態時使用。                   |
| `int subscribe(const char *sub, int qos=0)`                                                         | 訂閱主題。                | 你可能改用 subscribe_v5 等（若支援 v5 的話）。 |
| `int unsubscribe(const char *sub)`                                                                  | 取消訂閱。                | 同上。                              |
| `int publish(const char *topic, int payloadlen, const void *payload, int qos=0, bool retain=false)` | 發佈訊息到指定主題。           | payload 可為二進位資料。                 |
| `void loop_forever(int timeout=1000, int max_packets=1)`                                            | 阻塞式事件迴圈。             | 適合簡單用例。                          |
| `int loop(int timeout=100, int max_packets=1)`                                                      | 非阻塞／輪詢式事件迴圈。         | 在你多任務/embedded 系統中更有彈性。          |
| `int loop_start()` / `int loop_stop()`                                                              | 啟動／停止背景執行的 loop 執行緒。 | 若你在多串流或多 thread 環境中，可用。          |
| `static void lib_init()` / `static void lib_cleanup()`                                              | 初始化與清理 mosquitto 庫。  | 在 main 前／後呼叫。                    |


## mosquittopp.hpp／mosquittopp.h API 中文整理  
> C++ 封裝 libmosquitto 庫：類別 `mosqpp::mosquittopp`（注意：此封裝已被標示為 **DEPRECATED**） :contentReference[oaicite:2]{index=2}

## 1. 命名空間與類別  
```cpp
namespace mosqpp {
    class mosquittopp { … };
}
```

- 類別：`mosqpp::mosquittopp`：用於建立 MQTT 客戶端。 [Eclipse Mosquitto](https://mosquitto.org/api/files/cpp/mosquittopp-h.html?utm_source=chatgpt.com)
- 注意：由於標示為 DEPRECATED，未來可能不再維護，或者對 MQTT v5 的新特性支持不足。 [GitHub](https://github.com/eclipse/mosquitto/issues/2782?utm_source=chatgpt.com)

## 2. 靜態初始化／清理

```cpp
static int lib_init();
static int lib_cleanup();
```

- `lib_init()`：在使用 mosquittopp 前必呼叫，初始化底層 libmosquitto 庫。
- `lib_cleanup()`：於程式結束前呼叫，釋放 libmosquitto 資源。 
- 備註：如果多執行緒/多模組使用，要注意呼叫順序、安全性。

## 3. 建構與解構

``` cpp
mosquittopp(const char *id = NULL, bool clean_session = true);
~mosquittopp();
```

- 建構函式參數：
    - `id`：用於 MQTT client ID。如果為 NULL，系統可能生成隨機 ID（視底層實作）。
    - `clean_session`：是否為乾淨會話 (clean session)。
- 解構函式：清理 client 物件。若使用 `loop_start()` 等背景 thread，要確保先呼叫 `disconnect()`／`loop_stop()`。
- 注意：依你嵌入式環境，建議將 client 物件 RAII 化，避免資源洩漏。

## 4. 連線與斷線

``` cpp
int connect(const char *host, int port = 1883, int keepalive = 60);
int disconnect();
```

- `connect(...)`：連線到 MQTT broker。
    - `host`：broker 位址。
    - `port`：預設 1883。
    - `keepalive`：心跳秒數。
- `disconnect()`：主動斷線。
- 回傳值「int」表示成功／失敗（底層對應 libmosquitto 錯誤碼）。
- 提醒：在你的高效能系統中，連線／斷線不可阻塞主流程，建議非同步處理或回調機制。

## 5. 訂閱與取消訂閱

``` cpp
int subscribe(int *mid, const char *sub, int qos = 0);
int unsubscribe(int *mid, const char *sub);
```

- `subscribe(...)`：訂閱主題：
    - `mid`：訊息 ID，若非 NULL，可用於追蹤。
    - `sub`：主題字串。
    - `qos`：服務品質 (QoS) 0/1/2。
- `unsubscribe(...)`：取消訂閱。
- 備註：若用 MQTT v5，可能有 _v5 版本函式（但 mosquittopp 可能未實作完整） [GitHub](https://github.com/eclipse/mosquitto/issues/2782?utm_source=chatgpt.com)

## 6. 發佈訊息

``` cpp
int publish(int *mid, const char *topic, int payloadlen,
            const void *payload, int qos = 0, bool retain = false);
```

- `mid`：訊息 ID（可為 NULL 表示不追蹤）。
- `topic`：主題。
- `payloadlen`：負載長度（bytes）。
- `payload`：指向資料的指標。
- `qos`：服務品質。
- `retain`：是否為保留訊息。
- 備註：底層可能為非同步；若 payload 為大量資料，需注意記憶體與拷貝。 [cnblogs.com](https://www.cnblogs.com/embedded-linux/p/9386169.html?utm_source=chatgpt.com)

## 7. 事件迴圈 (loop)

```cpp
int loop(int timeout = 100, int max_packets = 1);
int loop_forever(int timeout = 1000, int max_packets = 1);
int loop_start();
int loop_stop();
```

- `loop(...)`：輪詢式事件迴圈，非阻塞；適合你這種嵌入式／多任務系統。
- `loop_forever(...)`：阻塞式事件迴圈，直到斷線或錯誤。
- `loop_start()`：啟動背景 thread 處理；
- `loop_stop()`：停止背景 thread。

## 8. TLS／安全設定

``` cpp
int tls_set(const char *cafile, const char *capath = NULL,
            const char *certfile = NULL, const char *keyfile = NULL,
            int (*pw_callback)(char *buf, int size, int rwflag, void *userdata) = NULL);
int tls_opts_set(int cert_reqs, const char *tls_version = NULL,
                 const char *ciphers = NULL);
int tls_insecure_set(bool value);
int tls_psk_set(const char *psk, const char *identity, const char *ciphers = NULL);
```

- `tls_set(...)`：設定 CA 憑證、客戶端憑證、金鑰等。 [Stack Overflow+1](https://stackoverflow.com/questions/65134467/c-mqtt-mosquitto-client-with-tls?utm_source=chatgpt.com)
- `tls_opts_set(...)`：設定憑證需求、TLS 版本、加密套件。
- `tls_insecure_set(...)`：是否允許忽略伺服器憑證驗證（**僅測試用**）。
- `tls_psk_set(...)`：設定 PSK (預共享密鑰) 模式。
- 提醒：在資料安全／嵌入式 IoT 設備中，務必以 TLS 1.2／1.3 ＋ CA 驗證模式為優。
## ## 9. 其他設定函式／選項

（以下為部分功能，文件未詳列每一參數）
- `int opts_set(int option, void *value)`：設定選項。
- `bool is_session_present_on_connect()`：在 connect 完成時，檢查 session 是否已存在（適用 MQTT 3.1.1）。
- 回調 函式（需由 子類覆寫）：
``` cpp
void on_connect(int rc) override;
void on_disconnect(int rc) override;
void on_message(const struct mosquitto_message *message) override;
void on_subscribe(int mid, int qos_count, const int *granted_qos) override;
```
… 等

這部分於 C++ 封裝中 virtual 定義。文件並未列出所有回調／參數說明。

## ## 10. 錯誤碼／返回值

- 多數方法返回 int 型別，代表成功或失敗。
- 底層錯誤碼為 libmosquitto 定義，如：`MOSQ_ERR_SUCCESS`, `MOSQ_ERR_INVAL`, `MOSQ_ERR_NOMEM`, `MOSQ_ERR_NO_CONN` 等。 [mosquitto簡單應用](https://www.cnblogs.com/embedded-linux/p/9386169.html?utm_source=chatgpt.com) [mosquitto函式庫常用的相關函數解析](https://blog.csdn.net/qq_57398262/article/details/124231147?utm_source=chatgpt.com)
- 建議你在開發時，檢查返回值並在 debug 階段記錄錯誤碼。

---

## 版本與注意事項

- 此 C++ 封裝已被標示為 **DEPRECATED**，意味著未來可能不再更新，或對 MQTT v5 新特性支援有限。 [Eclipse Mosquitto](https://mosquitto.org/api/files/cpp/mosquittopp-h.html?utm_source=chatgpt.com)[# mosquittopp with MQTT v5 RPC Response Topics](https://github.com/eclipse-mosquitto/mosquitto/issues/2782?utm_source=chatgpt.com?utm_source=chatgpt.com)
- 若你專案中需要 MQTT v5 完整功能（如 Response Topics、Properties 等），建議直接使用 C 庫 `mosquitto.h` + 自己封裝。
- 在你高效能／嵌入式系統（如 GPU＋網路串流、零拷貝 DMA）中，應評估背景 thread 、事件迴圈 CPU 負載、記憶體動態配置等細節.