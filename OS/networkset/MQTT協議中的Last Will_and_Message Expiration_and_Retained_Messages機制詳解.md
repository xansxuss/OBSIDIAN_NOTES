MQTT 協議中的 Last Will、Message Expiration 和 Retained Messages 機制詳解

原文連結：https://blog.csdn.net/u011487024/article/details/152118048

概述
在 MQTT 協議中，有三個重要機制對於構建健壯的物聯網應用至關重要：Last Will（遺囑消息）、Message Expiration（消息過期時間）和 Retained Messages（保留消息）。這三個機制各有其獨特的作用和應用場景，合理運用它們可以顯著提升 MQTT 系統的可靠性和效率。

1. Last Will（遺囑消息 / LWT）
   1. 定義與原理

    Last Will 是 MQTT 協議中的一個非常人性化的設計。它允許客戶端在建立連接時向 MQTT 代理（Broker）預先聲明一條「遺囑消息」。當客戶端由於非正常原因意外斷開連接時，代理會自動代表該客戶端向指定的主題發布這條預設的消息。

   2. 關鍵特性

    - 觸發條件：僅限非正常斷開；正常斷開不觸發。
    - 設置內容：
        - Will Topic：遺囑消息的主題。
        - Will Message：遺囑消息的內容（載荷）。
        - Will QoS：遺囑消息的服務質量等級（0、1 或 2）。
        - Will Retain：遺囑消息是否為保留消息的標誌。
    - 發布者身份：遺囑消息發布時，其發送者身份是那個意外斷開連接的客戶端。

   3. 工作流程示例

    1. 客戶端 A 連接代理，CONNECT 報文包含：
        - Will Topic: /status/deviceA
        - Will Message: {"status": "offline", "timestamp": "2024-01-01T10:00:00Z"}
        - Will QoS: 1
        - Will Retain: true
    2. 客戶端 B 訂閱主題 /status/deviceA。
    3. 客戶端 A 網絡突然中斷（心跳超時）。
    4. 代理檢測到客戶端 A 非正常斷開。
    5. 代理立即以客戶端 A 的名義向 /status/deviceA 發布遺囑消息。
    6. 客戶端 B 收到遺囑消息，得知設備 A 已意外離線。

   4. 應用場景

    - 設備狀態監控：實時監控設備在線狀態，及時發現設備故障。
    - 異常狀態通知：當設備意外離線時，通知其他系統組件。
    - 系統清理：觸發系統執行清理或恢復操作。
    - 告警系統：集成到監控和告警系統中。

   5. 代碼示例

    ``` python
    import paho.mqtt.client as mqtt

    def on_connect(client, userdata, flags, rc):
        print(f"連接結果: {rc}")

    def on_message(client, userdata, msg):
        print(f"收到消息: {msg.topic} -> {msg.payload.decode()}")

    # 創建客戶端
    client = mqtt.Client()

    # 設置遺囑消息
    will_topic = "/status/sensor001"
    will_message = '{"device_id": "sensor001", "status": "offline", "last_seen": "2024-01-01T10:00:00Z"}'

    # 連接時設置遺囑消息
    client.will_set(will_topic, will_message, qos=1, retain=True)

    # 設置回調函數
    client.on_connect = on_connect
    client.on_message = on_message

    # 連接到代理
    client.connect("broker.example.com", 1883, 60)
    client.loop_forever()
    ```

2. Message Expiration（消息過期時間）
   1. 定義與原理

    Message Expiration 是 MQTT 5.0 引入的重要特性。它允許為消息設置一個存活時間（TTL - Time To Live），超過這個時間後，消息將被代理丟棄，不再投遞給任何訂閱者。

   2. 關鍵特性

    - 設置方式：
        - 發布者設置：在 PUBLISH 報文中包含 Message Expiry Interval 屬性，單位為秒。
        - 代理設置：代理可以在配置中定義全局默認的消息過期時間。
        - 代理覆蓋：代理可以覆蓋或限制客戶端設置的值。
    - 過期處理機制：
        - 傳輸中過期：消息在代理嘗試傳遞給訂閱者時過期，則不會被投遞，從隊列中移除。
        - 存儲中過期：對於設置了 Retain 的消息，如果過期，代理必須移除該保留消息；對於持久會話中的離線消息，過期消息不會被存儲或在存儲期間被清理。
        - 靜默刪除：代理會盡快刪除過期消息以釋放資源，刪除過程是靜默的。

   3. 工作流程示例

      1. 傳感器發布溫度消息，PUBLISH 報文包含：
        - Topic: /sensors/temperature
        - Payload: 22.5
        - Message Expiry Interval: 300（5分鐘）
      2. 代理接收消息並開始計時。
      3. 5分鐘後，消息過期：
        - 如果消息還在隊列中等待投遞，則被丟棄。
        - 如果消息是保留消息，則從保留消息存儲中刪除。
      4. 訂閱者不會收到過期的消息。

   4. 應用場景

      - 防止數據過時：確保訂閱者不會收到已失效的狀態更新。
      - 控制資源佔用：避免代理存儲大量永遠不會被消費的陳舊消息。
      - 時效性信息：處理對時間極其敏感的信息，如實時價格、臨時狀態等。
      - 資源受限環境：特別適用於存儲和計算資源有限的物聯網設備。

   5. 代碼示例

    ``` python
    import paho.mqtt.client as mqtt
    import json

    def on_connect(client, userdata, flags, rc):
        print(f"連接結果: {rc}")

        # 發布帶過期時間的消息
        message = {
            "temperature": 22.5,
            "humidity": 65,
            "timestamp": "2024-01-01T10:00:00Z"
        }

        # 設置消息過期時間為 5 分鐘（300秒）
        client.publish(
            "/sensors/room1", 
            json.dumps(message), 
            qos=1, 
            retain=True,
            properties={"MessageExpiryInterval": 300}  # MQTT 5.0 特性
        )

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect("broker.example.com", 1883, 60)
    client.loop_forever()
    ```

3. Retained Messages（保留消息）
   1. 定義與原理

    Retained Messages 是 MQTT 協議中的一個特性，允許代理保留最後一條消息，並在有新的訂閱者訂閱某個主題時立即發送這條保留消息。這樣，訂閱者可以在訂閱後立即獲得該主題的最新狀態，而無需等待新的消息發佈。

   2. 關鍵特性

    - 存儲機制：代理會保留每個主題的最後一條消息。
    - 發送時機：當有新的訂閱者訂閱