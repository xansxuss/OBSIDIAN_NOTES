### BGP路徑協定
**『邊界閘門協定』（****Border Gateway Protocol, BGP****）**（RFC 1267）是一種自治系統之間的路徑選擇協定，目前已修正到第四版本（BGP4），詳細規格由 RFC 1771 上規範。一個自治系統大多是由一個或多個網路所構成，並在一個共通的管理環境及路由條件之下，一般都是由**『網際網路服務提供者』（****Internet Service Provider, ISP****）**的網路範圍。在一個 ISP 之內的路徑選擇也大多透過**『內部閘門協定』（****Interior Gateway Protocol, IGP****）**來達成，譬如：RIP-2、OSPF 等等。BGP 主要是被使用於 ISP 之間的路徑選擇。

### BGP 路徑選擇
基本上，BGP 路由器公佈和交換網路上可到達路徑的訊息給其它 BGP 路由器，該路徑訊息包含本身自治系統內和可到達其它自治系統的路徑訊息。BGP 協定也是屬於**『距離向量路徑協定』**，也相同的建構本身路由表，再傳送給相鄰的 BGP 路由器以更新路由表，如此週期性的更新路由表。但 BGP 和 RIP 有BGP 與 RIP 很大的不同點在於，RIP 只宣告可到達路徑的跳躍數目，而 BGP 必需列舉到每一目標的路由。BGP 所言的目標也許是一個自治系統或是一個子網路系統，一個自治系統或許會包含許多網路號碼（或 IP 網路）。因此 BGP 用一個 16 位元的數字來表示一個自治系統，每一自治系統在包含一系列的網路號碼，這些都按照大小順序排列。

BGP 路由器之間交換路徑訊息有兩種情況：起始信息交換和後來訊息更新。當路由器連結上網路時，BGP 路由器將互相交換路由表，而當路由表有變更時，只交換變更部份並不全部傳送。基本上，BGP 路由器之間並不週期性交換訊息，而是路由表變更或發現更佳路徑時才會傳送。雖然 BGP 也是採用單一路由值（Metric）來表示一個路徑的費用，以作為最佳路徑選擇的基礎，路由值的評估可能包含：跳躍數、傳輸速率、延遲時間等等，但最主要的還是政策的考量。

BGP 訊息的傳輸相異於 RIP 和 OSPF，RIP 是包裝於 UDP 封包傳輸，OSPF 是直接利用 IP 封包作多點傳輸，而 BGP 是利用 TCP 協定傳輸。首先 BGP 路由器之間必須建立 TCP 連線，且交換整個路由表，從此以後，新增加或變更內容將被視為路由表的變更而傳送出去。

為了提供路徑選擇的效率，BGP-4 採用**『無層級網域間路徑選擇』（****Classless Inter-Domain Routing, CIDR****）**之技術。BGP-4 路由器之間使用 IP 得前置（Prifix）位元數（IP Netmask 前面連續幾個 1）來簡化網路的**『等級』（****Class****）**，並將自治系統的路徑設定成若干個超級網（Suppernet）以簡化路徑訊息。但採用 CIDR 時，IGP 具有傳遞網路遮罩之功能，還好目前使用的 RIP-2 和 OSPF 都具有此功能。但在一自治系統（或超級網）之內也許會有其它次網域，如圖 6-29 所示，因此，在一自治系統內如使用 CIDR 來作路徑選擇，也必須利用 BGP-4 路由器，此路由器稱之為**『內部** **BGP****』（****Interior BGP, IBGP****）**，如果針對外部網路骨幹則稱之為**『外部** **BGP****』（****External BGP, EBGP****）**。基本上，IBGP 負責自治系統和外部骨幹的橋樑，如訊息路徑並非本自治系統之內，便利用 IBGP 轉送給 EBGP，並由 EBGP 轉送到其它自治系統上，其架構如圖 6-35 所示。
![[fig6-35 EBGP and IBGP path selection.png]]

**圖** **6-35 EBGP** **與** **IBGP** **路徑選擇**

### BGP-4運作方式

BGP-4 是利用 TCP 協定來互相交換訊息，所使用用的著名埠口（Well-Known）為 TCP 179。在 BGP-4 路由器之間又可區分為：前端路由器（Front Router）和同儕路由器（Peer Router）兩種運作方式，前端路由器的運作就像圖 6-32 中 IBGP 和 EBGP 之間的運作，基本上 IBGP 處理內部網路的路徑選擇（如 RIP-2 或 OSPF），並將本身內部網路的路徑圖經 CDIR 協定（IP 位址 + 前置位元數）處理後傳送給 EBGP 路由器（以 BGP-4 協定），此內部路徑圖又稱為**『****AS** **圖』**。同儕路徑選擇就如圖 6-32 中，國際性 ISP 網路上的路由器之間的路徑選擇，也是我們主要探討的 BGP-4 的運作方式。又針對一部 BGP-4 路由器所管轄的範圍也許是由多個網路所構成，並且公告路徑訊息不一定要由 BGP 路由器負責，如果針對較複雜的網路環境，甚至可以利用一部主機電腦來處理 BPG 路徑訊息，並負責傳遞給同儕路由器，因此，在 BGP-4 環境下，一個網路路由節點都稱之為**『****BGP-4** **系統』（****BGP-4 System****）**。

BGP-4 系統起始建立路由表後分別傳送給相鄰的同儕系統（或稱同儕路由器），而後隨著交換訊息來增加、修正或刪除某些路由表參數。BGP-4 並不需要週期性的廣播路由表給同儕系統，而是當路由表有所變更時，再利用**『****Update** **訊息』**通知相鄰之 BGP-4 系統有哪些路徑訊息變更，一般時候同儕系統之間週期性以較短的**『****KeepAlive** **訊息』**告知對方自己還是存在著。當網路有特殊狀況發生或有異常障礙時，BGP-4 系統會以**『****Notification** **訊息』**告知相鄰之同儕系統，譬如，TCP 連線中斷。

如果有一自治系統是透過多個 IBGP 路由器（或稱為 BGP-4 發言者）連結到外部骨幹網路，如圖 6-36 所示。每一個 BGP-4 發言者是利用內部閘門協定（如 RIP-2）交換訊息所得，因此，在每一個 BGP-4 發言者所建構的 AS-圖和其它發言者不一定相同，當這些訊息都前送到 EBGP 路由器時，可能會造成路徑選擇之間的困擾。在這種情況之下，為了達到 BGP-4 發言者之間訊息資料的一致性問題，並不希望任一發言者可以隨意傳送訊息給前端 BGP，而是在所有發言者之間選一個當作所有訊息的出入（Exit/Entry）端點，並由此出入端點集中管理整個自治系統的 AS-圖，其它發言者由此端點的路由器索取最新路徑訊息，再前送給 EBGP 路由器，並且當其它發言者收到變更網路訊息（內部網路或外部網路）時，必需即時利用內部閘門協定傳送給出入端點路由器。當然也希望內部閘門協定的傳送更新訊息能在 IBGP 發言者傳送更新訊息之前完成。
![[fig6-36 Multiple BGP Speakers.png]]
**圖** **6-36** **多重** **BGP** **發言者**

### 路徑訊息資料庫

每一個 BGP 發言者維護一只**『路徑訊息資料庫』（****Routing Information Base, RIB****）**，其包含了三個主要部分：

**(1)**  **Adj-RIBs-In**：Adj-RIBs-In 儲存經過 BGP 發言者之間的**『****Update** **訊息』**學習得來的訊息，這些訊息再經過**『判斷處理』（****Decision Process****）**後所得的路徑訊息，再填入 Adj-RIBs-In 欄位內。

**(2)**  **Loc-RIB**：Loc-RIB 儲存本地路徑選擇訊息，這些訊息也許是由 Adj-RIBs-In 欄位的資料，再經過本地政策所選擇的路徑訊息，以作為本地路徑選擇的主要依據。

**(3)**  **Adj-RIBs-Out**：儲存預備公佈給其它同儕系統的路徑訊。，當本地 BGP 發言者欲公告路徑訊息給其它同濟系統，便將 Adj-RIBs-Out 包裝在**『****Update** **訊息』**內，傳送出去。

基本上，Adj-RIBs-In 儲存未經處理的路徑訊息，它是由其它同儕系統傳送而來的。Loc-RIB 是依照本地路由策略再加上 Adj-RIBs-In 訊息處理所得的路徑訊息。Adj-RIBs-Out 是經過處理後，發現有變更訊息而必須傳送給其它同儕系統的路徑訊息。

### 路徑訊息宣傳與儲存

為了達成 BGP-4 協定的運作，一個路徑被定義成一個單元的訊息，該訊息是由一對的目的位址所形成的途徑（Path），每一路徑的儲存與宣傳如下：

**○** 路徑被一對 BGP 發言者以**『****Update** **訊息』**宣傳，其方式如下：該系統可以到達的 IP 位址儲存**『****Update** **訊息』**的**『網路層可到達訊息』（****Network Layer Reachability Information, NLRI****）**欄位中，並且該路徑的屬性（Attribute）也儲存於屬性欄位上。

**○** 路徑訊息被儲存於 RIB 資料庫中，並區分為：Adj-RIBs-In、Loc-RIB 與 Adj-RIBs-Out 三個不同欄位。

BGP-4 提供三種方法來讓 BGP 發言者通知同儕系統有哪些先前所宣傳的路徑已經不再有效，也就是讓 BGP 發言者取消（Withdrawn）該服務路徑：

**1.**    在先前宣傳之路徑的 IP 前置位址加入**『****Update** **訊息』**的**『****Withdrawn Routes****』**欄位上，傳送給相鄰之同儕系統，表示該路徑已不再有效使用。

**2.**    將另一路徑更新已不再使用路徑的**『網路可到達訊息』（****NLRI****），**並宣傳出去。

**3.**    關閉 BGP 發言者之間的連線，表示先前所宣傳的路徑訊息被移除掉。

### BGP-4訊息格式

BGP-4 訊息是利用 TCP 連線來互相宣傳，每一封包最大的容量為 4096 個位元組（Bytes），BGP 發言者之間傳遞有：Open Message、Update Message、Notification Message 與 Keep-alive Meaasge 等四種訊息。這四種訊息都使用相同的封包標頭，標頭長度為 19 Bytes，如圖 6-37 所示，標頭欄位之功能如下：

**○** **Marker**：內容為一個認證訊息，讓訊息接收者可以預定該值。如果為 Open 訊息但沒有認證功能時，該欄位設定為全部 1；如果有加入認證訊息，接收端可利用該訊息來確定資料的正確性。

**○** **Length**：表示整個封包的長度，該數值一定在 19 和 4096 之間，BGP 訊息並沒有填補（Padding）資料。

**○** **Type**：表示該訊息的型態：

**1.** 為**『開啟訊息』（****Open Message****）**

**2.** 為**『更新訊息』（****Update Message****）**

**3.** 為**『通知訊息』（****Notification Message****）**

**4.** 為**『保持存活訊息』（****Keep-alive Message****）**。

**○** **Data**：內容為各訊息型態的資料（Open、Update、Notification、或 Keep-alive 訊息），其長度依照各訊息型態而不同。
![[fig6-37 BGP-4 packet header.png]]
**圖** **6-37 BGP-4** **之封包標頭**

以下針對欄位 Type 所區分的四種訊息加以介紹，各種訊息是附加在封包標頭的後面，也就是如圖 6-37 上的 Data 欄位（依照各種訊息而有不同的長度）。

**開啟訊息（****Open Message****）**

Open Message 是建立兩個閘門之間的交談連線，它是連線後第一個訊息，如欲傳送其他訊息之前，必須使用Open Message 建立雙方對談連線。圖 6-38 為 Open Message 的資料內容，各欄位功能如下：

**○** **Version****（****Ver****）：**表示該封包的 BGP 版本。

**○** **Autonomous System****（****AS****）：**表示該發送封包者所在的自治系統編號。

**○** **Hold-Time****（****HT****）：**表示保持時間，在這時間內沒有回應的路由器，都被假設已失去功能。

**○** **BGP Identifier****（****BGP****）：**傳送該封包的外部閘門號碼（IP 位址）。

**○** **Optional Parameter Length****（****O-Len****）：**表示緊接在後的 Optional 欄位的長度。

**○** **Optional Parameter****：**任意參數。目前僅使用於認證（Authentication）訊息，有兩個部份：Authentication 
code 和 Authentication data。

![[fig6-38 Data content of Open Message.png]]
**圖** **6-38 Open Message** **的資料內容**

**更新訊息（****Update Message****）**

Update Message 是被用來更新同儕系統之間的路徑訊息，使各個路由器都能建立一個可觀察整個網路的拓樸圖。Update Message 是由 TCP 連線完成已確定訊息的可靠度。當網路上有任何路徑被抽離，該相連之 BGP 發言者便利用 Update Message 告知相鄰之閘門。圖 6-39 為 Update Message 的資料內容，各欄位功能如下：

● **Unfeasible Router Length****（****URL****）：**表示緊接著後面Withdrawn Router 欄位的長度。

● **Withdrawn Router****（****WR****）：**表示有那些已被抽離的路由器（IP 位址表示），可變長度表示之。

● **Total Path Attribute Length****（****TPAL****）：**表示後面緊接著兩個有關屬性欄位的長度。

● **Path Attribute****（****PA****）：**路徑屬性。描述該路徑之特性有能是下列屬性：

**○** **Origin****：**指派屬性（Mandatory attribute）。為原系統指定之路徑。

**○** **AS Path****：**經系統指定之經由多個自治系統片段所構成的路徑。

**○** **Next Hop****：**指派屬性。指定經由邊界網路的下一路徑可到達目的位址。

**○** **Mult Exit Disc****：**選擇屬性（Option attribute）。在多點路徑之中辨別可到達鄰近自治系統之路徑。

**○** **Local Pref****：**任意屬性（Discretionary attribute）。描述任意路由的級數。

**○** **Atomic Aggregate****：**任意屬性。被使用在表現有關路徑選擇的訊息。

**○** **Aggregator****：**選擇屬性。包含有關路徑聚集的訊息。

● **Network Layer Reachability Information****（****NLRI****）：『網路層可到達訊息』**是作為 BGP 發言者宣傳可到達的路徑，它包含一串列的 IP 位址的網路位址（前置位元），每一 IP 前置位元（IP Prefix）表示可到的路徑區段。

![[fig6-39 Data content of updata message.png]]
**圖** **6-39 Update Message** **的資料內容**

**通知訊息（****Notification Message****）**

當外部閘門發現任何異常狀態，便使用該訊息告知相鄰閘門，或被使用於中斷連線。圖 6-40 為 Notification Message 的資料內容，各欄位功能如下：

● **Error Code****（****EC****）：**該封包表示錯誤的種類，如下列：

**○** **Message Header Error****：**所傳送的封包標頭發生錯誤。

**○** **Open Message Error****：**所傳送的 Open Message 錯誤，如版本、自治系統或 IP 號碼、或認證錯誤。

● **Update Message Error****：**所傳送的 Update Message 錯誤，如屬性不合等。

● **Hold Time Expired****：**表示 Hold Time 溢時，將該區段之BGP 被視為沒有功能。

● **Finite State Machine Error****：**協定流程錯誤。

● **Cease****：**結束 BGP 連線，

**○** **Error Subcode****：**錯誤型態的附加描述碼。

**○** **Error Data****：**內容為有關錯誤型態的資料。

![[fig6-40 Data content of notification message.png]]
**圖** **6-40 Notification Message** **的資料內容**

**存活訊息（****Keep-alive Message****）**

用來測試連線中斷或 TCP 連線另一端的 BGP 路由器是否故障了，傳送訊息的建議是每 30 秒一次。該訊息並沒攜帶任何資料，因此沒有 Data 欄位（如圖 6-34），只在 Type 欄位上標示為 Keep-alive Message（Type = 4），整個封包長度為 19 Bytes。BGP 發言者就是利用此短的訊息，週期性的通知同濟系統自己還是存在著。

###  **BGP-4** **路徑屬性**

BGP-4 雖然也是採用**『距離向量演算法』**，但它和 RIP-2 之間有很大的不同點，RIP-2 只利用**『跳躍距離』（****Hop Count****）**來評估路徑費用。而 BGP-4 利用許多『路徑屬性』（Path Attributes）來評估每一條路徑的費用，這些路徑屬性將被包裝在 Update 訊息內（如圖 6-39），以讓 BGP 發言者之間來互相傳遞。BGP-4 路徑屬性可區分為以下四大類：

**1.** **著名指定性（****Well-known Mandatory****）**

**2.** **著名隨意性（****Well-known Discretionary****）**

**3.** **選項過渡性（****Optional Transitive****）**

**4.** **選項非過渡性（****Optional Non-transitive****）**

**『著名屬性』（****Well-known Attribute****）**必須經過所有 BGP-4 的製造者共同確認，又著名指定的屬性必需被包含每一 Update 訊息內；而著名隨意的屬性可視環境需要來決定是否要加入到 Update 訊息內。除了著名屬性外，每一路徑也許包含若干個選項性（Optional）屬性，但這些屬性並不需要 BGP-4 製造商共同確認，而是各個廠商依照環境需求而增加，某些選項屬性不被其它廠商採用，在評估路徑費用時可以不用理會。過渡性（Transitive）屬性是屬於較區域性或特殊自治系統所制定的屬性；非過渡性是針對某些特殊路徑所制定，也會制定相對應的路徑規則。在 RFC-1771 中規範有許多路徑屬性，但並非所有屬性都會被一般製造商採用，我們以 Cisco 公司所實現的路徑屬性來介紹，Cisco 採用：Weight、Local Preference、Multi-exit Discriminator、Origin、AS_path、Next hop 與 Community 屬性，分別介紹如下：

**衡權屬性（****Weight Attribute****）**

衡權屬性是 Cisco 所定義的本地路徑屬性，該屬性並不宣傳給其它同儕系統（或相鄰路由器）。如果路由器學習到同一目的位址有一個以上的路徑可以到達，便將衡權量較高的路徑填入路由表，並宣傳給其它相鄰路由器。如圖 6-41 中，路由器 A （AS 100）學習到（或收到宣傳訊息）兩條路徑可以到達 127.16.1.0/24 網路（AS 200），一條是經由路由器 C，衡權屬性設定為 100；另一條是經由路由器 B，衡權屬性設定為 50，因此，路由器 A 選擇路由器 C 路徑並填入路由表。

![[fig6-41Weight Attribute.png]]
**圖** **6-41** **衡權屬性**

**本地優先屬性（****Local Preference Attribute, Local_Pref****）**

Local_Pref 是屬於**『著名隨意性』**的屬性，被使用於表示選定本地自治系統的出口。它不同於衡權屬性，本地 BGP 發言者會將該訊息宣傳給相鄰的路由器，尤其在有多點出口的環境裡，Local_Pref 告知相鄰路由器哪一個才是本地自治系統的優先出口位址。如圖 6-42，AS 100 接收到兩個由 AS 200（172.16.1.0/24 網路）所宣傳的訊息，當路由器 A 收到由路由器 C 的宣傳訊息，則依照網路狀態設定為 Local_Pref = 50；路由器 B 收到由路由器 D 的宣傳，則設定 Local_Pref = 100，路由器 A 和 B 互相傳遞訊息後，判斷經由路由器 D 到達 172.16.1.0/24 的 Local_Pref 較高，因此，AS 100 前往 AS 200 網路路徑便選擇經由路由器 D。

![[fig6-42Local Preference Attribute, Local_Pref.png]]
**圖** **6-42** **本地優先屬性**

**多重出口鑑別屬性（****Muti-Exit Discriminator Attribute, MED****）**

或稱為**『向量值』（****Metric****）**屬性。MED 是用來建議外部自治系統進入本自治系統的向量值。如圖 6-43 所示，AS 200 的路由器 C 向 AS 100 系統的路由器 A 宣傳由本路徑進入本系統的 MED = 10；另一方面，路由器 D 宣傳進入本自治系統的 MED = 5，在 AS 100 內經過訊息交換後，判斷經由路由器 D 到 AS 200 系統費用較低，便選用該路徑到達 AS 200 系統。

![[fig6-43 Muti-Exit Discriminator Attribute, MED.png]]
**圖** **6-43** **多重出口鑑別屬性**

**起源屬性（****Origin Attribute****）**

Origin 也是屬於**『著名指定屬性』**，是用來表示該路由是 BGP 以何種途徑所學習得來。Origin 可能是下列三種數值之一：

● **IGP****：**表示該路由是經由**『內部閘門協定』（****IGP****）**學習得來的。

● **EGP****：**表示該路由是由**『外部邊界閘門協定』** **（****Exterior Border Gatway Protocol, EBGP****）**學習得來的。

● **Incomplete****：**表示該路由起源不明或是經由其它通訊協定學習得來，這種屬性的路由大多是被重新分配到另一個 BGP 上。

**自治系統路徑屬性（****AS_Path Attribute****）**

AS_Path 也是屬於**『著名指定屬性』**，它是被用來表示某一路由所經過的路徑。當一個路由被宣傳而經過某一路由器時，該路由器便將它的AS 識別值加入到此路由的次序串列中（AS_Path），再宣傳給其它自治系統，因此，由 AS_Path 屬性就可以觀察到該路由所經過的路徑。由圖 6-44 可以觀察到，AS 1 的起源路由為 172.16.1.0/24，並向 AS 2 與 AS 3 宣傳該路由，宣傳時將 AS_Path 設定為 [1]，當 AS 2 和 AS 3 收到該宣傳，便將自己的 AS 識別碼加到 AS_Path 內後傳遞給下一個自治系統（[3.1] 與[2.1]）。但當 AS 1 收到 AS_Path = [3.1] 或 [2.1]，也就知道該路由是由本地的路由器發出，便拒絕該路由訊息。又譬如 AS 2 收到 AS 3 系統所宣傳的 AS_Path = [3.1]，也判斷自行由 AS_Path = [1] 的路由路徑會較為短捷，因此，它會選用 AS_Path = 1 的路由到達 AS 1。

![[fig6-44 AS_Path Attribute.png]]
**圖** **6-44** **自治系統路徑屬性**

**下一跳躍屬性（****Next-Hop Attribute****）**

Next-Hop 屬性是針對 EBGP 的下一跳躍的位址規範，也就是邊界路由器的位址。在 RBGP 的同儕路由器之間，Next-Hop 是以一個 IP 位址來表示，也表示由此位址可進入哪一自治系統。如圖 6-45 所示，AS 200 系統宣傳進入 172.16.1.0/24 網路的 Next-Hop = 10.1.1.1，AS 100 的路由器 A 收到該訊息後，再宣傳給路由器 B，表示欲進入 176.16.1.0/24 網路的下一跳躍位址為 10.1.1.1。
![[fig6-45 Next-Hop Attribute.png]]
**圖** **6-45** **下一跳躍屬性**

**共同體屬性（****Community Attribute****）**

Community 屬性是提供一種方法來處理群體性之目的位址。可利用 Community 來規劃某一群組目的位址成為一個共同體，以作為決定是否傳送路由訊息給這一群組的成員。一般事先定義有下列三種共同體屬性：

● **No-Export****：**不要宣傳此路由給同儕邊界路由器（EBGP）。

● **No-Adverties****：**不要宣傳此路由給任何同儕路由器。

● **Internet****：**宣傳此路由給 Internet 共同體，所有路由器都屬於此共同體的成員。

圖 6-46 ~ 48 為上列三種屬性的宣傳傳遞方式，其中 圖 6-46 表示 No-Export 屬性的宣傳方式；而 圖 6-47 表示 No-Adverties 的宣傳方式；另外 圖 6-48 是 Internet 屬性的方式。

![[fig6-46 Community Attribute No-Export.png]]
**圖** **6-46** **共同體屬性之** **No-Export**
![[fig6-47 Community Attribute No-Advertise.png]]
**圖** **6-47** **共同體屬性之** **No-Advertise**
![[fig6-48 Community Attribute Internet.png]]
**圖** **6-48** **共同體屬性之** **Internet**

### 判斷處理

判斷處理（Decision Process）是由週期性的宣傳訊息中選擇適當的路由，當 BGP-4 路由器收到同儕系統所宣傳的訊息將其儲存於 RIB 的 Adj-RIB-in，再經過判斷處理後儲存於 Loc-RIB 資料庫內。由經過處理後的路由及其它訊息儲存於 Adj-RIB-Out，而準備向其他同儕系統宣傳。選擇適當路由是依照每一路由的屬性來作判斷的依據，依照路由的屬性關係來評估每一路由的優先等數（Degree of Preference），以選擇較高的優先等數的路由。在 RFC 1771 中，以三個處理時相（Phase）來測試各種不同的現象：

**1.** **Phase 1****：**負責計算由相鄰自治系統 BGP 發言者所傳送路由的優先等數，並將較高優先等數的路由向本自治系統之 BGP 發言者宣傳，每一較高優先級數路由都是一個獨立路徑。

**2.** **Phase 2****：**Phase 1 完成後，再執行 Phase 2。Phase 2 負責由較高優先級數的路由中選擇較適合的路由，而將其儲存於 Loc-RIB 中，每一路由也是一個獨立路徑。

**3.** **Phase 3****：**當 Loc-RIB 已被更新完成後，再執行 Phase 3。Phase 3 負責散播 Loc-RIB 上的路由給相鄰的同濟自治系統，針對路由的聚集和訊息簡化使達到最佳化的處理，也是在此時相裡完成。

前面我們介紹 Cisco 公司所使用的路由屬性，以下也針對 Cisco 公司的路由選擇規則來介紹。BGP 也許由多個來源的宣傳收到相同的路由，但它只能再選擇其中一個優先級數較高的路由。當某一路由被選擇到時，必須將其存放於 IP 路由表內（或 Loc-RIB），並傳播給相鄰的自治系統，BGP 依照下列準則選擇最佳路由：

● 如果有某一路徑被描述為下一路徑（Next Hop），便往該路徑傳送。但如果該路徑已經到達不了，便將其刪除掉（或不存在下一路徑），再依照下列步驟判斷選擇最佳路由。

● 首先選擇較高衡權（Weigth）屬性的路徑。

● 如果路徑的衡權屬性相同，則選擇最高『本地優先』（Local Preference）屬性的路徑。

● 如果本地優先屬性相同，則選擇本路由器之 BGP 執行中產生的路徑。

● 如果路由都沒有起源屬性，則選擇較短『自治系統路徑』（AS_Path）屬性的路徑。

● 如果都是相同長度的 AS_Path，則依照『起源』（Origin）屬性選擇較低屬性的路徑（IGP 比 EGP 屬性低，EGP 比 Incomplet 屬性低）。

● 如果起源屬性相同時，則選擇最低『多重出口鑑別』（MED）屬性的路徑。

● 如果所有屬性都相同時，則選擇最靠近相鄰的 IGP 路徑。

● 還是無法分出時，則選擇被標示為 BGP 之最小 IP 位址的路徑。

以上是針對網路環境來考量路徑選擇，但 BGP-4 允許以政策為基礎的路徑選擇，政策是由自治系統的系統管理者決定，並由規劃檔中設定成為路由值的一部份。政策的決定並不是通訊協定的一部份，但政策規格允許 BGP 實際應用時有多重選擇時，可作為路由器之間選擇的依據，並控制資料的重新分配，以配合路徑選擇的政策、安全性、商業性的考量。




### reference
 [BGP路徑協定](https://www.tsnien.idv.tw/Internet_WebBook/chap6/6-15%20BGP%20%E8%B7%AF%E5%BE%91%E5%8D%94%E5%AE%9A.html)