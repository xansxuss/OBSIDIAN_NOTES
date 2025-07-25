Ubuntu 24.04 建立NFS（Network File System）。

1. 安裝 NFS 服務端
``` bash
sudo apt update
sudo apt install nfs-kernel-server
```
請用 apt-cache policy nfs-kernel-server 檢查版本。

2. 建立共享資料夾
假設你要分享 /srv/nfs/shared 這個資料夾。
``` bash
sudo mkdir -p /srv/nfs/shared
sudo chown nobody:nogroup /srv/nfs/shared
sudo chmod 755 /srv/nfs/shared
```
權限設成 nobody:nogroup 是為了讓非特定用戶訪問，這可能導致安全風險，實務上建議用特定用戶或群組管理，並搭配防火牆限制來源 IP。

3. 編輯 /etc/exports 設定分享規則
編輯 /etc/exports，加入：
``` 
/srv/nfs/shared    192.168.1.0/24(rw,sync,no_subtree_check)
```
    192.168.1.0/24 是允許訪問的網段，請依你實際網路修改。
    rw 允許讀寫。
    sync 確保資料同步寫入，較安全但效能較低。
    no_subtree_check 防止子目錄檢查造成的問題。

4. 套用設定並重啟服務
``` bash
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```
5. 防火牆設定（假設使用 ufw）
確保 NFS 相關 port 開放：
``` bash
sudo ufw allow from 192.168.1.0/24 to any port nfs
sudo ufw reload
```
6. 用戶端掛載 NFS
在其他 Ubuntu（或 Linux）機器，先安裝 NFS 客戶端：
``` bash
sudo apt install nfs-common
```
然後掛載：
``` bash
sudo mount -t nfs <伺服器IP>:/srv/nfs/shared /mnt
```
7. 常駐掛載（選擇性）
編輯 /etc/fstab，加入：
```
<伺服器IP>:/srv/nfs/shared    /mnt    nfs    defaults    0 0
```

但有可能 systemd 或防火牆默認策略更嚴格，建議檢查 systemctl status nfs-kernel-server 以及 journalctl -xe 以防有錯誤訊息。

尤其是 NFSv4 的設定，/etc/idmapd.conf 的 domain 設定和用戶映射很重要。

NFS 真的還是最優解？
NFS 是老牌的共享協議，但遇到高頻 I/O 或安全性需求，它的效率和安全性常常被質疑。你是否考慮過用 Samba（尤其 Windows 混合環境）或更先進的分布式檔案系統（如 Ceph、GlusterFS）？

#### 權限設成 nobody:nogroup 是為了讓非特定用戶訪問，這可能導致安全風險，實務上建議用特定用戶或群組管理，並搭配防火牆限制來源 IP。

🔍 原因分析：為什麼會看到 nobody:nogroup？
在許多入門教學中，會用以下方式來建立 NFS 共享資料夾：
```bash
sudo chown nobody:nogroup /srv/nfs/shared
sudo chmod 755 /srv/nfs/shared
```
    這其實是源自於 NFSv3 的一種「懶人」共享方式：
    nobody 是一個無特權用戶，對應於匿名訪客身份（anonymous user），配合 /etc/exports 裡的 anonuid, anongid 參數時用到。
    這做法的目的很單純：
    允許任意用戶在客戶端掛載這個目錄；
    不做 UID/GID 對應（或強制 anonymous 身份）；
    降低 NFS 授權複雜度，但提升了風險。
⚠️ 問題與風險
1. 安全性極低
    使用 nobody:nogroup 意味著：
    所有客戶端用戶寫入資料會以 nobody 身份存在 server 上；
    無法追蹤誰寫了什麼（audit 無法落實）；
    搭配 rw 權限時，只要能連上網路，就能修改資料，沒有用戶層級防護。

2. 權限一致性問題
    在 NFSv3 中，伺服器無法得知客戶端用戶的身分，純粹靠 UID/GID 比對。
    而如果 server 和 client UID/GID 不一致，例如：
    alice 在 server 是 UID=1001
    alice 在 client 是 UID=2001
    那麼這位 Alice 會變成別人，或甚至無法存取檔案。

✅ 改進方案：安全又實用的權限設計
建立專用群組與使用者
在 server 上：
``` bash
sudo groupadd nfsusers
sudo useradd -r -g nfsusers -s /usr/sbin/nologin -d /srv/nfs/shared nfsuser
sudo chown nfsuser:nfsusers /srv/nfs/shared
```
你可以指定一個既有用戶，例如 eray：
``` bash
sudo chown eray:nfsusers /srv/nfs/shared
```
然後只允許同 UID/GID 的用戶存取，或在 exports 明確映射身份：
``` 
/srv/nfs/shared 192.168.1.0/24(rw,sync,subtree_check,all_squash,anonuid=1001,anongid=1001)
```
這樣一來，即使客戶端是匿名訪客，也會強制以 UID=1001 這個用戶寫入，兼顧授權與可追蹤性。

✦ 更進階：NFSv4 + Kerberos
如果你非常在意安全性：
使用 NFSv4 + krb5p（加密認證與通訊）
配合 LDAP / Active Directory 做身份對應
強制 client mount 時用 Kerberos ticket 認證
這樣就能像 Samba/SMB 一樣做到「使用者級別授權 + 加密 + 識別」。
但坦白說，這在一般家用網路或小公司實作起來略顯麻煩。
✦ 最實用妥協方案
建立特定使用者 / 群組
使用 all_squash 與 anonuid/anongid 控制寫入身份
配合防火牆限制 NFS port 僅對內部網段開放（e.g. ufw allow from 192.168.1.0/24 to any port nfs）
記錄 NFS access log（雖然不是預設功能，可透過 auditd 或 rsyslog 擴充）