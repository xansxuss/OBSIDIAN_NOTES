在Ubuntu上建立 SMB（Samba）伺服器可以讓 Windows、macOS 或其他 Linux 裝置存取你電腦的資料夾。

1. 安裝 Samba 套件
    ``` bash
    sudo apt update
    sudo apt install samba
    ```
2. 建立共享資料夾
    以 /srv/samba/share 為例：
    ``` bash
    sudo mkdir -p /srv/samba/share
    sudo chown nobody:nogroup /srv/samba/share
    sudo chmod 0775 /srv/samba/share
    ```
    🛑 若要讓特定用戶存取，建議使用：

    ``` bash
    sudo chown youruser:yourgroup /srv/samba/share
    sudo chmod 0750 /srv/samba/share
    ```
3. 編輯 Samba 設定檔
    編輯 /etc/samba/smb.conf
    ``` bash 
    sudo nano /etc/samba/smb.conf
    ``` 
在檔案尾端新增以下區段：
    ```
    [PublicShare]
    path = /srv/samba/share
    browseable = yes
    writable = yes
    guest ok = yes
    read only = no
    create mask = 0664
    directory mask = 0775
    force user = nobody
    ```
    guest ok = yes 會允許匿名用戶，但不安全；若要限制帳號，請改為：
    ```
    guest ok = no
    valid users = youruser
    ```
4. 建立 Samba 使用者（非匿名）
    ``` bash
    sudo smbpasswd -a youruser
    ```
    youruser 必須是已存在的 Linux 使用者。

5. 重新啟動 Samba 服務
    ``` bash
    sudo systemctl restart smbd
    sudo systemctl enable smbd
    ``` 
6. 防火牆設定（如有開啟 UFW）
    ``` bash
    sudo ufw allow 'Samba'
    ```
7. 測試連線
    Windows 檔案總管 輸入：
    ```
    \\<你的 Ubuntu IP>\PublicShare
    ``` 

    Linux / macOS 使用：
    ``` bash
    smbclient //192.168.x.x/PublicShare -U youruser
    ```

8. 卸載（umount）NFS 掛載點，只需要用 umount 指令。對 /mnt 來說，就是這樣：
``` bash
sudo umount /mnt
```
🔍 注意事項
✅ 檢查掛載狀態：
如果你不確定是否已經掛載成功，可以先列出掛載點確認：
``` bash
mount | grep nfs
# 或
df -hT | grep nfs
```
⚠️ 若無法卸載，可能的原因有：
有程式佔用該目錄（常見）

使用 lsof 查誰在佔用：
``` bash
sudo lsof +D /mnt
```
或直接用 fuser：
``` bash
sudo fuser -vm /mnt
```
NFS server 中斷或死掉了

客戶端會卡住，可能需要強制卸載：
``` bash
sudo umount -f /mnt
```
如果掛載有掛進背景自動掛載系統（如 autofs）

建議先停掉 autofs 相關守護行程再手動 umount。

🚀 延伸：批次卸載所有 NFS 掛載
如果你有很多個 NFS 掛載點要卸載：
``` bash
sudo umount -a -t nfs,nfs4
``` 
這會卸載所有 NFS 或 NFSv4 掛載點。

實務建議
| 類型	          | 建議設定|
| ------ | -------- | -------------------- |
| 家用臨時共享	   | guest ok = yes + force user = nobody|
| 公司／正式共享	   | guest ok = no + 使用帳號管理 + 防火牆限制 IP|
| 進階安全	       | 考慮整合 /etc/samba/smbusers、AD 或 LDAP 授權|

smbclient，它是 Samba 提供的命令列工具，可用來從 Linux 或 macOS 存取 SMB 共享資料夾，功能類似 Windows 的「檔案總管」。

1. 安裝 smbclient（Ubuntu 24.04）
``` bash
sudo apt update
sudo apt install smbclient
```

    安裝後可以透過以下指令測試：
    ``` bash
    smbclient -L //localhost -N
    ```
    這會列出本機共享（使用匿名連線 -N，如果不支援會報錯），也可以改用登入帳號：

    ``` bash
    smbclient -L //localhost -U youruser
    ```
    如果你已設定密碼，會提示你輸入 Samba 密碼（不是 Linux 密碼）。

2 常用測試指令
``` bash
smbclient //server-ip/share-name -U youruser
```
登入後你會進入類似 FTP 的介面，常見操作指令：
``` bash
ls：列出檔案

get file.txt：下載

put local.txt：上傳

exit：離開
``` 
若你想做「掛載」到本機檔案系統（非交互式操作），可以改用 cifs-utils：
``` bash
sudo apt install cifs-utils
```
並使用 mount -t cifs（需要 root 權限）掛載 SMB 共享，這邊就會進入 NAS 或 Windows share 掛載範疇。
smbclient 的互動式 shell，這個介面比較像 FTP、早期 DOS 的檔案操作介面，不是 Linux shell，所以指令像 ll 或 ls -l 是不支援的。

### 正確的 smbclient 指令集（類似 FTP）

| 指令	| 說明 |
| ------ | -------- | -------------------- |
|ls	|列出目前目錄下的檔案|
|cd dir	|進入子目錄|
|get file	|下載檔案|
|put file	|上傳本機檔案|
|mget *	|一次抓多個檔案（可搭配萬用字元）|
|mput *	|一次上傳多個檔案|
|pwd	|顯示遠端目前路徑|
|lcd	|切換本機端路徑（local change dir）|
|exit 或 quit	|離開|
