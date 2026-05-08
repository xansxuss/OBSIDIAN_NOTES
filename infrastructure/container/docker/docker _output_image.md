### **方法 1：直接 commit 成新 image**

這個方式會把 container 的當前狀態保存成 image。

```
docker commit 1ea74c2b0be2 my_cuda_image:latest
```

- `1ea74c2b0be2` → 你 container 的 ID
    
- `my_cuda_image:latest` → 新 image 名稱 + tag（可自訂）
    

檢查是否成功：

```
docker images
```

你應該能看到 `my_cuda_image` 出現在列表中

### 匯出image成 tar 檔

如果你要傳給別人或備份：

```
docker export 1ea74c2b0be2 -o cudaimage_backup.tar
```

- `cudaimage_backup.tar` → 你要儲存的檔名
    
- 注意：`docker export` 會只匯出檔案系統，不會保留 image 的歷史與 layer。
    

要再把 tar 重新變成 image：

```
docker import cudaimage_backup.tar my_cuda_image:latest
```

假設你剛把 container commit 成 image 或者已經有 image，`docker images` 的輸出大概長這樣：

```
REPOSITORY          TAG       IMAGE ID       CREATED         SIZE  
my_cuda_image       latest    abc123def456   2 minutes ago   2.1GB  
ubuntu              22.04     1a2b3c4d5e6f   3 weeks ago     77.9MB  
nvidia/cuda         12.6      9f8e7d6c5b4a   2 months ago    3.2GB
```

說明：

- **REPOSITORY** → image 名稱
    
- **TAG** → image 標籤（常用 `latest`）
    
- **IMAGE ID** → image 的唯一 ID
    
- **CREATED** → 建立時間
    
- **SIZE** → image 大小
    

如果你要匯出這個 image，可以用：


```
docker save my_cuda_image:latest -o my_cuda_image.tar
```

這會生成一個 tar 檔，大小大約就是上面 `SIZE` 欄顯示的大小。

### **匯入 Docker**

到目標機器，使用 `docker load`：

docker load -i /path/to/my_cuda_image.tar

執行後會看到類似：

Loaded image: my_cuda_image:latest

---

### **檢查 image**

確認 image 是否已經載入：

docker images

應該會看到 `my_cuda_image` 在列表中。

---

### **運行 container**

直接用：

docker run -it --name new_cuda_container my_cuda_image:latest /bin/bash

這樣你在新機器上就能從這個 image 開 container 了。