# Ubuntu 18 安裝 Docker 、NVIDIA Container Toolkit ( Verified )

# Check Ubuntu Version

```bash
$ lsb_release -a 
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a855b68-b3b2-4cbf-8891-74844c418a8b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a855b68-b3b2-4cbf-8891-74844c418a8b/Untitled.png)

# Check cuda is available

```bash
$ nvidia-smi
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/65c6c197-9d11-464d-99d4-65b886f921ce/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/65c6c197-9d11-464d-99d4-65b886f921ce/Untitled.png)

```bash
$ nvcc --version
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11a401f1-ff73-492a-bf7f-3ed23b0e21ce/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11a401f1-ff73-492a-bf7f-3ed23b0e21ce/Untitled.png)

# 安裝Docker

## 刪除舊版本

```bash
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

## 設定 Repository

```bash
$ sudo apt-get update
$ sudo apt-get install \\
    apt-transport-https \\
    ca-certificates \\
    curl \\
    gnupg \\
    lsb-release
```

## 加入Docker的官方GPG金鑰

```bash
curl -fsSL <https://download.docker.com/linux/ubuntu/gpg> | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

## 設定stable repository

```bash
echo \\
"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] <https://download.docker.com/linux/ubuntu> \\
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

## 安裝最新版本的Docker Engine

```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## 確認是否安裝成功

```bash
$ docker --version
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/970489c0-ce76-494d-8b19-10c2d8897106/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/970489c0-ce76-494d-8b19-10c2d8897106/Untitled.png)

以上參考自 [docker 官方安裝手冊](https://docs.docker.com/engine/install/ubuntu/)

## 加入群組

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
# logout and login

docker run hello-world
```