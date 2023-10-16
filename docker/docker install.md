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

---

# 安裝 NVIDIA Container Toolkit

參考自 [nvidia docker installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## 先決條件

The list of prerequisites for running NVIDIA Container Toolkit is described below:

1. GNU/Linux x86_64 with kernel version > 3.10
2. Docker >= 19.03 (recommended, but some distributions may include older versions of Docker. The minimum supported version is 1.12)
3. NVIDIA GPU with Architecture >= Kepler (or compute capability 3.0)
4. [NVIDIA Linux drivers](http://www.nvidia.com/object/unix.html) >= 418.81.07 (Note that older driver releases or branches are unsupported.)

## 設定 NVIDIA Container Toolkit

設定 GPG key

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \\
&& curl -s -L <https://nvidia.github.io/nvidia-docker/gpgkey> | sudo apt-key add -\\
&& curl -s -L <https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list> | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

## 安裝 nvidia-docker2

雖然聽說新版本的docker不需要docker2套件，不過官方的文件中沒有提到

```bash
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2 -y
```

## 重新啟動docker

```bash
$ sudo systemctl restart docker
```

## 確認docker是否能夠讀取到GPU

```bash
$ docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00dc877a-1adf-4b15-a2e9-7040392e8c26/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00dc877a-1adf-4b15-a2e9-7040392e8c26/Untitled.png)

如果可以看到nvidia-smi的畫面則代表可以讀取到了。