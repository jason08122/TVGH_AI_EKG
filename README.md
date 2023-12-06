# Installation guide

### 安裝 Git
```typescript!
$ sudo apt-get install git

$ git clone https://github.com/jason08122/TVGH_AI_EKG_WEB.git
```

### Add Docker's official GPG key

```typescript
# Add Docker's official GPG key:
$ sudo apt-get update
$ sudo apt-get install ca-certificates curl gnupg
$ sudo install -m 0755 -d /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

$ sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

### Add the repository to Apt sources

```typescript
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

$ sudo apt-get update
```

### Install the Docker packages

```typescript
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Build up fastapi docker image

```typescript!
$ cd TVGH_AI_EKG_WEB/TVGH_AI_EKG

$ sudo docker build -t "tvgh-ai-ekg:v1.2.12" .
```

### Deploy the service with docker-compose

```typescript!
$ sudo apt install docker-compose

$ sudo docker-compose up -d
```

### 處理 Port Forwarding

 - Setting->Network->Advanced->Port Forwarding->Add
 
    | Name | Protocol | Host IP | Host Port | Guest IP     | Guest Port |
    | ---- | -------- | ------- |    ---    |    ---       |    ---     |
    | Http | TCP      | 127.0.01|  6060     |  10.0.2.15   |  32080     |


### 開啟網頁

    - http://localhost:6060/tvghTwoDiagnos.html