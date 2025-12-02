# 🔥 做法 1：同一個 repo 綁兩個 remote（最常用）

你的本地專案只要加兩個 remote：

``` bash
git remote add github git@github.com:USER/REPO.git 
git remote add gitlab git@gitlab.com:USER/REPO.git
```

之後你要推兩邊就：

```bash
git push github main git push gitlab main
```
如果懶得兩次 push，可以設定 **多 remote 一次推完**：

``` bash
git remote add all git@github.com:USER/REPO.git 
git remote set-url --add --push all git@github.com:USER/REPO.git 
git remote set-url --add --push all git@gitlab.com:USER/REPO.git
```

之後：

``` bash
git push all main
```

就會一次推到 GitHub + GitLab。

> ⚠️ 注意：fetch 只會從第一個 URL 抓，push 可以多個。

---

# 🔥 做法 2：GitLab 當鏡像（Mirror GitHub）

如果懶得本地 push 兩次，也不想設定 multi-remote，可以交給 **GitLab 自動鏡像 GitHub**。

GitLab → Create project → _“Mirror repository”_ → 填你的 GitHub URL（需 personal token）。

優點：你只要 push GitHub，GitLab 自動同步。  
缺點：CI/CD 偶爾會延遲，GitHub 斷線就鏡不了。

---

# 🔥 做法 3：GitHub Actions 自動同步 GitLab

如果你更偏愛 GitHub（或 CI/CD 都在 GitHub），你可以讓 GitHub Action 自動 push 到 GitLab：

`.github/workflows/mirror.yml`：

``` yaml
name: Mirror to GitLab 
on:   
	push:     branches: [ "main" ] 
jobs:   
	mirror:     
		runs-on: ubuntu-latest     
		steps:       
			- uses: actions/checkout@v3        
			- name: Push to GitLab         
			  run: |           
			  git remote add gitlab https://oauth2:${{ secrets.GITLAB_TOKEN }}@gitlab.com/USER/REPO.git           
			  git push gitlab main --force
```

優點：全自動。  
缺點：你的 GitLab 就變成 read-only（但這就是鏡像該做的事）。