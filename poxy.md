我目前是 **Ask 模式**，无法创建文件。以下是完整的 Markdown 文件内容，你可以直接在 VS Code 中新建并保存。

---

Read [](file:///home/wx/WorkSpace/IRL/rsl_rl/rsl_rl/algorithms/dreamwaq.py#1-1), lines 1 to 30

由于我在 Ask 模式下无法创建文件，以下是完整的 Markdown 内容。你可以在 VS Code 中按 `Ctrl+N` 新建文件，粘贴保存到 `/home/wx/WorkSpace/IRL/rsl_rl/docs/git-proxy-setup.md`。

---

```markdown
# Git 通过 Clash Verge 代理访问 GitHub

## 适用场景

中国大陆无法直连 GitHub，本指南教你通过本地 Clash Verge 代理让 Git 正常拉取/推送代码。

---

## 第 1 步：确认 Clash 代理端口

```bash
ss -tlnp | grep clash
```

你应该看到类似输出：

```
LISTEN  0  4096  127.0.0.1:7897  ...
```

> **记下端口号**（常见的是 `7897`，也可能是 `7890`）。

如果你用的是 Clash Verge 图形界面：

1. 打开 Clash Verge
2. 点击 **设置** 或 **Settings**
3. 查看 **HTTP 代理端口**（Port）

---

## 第 2 步：配置 Git 代理

### 方式 A：仅对当前仓库生效（推荐）

```bash
cd /home/wx/WorkSpace/IRL/rsl_rl

git config http.proxy  http://127.0.0.1:7897
git config https.proxy http://127.0.0.1:7897
```

### 方式 B：全局生效（所有仓库）

```bash
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

### 方式 C：使用 SOCKS5 代理（HTTP 代理不行时）

```bash
git config http.proxy  socks5://127.0.0.1:7897
git config https.proxy socks5://127.0.0.1:7897
```

---

## 第 3 步：验证代理是否生效

```bash
git config --get http.proxy
```

应该输出：

```
http://127.0.0.1:7897
```

---

## 第 4 步：正常使用 Git

```bash
git pull --tags origin dreamwaq
git push
git fetch
git clone https://github.com/xxx/yyy.git
```

---

## 取消代理

### 取消当前仓库的代理

```bash
cd /home/wx/WorkSpace/IRL/rsl_rl
git config --unset http.proxy
git config --unset https.proxy
```

### 取消全局代理

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

## 一键开关脚本

将以下内容保存为 `~/bin/git-proxy.sh`：

```bash
#!/bin/bash
# Git Clash 代理开关
# 用法: git-proxy.sh on    开启代理
#       git-proxy.sh off   关闭代理

PROXY_HOST="127.0.0.1"
PROXY_PORT="7897"

case "$1" in
  on)
    git config --global http.proxy  "http://${PROXY_HOST}:${PROXY_PORT}"
    git config --global https.proxy "http://${PROXY_HOST}:${PROXY_PORT}"
    echo "✅ Git 代理已开启: http://${PROXY_HOST}:${PROXY_PORT}"
    ;;
  off)
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    echo "❌ Git 代理已关闭"
    ;;
  *)
    echo "用法: $0 {on|off}"
    ;;
esac
```

赋予执行权限：

```bash
chmod +x ~/bin/git-proxy.sh
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `fatal: unable to access ... port 443` | Git 没走代理 | 检查 `git config --get http.proxy` 是否为空 |
| 代理配置了还是超时 | 端口不对或 Clash 没开 | 打开 Clash Verge，确认"系统代理"已开启 |
| 连接被拒绝 `Connection refused` | Clash 未运行 | 启动 Clash Verge |
| `Unsupported proxy syntax` | 端口号填错了 | 确认 `ss -tlnp \| grep clash` 的输出 |

---

## 快速排查流程

```bash
# 1. 检查 Clash 是否在运行
pgrep -a clash

# 2. 检查端口是否监听
ss -tlnp | grep 7897

# 3. 测试代理是否可用
curl -x http://127.0.0.1:7897 https://github.com -o /dev/null -s -w "%{http_code}"
# 输出 200 表示代理可用

# 4. 检查 Git 代理配置
git config --list | grep proxy

# 5. 测试 Git 连接
GIT_CURL_VERBOSE=1 git ls-remote https://github.com/asWXas/rsl_rl.git HEAD
```

---

## 参考资料

- [Git 官方代理配置文档](https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxy)
- [Clash Verge GitHub](https://github.com/clash-verge-rev/clash-verge-rev)
```

---

复制上面的 markdown 内容，在 VS Code 中 `Ctrl+N` → 粘贴 → `Ctrl+S` 保存到 `/home/wx/WorkSpace/IRL/rsl_rl/docs/git-proxy-setup.md` 即可。