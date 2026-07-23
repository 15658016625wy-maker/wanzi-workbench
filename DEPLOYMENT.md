# 丸子的工作台 — 公网部署指南

## 当前状况诊断

| 问题 | 结论 |
|------|------|
| 当前访问地址 | `http://localhost:5200/personal-os.html` — 本地地址 |
| 是否仅本地运行 | 是，Flask 服务器运行在你的 Mac 上 |
| WorkBuddy 能否一键发布 | **不能**。CloudStudio 仅支持纯静态网站，本项目有 Flask 后端 |
| 是否包含后端 API | 是，4个 AI 接口 + 1个健康检查 |
| 数据存储方式 | 浏览器 localStorage（设备本地，非云数据库） |
| AI 后端能否公网运行 | 能，但需要部署到支持 Python 的云平台 |

## 部署方案：Railway

### 为什么选 Railway
- 原生支持 Python + Flask，**无需改代码**
- 前端和后端同一域名，相对路径 `/api/...` 天然可用
- 自动 HTTPS
- 环境变量在控制台设置，API Key 不进代码仓库
- 有免费额度（每月 $5 信用，个人用足够）

### 我已完成的部分

1. ✅ 创建 `requirements.txt`（flask, openai, pillow, gunicorn）
2. ✅ 创建 `Procfile`（gunicorn 启动命令）
3. ✅ 在 `server.py` 添加根路由 `/`（访问域名直接打开应用）
4. ✅ 更新 `.gitignore`（排除 .env、.workbuddy、缓存文件）
5. ✅ 初始化 Git 仓库并完成首次提交
6. ✅ 验证 .env 未被提交（API Key 安全）

### 你需要完成的步骤

#### 第1步：注册 GitHub 账号（如已有跳过）
- 打开 https://github.com/signup
- 注册并登录

#### 第2步：创建 GitHub 仓库并推送代码
1. 在 GitHub 网页点击 "New repository"
2. 仓库名填 `wanzi-workbench`，选 Private（私有），点击 Create
3. 在终端执行（替换 YOUR_USERNAME）：
```bash
cd /Users/wangyan/WorkBuddy/2026-07-22-16-57-20
git remote add origin https://github.com/YOUR_USERNAME/wanzi-workbench.git
git branch -M main
git push -u origin main
```

#### 第3步：注册 Railway 账号
- 打开 https://railway.app
- 点击 "Login" → 用 GitHub 账号登录（自动关联）

#### 第4步：部署项目
1. 在 Railway 控制台点击 "New Project"
2. 选择 "Deploy from GitHub repo"
3. 选择刚才创建的 `wanzi-workbench` 仓库
4. Railway 会自动识别 Python 项目，开始构建

#### 第5步：设置环境变量（关键！）
在 Railway 项目的 "Variables" 标签页，逐个添加：
```
ARK_API_KEY=<你的豆包API Key，在本地 .env 文件中查找>
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_TEXT_MODEL=<你的推理接入点ID>
ARK_VISION_MODEL=<你的推理接入点ID>
```
设置后 Railway 会自动重新部署。

#### 第6步：获取公网地址
1. 在 Railway 项目的 "Settings" → "Networking"
2. 点击 "Generate Domain"
3. 会得到一个地址如 `wanzi-workbench-production.up.railway.app`
4. 这就是你的公网 HTTPS 地址

#### 第7步：在 iPhone 上使用
1. 用 iPhone Safari 打开公网地址
2. 点击底部分享按钮 → "添加到主屏幕"
3. 以后从主屏幕图标打开即可，与原生 App 体验一致

### 验署后验证清单
- [ ] 手机流量打开公网地址，页面正常显示
- [ ] 输入文字快速记录，AI 分类正常返回
- [ ] 上传截图识别，AI 分类正常返回
- [ ] 刷新页面后数据仍在（localStorage 保留）
- [ ] 左滑完成、右滑聚焦功能正常
- [ ] 地址栏显示 https://（安全连接）

### 关于数据存储的说明
- 当前数据存在浏览器 localStorage 中
- iPhone 上的数据独立于 Mac，不会同步
- 刷新页面不会丢失数据（localStorage 持久化）
- 如果清除 Safari 缓存或删除 PWA，数据会丢失
- 如需跨设备同步，后续可接入云数据库（如 Supabase）
