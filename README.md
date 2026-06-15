# learn-omni

omni 学习笔记与零散知识库，基于 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 构建，部署在 GitHub Pages。

🔗 在线访问：<https://fayespica.github.io/learn-omni/>

## 目录结构

```
docs/
├── index.md          # 首页
├── omni/             # omni 学习笔记
│   └── index.md
└── notes/            # 零散知识
    └── index.md
mkdocs.yml            # 站点配置与导航
requirements.txt      # 构建依赖
.github/workflows/    # 推送到 main 自动部署
```

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve
```

打开 <http://127.0.0.1:8000> 即可实时预览。

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并发布到 GitHub Pages。

首次启用：仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。

## 新增内容

- omni 笔记放在 `docs/omni/`，零散知识放在 `docs/notes/`
- 新增文件后，在 `mkdocs.yml` 的 `nav` 中登记对应入口
