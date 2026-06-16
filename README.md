# learn-omni

vLLM / vllm-ascend / vllm-omni 学习笔记，基于 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 构建，部署在 GitHub Pages。

🔗 在线访问：<https://fayespica.github.io/learn-omni/>

## 目录结构

学习笔记按项目分成三块：

```
docs/
├── index.md          # 首页
├── vllm/             # vLLM 内核
│   └── index.md
├── vllm-ascend/      # vllm-ascend 昇腾适配
│   └── index.md
└── vllm-omni/        # vllm-omni 任意模态框架
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

- 按主题放入对应目录：`docs/vllm/`、`docs/vllm-ascend/`、`docs/vllm-omni/`
- 新增文件后，在 `mkdocs.yml` 的 `nav` 中登记对应入口
