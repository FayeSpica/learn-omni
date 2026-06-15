# 零散知识

把平时积累的零散知识点、命令速查、技巧与经验整理公开。

## 分类

> 内容增多后，可按主题拆分子目录（如 `notes/git/`、`notes/linux/`），并在 `mkdocs.yml` 的 `nav` 中分组。

### 昇腾 / NPU

- [昇腾(vllm-ascend)量化特性支持速查](ascend-quantization.md) — W8A8 / W4A8 / W4A4 / MXFP8 / KV C8 支持矩阵

## 如何新增一条知识

1. 在 `docs/notes/` 下新建 Markdown 文件，例如 `docs/notes/git-tips.md`
2. 在 `mkdocs.yml` 的 `nav` → `零散知识` 下添加一行：

   ```yaml
   - Git 技巧: notes/git-tips.md
   ```

3. 可在文章顶部用 `tags` 打标签，便于聚合：

   ```markdown
   ---
   tags:
     - git
     - 速查
   ---
   ```
