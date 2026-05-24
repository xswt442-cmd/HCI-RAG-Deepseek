# HCI-RAG-Deepseek: 基于 DeepSeek 的 RAG 信息检索系统

## 项目简介

本系统实现了 **检索增强生成 (RAG)** 技术，结合 DeepSeek 大语言模型与本地向量数据库，支持对外部知识库进行精准问答。

## 环境配置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-your-key-here
```

## 知识库

| 文档 | 内容 |
|------|------|
| `deepseek的技术报告.txt` | DeepSeek V3/V4 架构、参数、上下文窗口、训练技术 |
| `Python编程技巧.txt` | 代码可读性、数据结构、装饰器、性能优化、错误处理 |
| `人工智能发展简史.txt` | 从图灵测试到 GPT-5 的 70 年 AI 发展历程 |

### 切分策略

- **Chunk 大小**: 400 字符 — 在保留上下文完整性和检索精准度之间的平衡
- **Overlap**: 80 字符 — 确保边界信息不丢失
- **分隔符**: `\n\n` → `\n` → `。` → 空格 — 优先保持句子完整性
- **总计**: 38 个 chunk

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 向量数据库 | ChromaDB | 轻量级、Python 原生、支持持久化 |
| Embedding | BAAI/bge-small-zh | 中文语义理解优秀，模型体积小 |
| LLM | DeepSeek API (deepseek-chat) | 与知识库内容一致，性价比高 |
| 框架 | LangChain | 完整的 RAG 工具链，文档丰富 |
| 界面 | Gradio | 快速构建 Web UI，交互友好 |

## 运行指南

### CLI 命令行模式

```bash
python rag_system.py
```

支持三种问答模式：
- 模式 1: 仅 LLM（无检索）
- 模式 2: 仅检索（无生成）
- 模式 3: RAG（检索 + 生成）

### Web 界面

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`

## 项目结构

```
├── app.py               # Gradio Web 界面
├── rag_system.py         # RAG 核心系统
├── requirements.txt      # 依赖列表
├── .env.example          # API Key 配置模板
├── Materials/            # 知识库文档
│   ├── deepseek的技术报告.txt
│   ├── Python编程技巧.txt
│   └── 人工智能发展简史.txt
└── vector_db/            # ChromaDB 持久化数据
```
