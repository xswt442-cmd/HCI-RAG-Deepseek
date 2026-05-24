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

### 向量库构建

首次运行 `python app.py` 或 `python rag_system.py` 时自动构建到 `vector_db/`，后续重启检测到已有数据自动跳过。如需重建，删除 `vector_db/` 目录即可。

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

```markdown
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

## 测试截图

### LLM vs 仅检索 vs RAG

直接启动 `app.py` 切换不同模式进行提问测试。

#### 问题一 Deepseek 相关

- **LLM：**![image-20260524174114249](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524174114249.png)

- **仅检索：**![image-20260524174320780](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524174320780.png)

- **RAG：**![image-20260524174839724](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524174839724.png)

#### 问题二 Python相关

- **LLM：**![image-20260524175018874](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175018874.png)

- **仅检索：**![image-20260524175050837](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175050837.png)

- **RAG：**![image-20260524175112790](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175112790.png)

#### 问题三 人工智能发展相关

- **LLM：**![image-20260524175208479](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175208479.png)

- **仅检索：**![image-20260524175245631](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175245631.png)

- **RAG：**![image-20260524175305342](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524175305342.png)

### 多格式文档

所用测试文档均在 `Testdocs/` 中，所问问题相关信息由其中的文档给出，提问结果如下：

![image-20260524182225892](E:\.codes\hcil3\HCI-RAG-Deepseek\assets\image-20260524182225892.png)
