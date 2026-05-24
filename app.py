"""
RAG 系统 Gradio Web 界面 — 支持多格式文档上传
"""
import os
import tempfile
from pathlib import Path

import gradio as gr
from rag_system import RAGSystem

rag = RAGSystem()

print("正在初始化知识库...")
rag.build_knowledge_base()
print("初始化完成！")


def answer_question(question, mode):
    if not question.strip():
        return "", "", "请输入问题。"

    if mode == "仅 LLM (无检索)":
        answer = rag.mode_llm_only(question)
        return "", answer

    elif mode == "仅检索 (无生成)":
        results = rag.mode_retrieval_only(question)
        sources_text = ""
        for r in results:
            sources_text += f"### 文档{r['rank']} — {r['source']} (相似度: {r['similarity']})\n\n"
            sources_text += f"{r['content']}\n\n---\n\n"
        return sources_text, "(仅展示检索结果，未经过 LLM 生成)"

    elif mode == "RAG (检索 + 生成)":
        answer, sources = rag.mode_rag(question)
        sources_text = ""
        for s in sources:
            sources_text += f"### 来源 {s['rank']}: {s['source']}\n\n"
            sources_text += f"{s['content']}\n\n---\n\n"
        return sources_text, answer


def upload_file(file):
    """处理上传文件: 提取文本、向量化、增量入库"""
    if file is None:
        return "未选择文件。"

    filepath = Path(file.name) if hasattr(file, "name") else None
    if filepath is None:
        return "文件路径无效。"

    ext = filepath.suffix.lower()
    allowed = {".pdf", ".docx", ".md", ".txt"}
    if ext not in allowed:
        return f"不支持 .{ext} 格式，仅支持 PDF, DOCX, MD, TXT。"

    try:
        info = rag.add_document(str(filepath))
        total = rag.collection.count()
        return (
            f"上传成功！  **{info['name']}** 已入库\n\n"
            f"- 文件大小: {info['chars']} 字符\n"
            f"- 切分 chunk: {info['chunks']} 个\n"
            f"- 向量库总计: {total} 个 chunk"
        )
    except Exception as e:
        return f"上传失败: {str(e)}"


def get_kb_stats():
    """获取知识库统计信息"""
    try:
        if rag.collection:
            count = rag.collection.count()
            # 获取所有唯一来源
            all_data = rag.collection.get()
            sources = set(m["source"] for m in all_data["metadatas"]) if all_data["metadatas"] else set()
            return f"知识库已就绪: {count} 个 chunk, 覆盖 {len(sources)} 篇文档"
    except Exception:
        pass
    return "知识库未初始化"


with gr.Blocks(title="RAG 信息检索系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 基于 DeepSeek 的 RAG 信息检索系统
    支持 PDF、DOCX、MD、TXT 多格式文档动态上传
    """)

    with gr.Row():
        # 左侧: 文档上传区域
        with gr.Column(scale=1):
            gr.Markdown("## 上传新文档")
            file_input = gr.File(
                label="选择文档 (PDF/DOCX/MD/TXT)",
                file_types=[".pdf", ".docx", ".md", ".txt"],
            )
            upload_btn = gr.Button("上传并入库", variant="secondary")
            upload_status = gr.Markdown(value=get_kb_stats())

        # 右侧: 问答区域
        with gr.Column(scale=2):
            gr.Markdown("## 提问")
            question_input = gr.Textbox(
                label="输入你的问题",
                placeholder="例如: DeepSeek 的上下文窗口是多少？",
                lines=2,
            )
            mode_radio = gr.Radio(
                choices=["仅 LLM (无检索)", "仅检索 (无生成)", "RAG (检索 + 生成)"],
                value="RAG (检索 + 生成)",
                label="问答模式",
            )
            submit_btn = gr.Button("提问", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="检索到的文档片段", value="*等待提问...*")
        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="最终答案", value="*等待提问...*")

    examples = gr.Examples(
        examples=[
            ["DeepSeek 的上下文窗口是多少？", "RAG (检索 + 生成)"],
            ["Python 中如何使用装饰器？", "RAG (检索 + 生成)"],
            ["AlexNet 是什么时候提出的？有什么意义？", "RAG (检索 + 生成)"],
            ["如何优化 Python 循环的性能？", "仅 LLM (无检索)"],
        ],
        inputs=[question_input, mode_radio],
    )

    upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_status])
    submit_btn.click(fn=answer_question, inputs=[question_input, mode_radio],
                     outputs=[sources_output, answer_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
