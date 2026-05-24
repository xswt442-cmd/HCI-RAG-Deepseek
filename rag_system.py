"""
基于 DeepSeek 的 RAG 信息检索系统
支持文档向量化、检索增强生成、三种模式对比
"""
import os
from pathlib import Path

import hashlib
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 配置
BASE_DIR = Path(__file__).parent
MATERIALS_DIR = BASE_DIR / "Materials"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 3

# DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# Prompt 模板
RAG_PROMPT = """请根据以下【参考内容】回答用户的问题。如果参考内容中没有相关信息，请直接说"根据现有资料无法回答"，不要编造信息。

【参考内容】
{context}

用户问题：{question}

请给出准确、简洁的回答，并说明信息来源（例如"根据文档X……"）。"""

DIRECT_PROMPT = """请回答以下问题：

{question}

请给出准确、简洁的回答。"""


# RAG 系统核心类
class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        self.doc_names = []

    # 文档加载
    def load_documents(self):
        docs = []
        for filepath in MATERIALS_DIR.glob("*.txt"):
            text = filepath.read_text(encoding="utf-8")
            docs.append({"name": filepath.stem, "content": text})
        return docs

    # 多格式文件文本提取
    @staticmethod
    def extract_text(filepath: str) -> str:
        """从 PDF/DOCX/MD/TXT 文件中提取纯文本"""
        ext = Path(filepath).suffix.lower()
        if ext in (".txt", ".md"):
            return Path(filepath).read_text(encoding="utf-8")
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError(f"不支持的格式: {ext}。支持: PDF, DOCX, MD, TXT")

    # 增量添加文档到向量库
    def add_document(self, filepath: str) -> dict:
        """将单个文件切分向量化后增量加入向量库，返回入库信息"""
        text = self.extract_text(filepath)
        doc_name = Path(filepath).stem
        file_hash = hashlib.md5(text.encode()).hexdigest()[:8]

        # 去重: 先删旧同名文档再插入
        if self.collection:
            try:
                existing = self.collection.get(where={"source": doc_name})
                if existing["ids"]:
                    self.collection.delete(ids=existing["ids"])
            except Exception:
                pass

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        doc_chunks = text_splitter.split_text(text)

        embeddings = self.embedding_model.encode(doc_chunks, normalize_embeddings=True).tolist()
        ids = [f"{doc_name}_{file_hash}_{i}" for i in range(len(doc_chunks))]
        metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(doc_chunks))]

        if self.collection is None:
            self.collection = self.chroma_client.get_or_create_collection("rag_docs")
        self.collection.add(embeddings=embeddings, documents=doc_chunks, metadatas=metadatas, ids=ids)

        return {"name": doc_name, "chunks": len(doc_chunks), "chars": len(text)}

    # 文本切分 + 向量化 + 入库
    def build_knowledge_base(self, force_rebuild=False):
        collection_name = "rag_docs"
        try:
            self.chroma_client.delete_collection(collection_name) if force_rebuild else None
        except Exception:
            pass

        existing = [c.name for c in self.chroma_client.list_collections()]
        if not force_rebuild and collection_name in existing:
            self.collection = self.chroma_client.get_collection(collection_name)
            count = self.collection.count()
            if count > 0:
                print(f"向量库已存在 ({count} 个chunk)，跳过重建。")
                return

        docs = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        chunks, metadatas, ids = [], [], []
        for doc in docs:
            doc_chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadatas.append({"source": doc["name"], "chunk_id": i})
                ids.append(f"{doc['name']}_{i}")

        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True).tolist()

        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        self.collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        print(f"知识库构建完成: {len(chunks)} 个chunk 已入库。")

    # 检索
    def retrieve(self, question: str, k: int = TOP_K):
        if self.collection is None:
            self.collection = self.chroma_client.get_collection("rag_docs")
        q_embedding = self.embedding_model.encode([question], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=q_embedding, n_results=k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [[0]] * len(docs))[0]
        return docs, metas, distances

    # LLM 调用
    def ask_llm(self, prompt: str, system: str = None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.3, max_tokens=1024,
        )
        return resp.choices[0].message.content

    # 模式1: 仅LLM
    def mode_llm_only(self, question: str):
        return self.ask_llm(DIRECT_PROMPT.format(question=question))

    # 模式2: 仅检索
    def mode_retrieval_only(self, question: str):
        docs, metas, distances = self.retrieve(question)
        results = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            results.append({
                "rank": i + 1,
                "source": meta["source"],
                "content": doc[:300] + ("..." if len(doc) > 300 else ""),
                "similarity": f"{1 - dist:.4f}",
            })
        return results

    # 模式3: RAG 
    def mode_rag(self, question: str):
        docs, metas, _ = self.retrieve(question)
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            context_parts.append(f"【文档{i+1}: {meta['source']}】\n{doc}")
        context = "\n\n".join(context_parts)
        answer = self.ask_llm(RAG_PROMPT.format(context=context, question=question))
        sources = [{"rank": i + 1, "source": meta["source"], "content": doc[:200] + "..."}
                   for i, (doc, meta) in enumerate(zip(docs, metas))]
        return answer, sources


# CLI 操作
def run_cli():
    rag = RAGSystem()
    print("=" * 50)
    print("  基于 DeepSeek 的 RAG 信息检索系统")
    print("=" * 50)
    print("\n正在初始化知识库...")
    rag.build_knowledge_base()

    while True:
        print("\n" + "-" * 40)
        print("模式选择:")
        print("  1 - 仅 LLM (无检索)")
        print("  2 - 仅检索 (无生成)")
        print("  3 - RAG (检索 + 生成)")
        print("  q - 退出")
        choice = input("\n请选择模式: ").strip()

        if choice.lower() == "q":
            print("再见!")
            break
        if choice not in ("1", "2", "3"):
            print("无效选择，请重试。")
            continue

        question = input("请输入问题: ").strip()
        if not question:
            continue

        if choice == "1":
            print("\n[模式: 仅LLM]")
            answer = rag.mode_llm_only(question)
            print(f"\n答案:\n{answer}")

        elif choice == "2":
            print("\n[模式: 仅检索]")
            results = rag.mode_retrieval_only(question)
            for r in results:
                print(f"\n--- 文档{r['rank']} (来源: {r['source']}, 相似度: {r['similarity']}) ---")
                print(r["content"])

        elif choice == "3":
            print("\n[模式: RAG]")
            answer, sources = rag.mode_rag(question)
            print(f"\n===== 检索到的文档片段 =====")
            for s in sources:
                print(f"\n--- 来源: {s['source']} (第{s['rank']}名) ---")
                print(s["content"])
            print(f"\n===== 最终答案 =====")
            print(answer)


if __name__ == "__main__":
    run_cli()
