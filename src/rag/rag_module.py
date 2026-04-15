"""RAG 探索模块：LangChain 实现文档检索增强问答

流程：加载文档 → 切分 → 向量化(Chroma) → 检索 top_k 片段 → 拼接上下文 → 调 LLM 回答

langchain 相关依赖延迟导入，保证本模块在未装 langchain 时仍可被 import
（RAG 是探索性功能，主评测流程不依赖它）。

注：langchain 1.x 已移除 VectorStoreIndexCreator，这里用显式的
loader→splitter→Chroma 流程，更清晰可控。
"""
import os


class RAGModule:
    def __init__(
        self,
        doc_path: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        self._llm_model = llm_model or os.getenv("DEFAULT_MODEL", "deepseek-ai/DeepSeek-V3")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.vectorstore = None
        if doc_path:
            self.load(doc_path)

    def load(self, doc_path: str) -> "RAGModule":
        """加载文档 → 切分 → 向量化，构建 Chroma 向量库"""
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        docs = TextLoader(doc_path, encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            chunks, OpenAIEmbeddings(model=self._embedding_model)
        )
        return self

    def query(self, question: str, top_k: int = 3) -> dict:
        """检索相关文档片段，拼成上下文让 LLM 回答"""
        if self.vectorstore is None:
            raise RuntimeError("未加载文档，请先调用 load(doc_path)")

        from langchain_openai import ChatOpenAI

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        llm = ChatOpenAI(model=self._llm_model, temperature=0)
        prompt = (
            "请根据以下资料回答问题。如果资料中没有答案，请说明。\n\n"
            f"资料：\n{context}\n\n问题：{question}"
        )
        answer = llm.invoke(prompt).content

        return {
            "question": question,
            "answer": answer,
            "sources": [d.page_content[:80] for d in docs],
        }
