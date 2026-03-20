"""RAG 文档问答演示

用法（需先装 langchain 依赖 + 配置 .env）：
  python examples/rag_demo.py
  python examples/rag_demo.py --question "RAG 解决了什么问题？"
  python examples/rag_demo.py data/datasets/sample_doc.txt --question "向量数据库有哪些？"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.rag.rag_module import RAGModule


def parse_args(argv):
    doc = "data/datasets/sample_doc.txt"
    question = "什么是 RAG？它解决了大模型的什么问题？"
    positional = [a for a in argv[1:] if not a.startswith("--")]
    if positional:
        doc = positional[0]
    if "--question" in argv:
        i = argv.index("--question")
        if i + 1 < len(argv):
            question = argv[i + 1]
    return doc, question


def main():
    doc, question = parse_args(sys.argv)
    print(f"加载文档：{doc}")
    rag = RAGModule(doc_path=doc)
    print(f"问题：{question}\n")

    result = rag.query(question)
    print(f"回答：{result['answer']}\n")
    print("参考来源片段：")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src}...")


if __name__ == "__main__":
    main()
