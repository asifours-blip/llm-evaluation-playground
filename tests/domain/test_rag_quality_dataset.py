from pathlib import Path

from rag_quality_lab.config.loaders import load_dataset
from rag_quality_lab.domain.models import Answerability

DATASET_PATH = Path("data/eval/rag_quality_v1.json")
CORPUS_DIR = Path("data/knowledge_base")


def document_id(path: Path) -> str:
    return "-".join(path.stem.split("-")[:2])


def test_rag_quality_dataset_has_required_distribution() -> None:
    dataset = load_dataset(DATASET_PATH)
    answerable = [
        case for case in dataset.cases if case.answerability is Answerability.ANSWERABLE
    ]
    unanswerable = [
        case for case in dataset.cases if case.answerability is Answerability.UNANSWERABLE
    ]

    assert len(dataset.cases) == 48
    assert len(answerable) == 36
    assert len(unanswerable) == 12
    assert sum(len(case.expected_document_ids) == 1 for case in answerable) == 24
    assert sum(len(case.expected_document_ids) >= 2 for case in answerable) == 12
    assert sum("explicit_oos" in case.tags for case in unanswerable) == 6
    assert sum("plausible_unsupported" in case.tags for case in unanswerable) == 6
    assert len(list(CORPUS_DIR.glob("*.md"))) == 12


def test_all_expected_documents_and_evidence_exist() -> None:
    dataset = load_dataset(DATASET_PATH)
    documents = {
        document_id(path): path.read_text(encoding="utf-8-sig")
        for path in CORPUS_DIR.glob("*.md")
    }

    assert set(documents) == {f"doc-{index:02d}" for index in range(1, 13)}
    for case in dataset.cases:
        assert set(case.expected_document_ids) <= set(documents)
        expected_text = "\n".join(documents[doc_id] for doc_id in case.expected_document_ids)
        for evidence in case.reference_evidence:
            assert evidence in expected_text, case.id


def test_corpus_documents_have_required_sections() -> None:
    for path in CORPUS_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8-sig")
        assert f"ID: {document_id(path)}" in content
        assert "## Definition" in content
        assert "## Trade-offs" in content
        assert "## Example" in content
        assert "## Limitations" in content
