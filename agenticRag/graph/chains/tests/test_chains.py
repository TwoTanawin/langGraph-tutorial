from pprint import pprint

from dotenv import load_dotenv

load_dotenv()
import sys
import os
# Add path to root: /Users/two-mac/Documents/LLM-Engineer/LangGraph/agenticRag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from graph.chains.retrieve_grader import generation_chain
from graph.chains.retrieve_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"