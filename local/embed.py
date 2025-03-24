from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document
from google.cloud import aiplatform
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\GitHub\RAG-with-metadata-filtering\key\teak-kit-453818-e0-a28e80210053.json"
aiplatform.init(project="teak-kit-453818-e0", location="us-central1")

def populate_vector_db_secure(docs: list[Document],
                                embedding_model_name: str = "textembedding-gecko-multilingual@001",
                                collection_name: str = "demo_collection",
                                persist_path: str = r"F:\GitHub\RAG-with-metadata-filtering\vector-store"
                            ) -> QdrantVectorStore:

    embeddings = VertexAIEmbeddings(model=embedding_model_name)
    test_vector = embeddings.embed_query("This is a test.")
    vector_dim = len(test_vector)

    client = QdrantClient(path=persist_path)
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(collection_name=collection_name,
                                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE))

    vector_store = QdrantVectorStore(client=client,
                                    collection_name=collection_name,
                                    embedding=embeddings,
                                    retrieval_mode=RetrievalMode.DENSE)

    vector_store.add_documents(documents=docs)

    return vector_store


def display_search_similar_docs(vector_store: QdrantVectorStore, 
                                query: str, k: int = 2):
    results = vector_store.similarity_search(query, k=k)
    print(f"\n Top {k} results for: '{query}'\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:150]}...")
        print(f"   Metadata: {res.metadata}\n")

#Test the function
document_1 = Document(
    page_content="Sáng nay tôi đã ăn bánh pancake sô-cô-la chip và trứng bác cho bữa sáng.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="Dự báo thời tiết ngày mai sẽ có mây và u ám, nhiệt độ cao nhất khoảng 17 độ C.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Đang xây dựng một dự án mới đầy thú vị với LangChain - ghé xem thử nhé!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Những tên cướp đã đột nhập vào ngân hàng thành phố và lấy trộm 1 triệu đô la tiền mặt.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! Bộ phim đó thật tuyệt vời. Tôi không thể chờ để xem lại lần nữa.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Chiếc iPhone mới có đáng giá không? Đọc bài đánh giá này để biết thêm.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="Top 10 cầu thủ bóng đá xuất sắc nhất thế giới hiện nay.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph là framework tốt nhất để xây dựng ứng dụng có trạng thái và mang tính tác nhân!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="Thị trường chứng khoán giảm 500 điểm hôm nay do lo ngại về suy thoái kinh tế.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="Tôi có linh cảm xấu rằng tôi sắp bị xóa rồi :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
vector_store = populate_vector_db_secure(docs=documents)
display_search_similar_docs(vector_store, "lập trình viên", k=2)