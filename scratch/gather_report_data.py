import sys
import os
from pathlib import Path

# Ensure src is importable
sys.path.append(os.getcwd())

from src.chunking import ChunkingStrategyComparator, compute_similarity
from src.store import EmbeddingStore
from src.embeddings import _mock_embed
from src.models import Document
from src.agent import KnowledgeBaseAgent

def gather_chunking_stats():
    print("--- CHUNKING STATS ---")
    comparator = ChunkingStrategyComparator()
    # Use the first cardiology file
    file_path = "data-tim-mach/01_benh_mach_vanh.txt"
    with open(file_path, "r") as f:
        text = f.read()
    
    results = comparator.compare(text, chunk_size=500)
    for name, data in results.items():
        print(f"Strategy: {name}")
        print(f"  Count: {data['count']}")
        print(f"  Avg Length: {data['avg_length']:.2f}")
    print()

def gather_similarity_scores():
    print("--- SIMILARITY SCORES ---")
    pairs = [
        ("Bệnh mạch vành có thể gây đau thắt ngực.", "Cơn đau thắt ngực là biểu hiện của bệnh mạch vành."),
        ("Tập thể dục tốt cho tim mạch.", "Ăn nhiều rau xanh giúp giảm huyết áp."),
        ("Suy tim là tình trạng cơ tim yếu.", "Lập trình viên thường ngồi nhiều."),
        ("Hút thuốc lá làm tăng nguy cơ xơ vữa.", "Tránh khói thuốc giúp bảo vệ thành mạch."),
        ("Can thiệp mạch vành bằng đặt stent.", "Phẫu thuật bắc cầu là phương pháp hiện đại."),
    ]
    for a, b in pairs:
        vec_a = _mock_embed(a)
        vec_b = _mock_embed(b)
        score = compute_similarity(vec_a, vec_b)
        print(f"A: {a}")
        print(f"B: {b}")
        print(f"Score: {score:.4f}")
    print()

def demo_llm(prompt: str) -> str:
    # The prompt starts with "Context:\n" and then chunks joined by "\n\n"
    # Then "\n\nQuestion:"
    if "Context:\n" in prompt and "\n\nQuestion:" in prompt:
        context_part = prompt.split("Context:\n")[1].split("\n\nQuestion:")[0]
        # In this specific implementation, chunks are just joined by \n\n
        # Since I used top_k=1 in the benchmark, it should be 1.
        # But let's count \n\n if we used more.
        if not context_part.strip():
            count = 0
        else:
            # This is a bit tricky to count exactly without markers, 
            # but since we know top_k=1, we can just say "Answered using context."
            count = 1 
        return f"[DEMO LLM] Answered using context (top_k=1)."
    return "[DEMO LLM] No context found."

def run_benchmarks():
    print("--- BENCHMARK QUERIES ---")
    store = EmbeddingStore(embedding_fn=_mock_embed)
    
    # Load all cardiology docs
    data_dir = Path("data-tim-mach")
    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        content = p.read_text(encoding="utf-8")
        docs.append(Document(id=p.stem, content=content, metadata={"source": str(p)}))
    
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    
    queries = [
        "Bệnh mạch vành là gì?",
        "Tại sao nam giới có nguy cơ mắc bệnh tim mạch cao hơn?",
        "Các triệu chứng đau thắt ngực được mô tả như thế nào?",
        "Làm thế nào để phòng ngừa bệnh mạch vành qua lối sống?",
        "Những phương pháp điều trị y học hiện đại cho bệnh mạch vành?",
    ]
    
    for q in queries:
        results = store.search(q, top_k=1)
        top_chunk = results[0] if results else None
        answer = agent.answer(q, top_k=1)
        
        print(f"Query: {q}")
        if top_chunk:
            print(f"  Top-1 Score: {top_chunk['score']:.4f}")
            print(f"  Top-1 Chunk: {top_chunk['content'][:150].replace(chr(10), ' ')}...")
        print(f"  Agent: {answer}")
    print()

if __name__ == "__main__":
    gather_chunking_stats()
    gather_similarity_scores()
    run_benchmarks()
