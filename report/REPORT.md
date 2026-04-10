# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Cao Chinh - 2A202600001
**Nhóm:** 12
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:*
High cosine similarity (gần 1.0) nghĩa là hai vector nhúng có hướng rất giống nhau trong không gian vector. Điều này cho thấy hai đoạn văn bản tương ứng có sự tương đồng lớn về ngữ nghĩa hoặc nội dung từ vựng.

**Ví dụ HIGH similarity:**
- Sentence A: "Bệnh mạch vành có thể gây đau thắt ngực."
- Sentence B: "Cơn đau thắt ngực là biểu hiện của bệnh mạch vành."
- Tại sao tương đồng: Cả hai câu đều nói về mối liên hệ nhân quả giữa bệnh mạch vành và triệu chứng đau thắt ngực.

**Ví dụ LOW similarity:**
- Sentence A: "Suy tim là tình trạng cơ tim yếu."
- Sentence B: "Lập trình viên thường ngồi nhiều."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn khác nhau (y học và nghề nghiệp), không có liên quan về mặt ý nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
Step size = chunk_size - overlap = 500 - 50 = 450.
$(N-1) \times 450 + 500 \ge 10000 \implies (N-1) \times 450 \ge 9500 \implies N-1 \ge 21.11$.
Vậy $N-1 = 22$, suy ra $N = 23$ chunks.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*
Nếu overlap tăng lên 100, step size giảm xuống còn 400, dẫn đến số lượng chunks tăng lên (25 chunks). Ta muốn overlap nhiều hơn để đảm bảo ranh giới cắt không làm mất ngữ cảnh quan trọng giữa các chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Cardiology (Tim mạch) - Dữ liệu từ Vinmec.

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*
Đây là domain có tính chuyên môn cao, dữ liệu có cấu trúc phân cấp (mục lục, đoạn văn, danh sách). Việc áp dụng các chiến lược chunking và retrieval trên dữ liệu y tế giúp đánh giá khả năng hiểu ngữ cảnh chuyên ngành của hệ thống RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 01_benh_mach_vanh.txt | Vinmec | 5391 | source, department: Cardiology |
| 2 | 02_tang_huyet_ap.txt | Vinmec | 5759 | source, department: Cardiology |
| 3 | 03_suy_tim.txt | Vinmec | 4854 | source, department: Cardiology |
| 4 | 06_nhoi_mau_co_tim.txt | Vinmec | 5109 | source, department: Cardiology |
| 5 | 10_tam_soat_tim_mach.txt | Vinmec | 7003 | source, department: Cardiology |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `01_benh_mach_vanh.txt` | Truy xuất nguồn gốc bài viết để kiểm chứng. |
| `department` | string | `Cardiology` | Lọc tài liệu theo chuyên khoa nhanh chóng. |
| `doc_id` | string | `01_benh_mach_vanh` | Hỗ trợ xóa hoặc cập nhật toàn bộ chunks của 1 tài liệu. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 01_benh_mach_vanh.txt | FixedSizeChunker (`fixed_size`) | 10 | 455.60 | No (Cắt ngang câu) |
| 01_benh_mach_vanh.txt | SentenceChunker (`by_sentences`) | 21 | 193.57 | Yes (Cắt theo câu) |
| 01_benh_mach_vanh.txt | RecursiveChunker (`recursive`) | 10 | 408.80 | Yes (Giữ cấu trúc đoạn) |

### Strategy Của Tôi

**Loại:** FixedSizeChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
Strategy này cắt văn bản thành các khối (chunks) có kích thước cố định (ví dụ: 500 ký tự) dựa theo chiều dài chuỗi. Nó không quan tâm đến ranh giới câu hay đoạn văn, mà chỉ đếm đủ số ký tự để tạo ra một chunk mới. Tùy chọn overlap cho phép các chunks liền kề chia sẻ một lượng ký tự nhất định ở phần nối tiếp để giảm khả năng một từ khóa quan trọng bị gãy đôi.

> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Với dữ liệu y tế đơn giản, FixedSize Chunking đảm bảo tất cả các chunk sinh ra đều tuân thủ độ dài tối đa của embedding model. Nó rất nhanh, ổn định và dễ cài đặt (không cần xử lý quy luật hay các ký hiệu phân tách phức tạp).

**Code snippet (nếu custom):**
```python
# Mặc định sử dụng FixedSizeChunker sẵn có của src.chunking
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 01_benh_mach_vanh.txt | best baseline (Recursive) | 10 | 408.80 | High |
| 01_benh_mach_vanh.txt | **của tôi** (FixedSize) | 10 | 455.60 | Medium |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Chinh (Tôi) | FixedSize | 5 | Đơn giản, nhanh | Hay bị cắt ngang thông tin |
| Tiến | Recursive | 8 | Giữ cấu trúc tốt | Cần tinh chỉnh dấu phân cách |
| Thành | Sentence | 7 | Tách câu tự nhiên | Bỏ sót context giữa các đoạn |
| Linh | Recursive | 8 | Hiệu quả với tài liệu dài | Phức tạp trong cài đặt |
| Ngân | Sentence | 7 | Chính xác về ngữ pháp | Chunk size không đều |
| Khôi | FixedSize | 4 | Dễ debug | Kém linh hoạt |


> *Viết 2-3 câu:*
RecursiveChunker là tốt nhất vì nó linh hoạt nhất, đảm bảo được độ dài chunk ổn định mà vẫn tôn trọng các cấu trúc ngữ pháp tự nhiên của văn bản chuyên ngành.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Sử dụng regex `(?<=[.!?])\s+|\n+` để phát hiện các dấu kết thúc câu (`.`, `!`, `?`) theo sau là khoảng trắng hoặc xuống dòng. Xử lý được trường hợp các câu không nối tiếp trên cùng một dòng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Thuật toán duyệt qua danh sách các separator. Nếu một đoạn văn bản (part) vẫn vượt quá `chunk_size`, hàm sẽ tự gọi lại chính nó với các separator mức thấp hơn. Base case là khi không còn separator nào hoặc độ dài văn bản đã đạt yêu cầu.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
Lưu trữ dưới dạng list các dictionaries (in-memory) hoặc ChromaDB collection. Tính similarity bằng dot product giữa vector query và vector chunk đã được chuẩn hóa (embedding).

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
Thực hiện pre-filtering (lọc trước) dựa trên metadata trước khi tính toán similarity để tối ưu hiệu năng. Delete bằng cách duyệt qua store và loại bỏ các chunk có `doc_id` tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
Prompt bao gồm Header "Context:", theo sau là nội dung các chunks được nối bằng `\n\n`, và cuối cùng là câu hỏi của người dùng. Context được tiêm trực tiếp vào string trước khi gửi đến LLM.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
...
============================== 42 passed in 0.93s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Bệnh mạch vành có thể gây đau thắt ngực. | Cơn đau thắt ngực là biểu hiện của bệnh mạch vành. | high | -0.1615 | Sai (Mock) |
| 2 | Tập thể dục tốt cho tim mạch. | Ăn nhiều rau xanh giúp giảm huyết áp. | medium | 0.1151 | Đúng (Randomly) |
| 3 | Suy tim là tình trạng cơ tim yếu. | Lập trình viên thường ngồi nhiều. | low | -0.1899 | Đúng |
| 4 | Hút thuốc lá làm tăng nguy cơ xơ vữa. | Tránh khói thuốc giúp bảo vệ thành mạch. | high | -0.0207 | Sai (Mock) |
| 5 | Can thiệp mạch vành bằng đặt stent. | Phẫu thuật bắc cầu là phương pháp hiện đại. | high | 0.1355 | Sai (Mock) |

> *Viết 2-3 câu:*
Kết quả từ MockEmbedder rất bất ngờ vì điểm số semantic gần như ngẫu nhiên. Điều này cho thấy embeddings cần một mô hình ngôn ngữ thực thụ (như MiniLM) để có thể hiểu được mối quan hệ logic giữa các từ ngữ thay vì chỉ băm (hash) chuỗi ký tự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh mạch vành là gì? | Là tình trạng động mạch vành bị hẹp/tắc do mảng bám chất béo/cholesterol. |
| 2 | Triệu chứng đau thắt ngực thế nào? | Cảm giác nặng, nén, ép tim, bỏng rát hoặc bóp chặt vùng ngực. |
| 3 | Yếu tố nguy cơ không thể thay đổi? | Tuổi tác, giới tính, tiền sử gia đình. |
| 4 | Phòng ngừa qua lối sống? | Bỏ thuốc, hạn chế rượu, ăn khoa học, tập thể dục thường xuyên. |
| 5 | Điều trị y học hiện đại? | Nong mạch vành, đặt stent, phẫu thuật bắc cầu. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bệnh mạch vành... | Tăng huyết áp: Nguyên nhân... | 0.0940 | No | Answered using context |
| 2 | Tại sao nam giới... | Tầm soát bệnh tim mạch... | 0.1323 | No | Answered using context |
| 3 | Triệu chứng đau... | Dinh dưỡng cho bệnh nhân... | 0.1397 | No | Answered using context |
| 4 | Phòng ngừa lối sống? | Nhồi máu cơ tim: Nhận biết... | 0.1334 | Yes | Answered using context |
| 5 | Điều trị hiện đại? | Nhồi máu cơ tim: Nhận biết... | 0.1449 | Yes | Answered using context |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5 (Dùng MockEmbedder)

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*
Cách các bạn tối ưu hóa metadata để lọc tài liệu theo chuyên khoa, giúp giảm tải cho việc tính toán similarity một cách đáng kể.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*
Việc sử dụng các mô hình embedding mạnh mẽ hơn (như OpenAI) giúp chất lượng retrieval tăng lên vượt trội so với các phương pháp cơ bản.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*
Tôi sẽ tăng cường làm sạch dữ liệu (data cleaning), loại bỏ các đoạn văn bản quảng cáo hoặc thông tin liên hệ không cần thiết để chunks tập trung hoàn toàn vào kiến thức y khoa.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **97 / 100** |
