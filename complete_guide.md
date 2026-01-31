#  Complete Guide - Multimodal PDF Extractor with MongoDB

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Step-by-Step Usage](#step-by-step-usage)
5. [Full Pipeline](#full-pipeline)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 System Requirements

### Required Software

- **Python**: 3.8+
- **MongoDB**: 4.4+ (Community Edition)
- **GPU**: Optional (CUDA for faster embedding, CPU works fine)

### Required API Keys

- **Gemini API Key**: Get from https://aistudio.google.com/app/apikey

---

## 📦 Installation

### 1. Install Python Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# If you have GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# If CPU only
pip install torch torchvision
```

### 2. Install MongoDB

#### Windows:
```bash
# Download from: https://www.mongodb.com/try/download/community
# Or use installer
choco install mongodb

# Start MongoDB service
net start MongoDB
```

#### Linux (Ubuntu):
```bash
# Import public key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Add repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Install
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start service
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### Mac:
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start service
brew services start mongodb-community
```

### 3. Verify MongoDB Installation

```bash
# Check if MongoDB is running
mongo --eval "db.version()"

# Or
mongosh --eval "db.version()"
```

---

## ⚙️ Configuration

### 1. Create `.env` File

```bash
# Copy template
cp .env.template .env

# Edit .env
GEMINI_API_KEY=your_gemini_api_key_here
MONGO_URI=mongodb://localhost:27017
GEMINI_MODEL=gemini-2.0-flash-exp
```

### 2. Verify Configuration

```bash
# Test Gemini API
python -c "from analysis.config import AnalysisConfig; AnalysisConfig.validate()"

# Test MongoDB connection
python -c "from storage import MongoDBHandler; MongoDBHandler().print_statistics()"

# Test CLIP embedding
python -c "from embedding import CLIPEmbedder; CLIPEmbedder()"
```

---

## 🎯 Step-by-Step Usage

### STEP 0: Extract PDF (You already have this)

Run your existing extraction pipeline to get:
```
extracted_content/
├── figures/
│   ├── page1_fig1.png
│   └── ...
├── tables/
│   ├── page3_table1.json
│   └── ...
└── metadata.json
```

---

### STEP 1: AI Analysis (Gemini Vision)


```

**Output**: `extracted_content/paper_name/analysis_results.json`

**What it contains**:
```json
[
  {
    "path": "extracted_content/figures/page1_fig1.png",
    "caption": "Figure 1: Architecture",
    "type": "figure",
    "summary": "Shows a neural network with transformer layers...",
    "status": "success"
  },
  ...
]
```

---

### STEP 2: Generate Embeddings (CLIP)



**Output**: `extracted_content/paper_name/processed_with_embeddings.json`

**What it contains**:
```json
{
  "pdf_file": "sample_paper.pdf",
  "figures": [
    {
      "path": "...",
      "caption": "...",
      "analysis": { "summary": "..." },
      "embeddings": {
        "text_embedding": [0.123, -0.456, ...],  // 512-dim
        "image_embedding": [0.789, -0.234, ...], // 512-dim
        "similarity": 0.87,
        "model": "clip-vit-base-patch32"
      }
    }
  ],
  "tables": [...]
}
```

---

### STEP 3: Store in MongoDB


**What gets stored**:

**Papers Collection**:
```javascript
{
  _id: ObjectId("..."),
  pdf_name: "sample_paper.pdf",
  total_pages: 12,
  figure_count: 5,
  table_count: 3,
  extraction_date: ISODate("2025-10-14T10:30:00Z"),
  metadata: {
    has_analysis: true,
    has_embeddings: true,
    processing_complete: true
  }
}
```

**Figures Collection**:
```javascript
{
  _id: ObjectId("..."),
  paper_id: "...",
  type: "figure",
  page: 3,
  caption: "Figure 1: CLIP architecture",
  file_path: "extracted_content/figures/page3_fig1.png",
  analysis: {
    summary: "Shows dual-encoder architecture...",
    model: "gemini-2.0-flash-exp",
    status: "success"
  },
  embeddings: {
    text_embedding: [0.123, ...],  // 512-dim
    image_embedding: [0.789, ...], // 512-dim
    similarity: 0.87,
    embedding_model: "clip-vit-base-patch32"
  }
}
```

**Tables Collection**: (Similar structure, text embedding only)

---

## 🚀 Full Pipeline (All-in-One)

### Option 1: Run Complete Pipeline

```bash
python main_pipeline.py -all
```

This runs **all steps automatically**:
1. ✅ AI Analysis (Gemini)
2. ✅ Embedding Generation (CLIP)
3. ✅ MongoDB Storage

### Option 2: Run 1 sample Pipeline
```bash
python main_pipeline.py extracted_content/sample_A/metadata.json

```
### Option 3: Cac ví du khac

```bash
# Xử lý tất cả, skip analysis (dùng kết quả cũ)
python main_pipeline.py --all --skip-analysis

# Xử lý sample B, không lưu vào MongoDB
python main_pipeline.py extracted_content/sample_B/metadata.json --skip-storage

# Xử lý tất cả với custom MongoDB URI
python main_pipeline.py --all --mongodb-uri "mongodb://localhost:27017"


---

## 📊 Query & Retrieve from MongoDB

### Python API Examples

#### 1. Text Search by Caption

```python
from storage import MongoDBHandler

handler = MongoDBHandler()

# Search figures/tables by caption
results = handler.search_by_caption("architecture")

for result in results:
    print(f"{result['type']}: {result['caption']}")
    print(f"Page: {result['page']}")
    print(f"Summary: {result['analysis']['summary'][:100]}...")
```

#### 2. Vector Similarity Search

```python
from storage import MongoDBHandler
from embedding import CLIPEmbedder

handler = MongoDBHandler()
embedder = CLIPEmbedder()

# Query with text
query_text = "transformer architecture diagram"
query_embedding = embedder.embed_text(query_text)[0].tolist()

# Find similar figures
results = handler.vector_search(
    query_embedding=query_embedding,
    element_type='figure',
    top_k=5
)

for result in results:
    print(f"Similarity: {result.get('similarity', 0):.4f}")
    print(f"Caption: {result['caption']}")
```

#### 3. Get All Data for a Paper

```python
from storage import MongoDBHandler

handler = MongoDBHandler()

# Get paper
paper = handler.get_paper_by_name("sample_paper.pdf")
paper_id = str(paper['_id'])

# Get all figures
figures = handler.get_figures_by_paper(paper_id)
print(f"Found {len(figures)} figures")

# Get all tables
tables = handler.get_tables_by_paper(paper_id)
print(f"Found {len(tables)} tables")
```

---

## 🐛 Troubleshooting

### Problem 1: Unicode Encoding Error (Windows)

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**: Already fixed in code with `utils/print_helper.py`

If still occurs, run:
```bash
set PYTHONIOENCODING=utf-8
python your_script.py
```

---

### Problem 2: MongoDB Connection Failed

**Error**: `Failed to connect to MongoDB`

**Check if MongoDB is running**:
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl status mongod
```

**Check connection string**:
```bash
# Test connection
mongosh mongodb://localhost:27017

# Or
mongo mongodb://localhost:27017
```

---

### Problem 3: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution 1**: Use CPU instead
```python
# In embedding/embedding_config.py
DEVICE = "cpu"  # Force CPU
```

**Solution 2**: Reduce batch size
```python
# In embedding/embedding_config.py
BATCH_SIZE = 4  # Reduce from 16
```

---

### Problem 4: Gemini API Rate Limit

**Error**: `Rate limit exceeded`

**Solution**: Already handled with automatic retry and backoff

To reduce rate:
```python
# In analysis/config.py
RATE_LIMIT_RPM = 10  # Reduce from 15
BATCH_SIZE = 5       # Process fewer items at once
```

---

### Problem 5: No Figures/Tables Found

**Check extraction output**:
```bash
ls extracted_content/figures/
ls extracted_content/tables/
cat extracted_content/metadata.json
```

If empty, re-run your detection/extraction pipeline first.

---

### Problem 6: CLIP Model Download Slow

**Solution**: Models download on first run (~1GB)

Download location: `cache/models/`

To use cached models:
```python
# Already configured in embedding/embedding_config.py
CACHE_DIR = "cache/models"
```

---

## 📈 Performance Tips

### 1. GPU Acceleration

If you have NVIDIA GPU:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Speed improvement**: 5-10x faster embeddings

---

### 2. Batch Processing

Process multiple PDFs efficiently:
```python
import asyncio
from main_pipeline import MultimodalPipeline

async def process_multiple():
    pipeline = MultimodalPipeline()
    
    pdf_metadatas = [
        "paper1/metadata.json",
        "paper2/metadata.json",
        "paper3/metadata.json"
    ]
    
    for metadata_path in pdf_metadatas:
        await pipeline.process_pdf(metadata_path)
    
    pipeline.close()

asyncio.run(process_multiple())
```

---

### 3. MongoDB Indexing

For large datasets, create vector search index (MongoDB Atlas only):

```javascript
// In MongoDB Atlas UI or mongosh
db.figures.createSearchIndex(
  "figure_vector_index",
  {
    mappings: {
      dynamic: false,
      fields: {
        "embeddings.image_embedding": {
          type: "knnVector",
          dimensions: 512,
          similarity: "cosine"
        }
      }
    }
  }
);
```

---

## 🎓 Usage Examples

### Example 1: Process Research Paper

```bash
# 1. Extract PDF (your existing pipeline)
python your_extraction_script.py paper.pdf

# 2. Run full pipeline
python main_pipeline.py extracted_content/metadata.json

# 3. Query results
python -c "
from storage import MongoDBHandler
handler = MongoDBHandler()
handler.print_statistics()
"
```

---

### Example 2: Find Similar Figures

```python
from storage import MongoDBHandler
from embedding import CLIPEmbedder

# Initialize
handler = MongoDBHandler()
embedder = CLIPEmbedder()

# Query
query = "attention mechanism diagram"
query_vec = embedder.embed_text(query)[0].tolist()

# Search
results = handler.vector_search(
    query_embedding=query_vec,
    element_type='figure',
    top_k=3
)

# Display
for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Similarity: {result.get('similarity', 0):.4f}")
    print(f"Caption: {result['caption']}")
    print(f"File: {result['file_path']}")
```

---

### Example 3: Export Data

```python
from storage import MongoDBHandler
import json

handler = MongoDBHandler()

# Get all figures with embeddings
figures = list(handler.figures.find(
    {"embeddings": {"$exists": True}}
))

# Export to JSON
with open("all_figures_export.json", "w") as f:
    json.dump(figures, f, indent=2, default=str)

print(f"Exported {len(figures)} figures")
```

---

##  Support

### Check Logs

All modules print detailed logs:
- `[OK]` - Success
- `[INFO]` - Information
- `[WARNING]` - Warning (non-critical)
- `[ERROR]` - Error

### Common Commands

```bash
# Check system status
python -c "from embedding import EmbeddingConfig; EmbeddingConfig.validate()"
python -c "from storage import MongoDBHandler; MongoDBHandler().print_statistics()"

# Clear database (CAUTION!)
python -c "from storage import MongoDBHandler; MongoDBHandler().clear_database()"

# View MongoDB data
mongosh
> use multimodal_rnd
> db.papers.find()
> db.figures.countDocuments()
```

---

##  Summary

### Complete Workflow

```
1. Extract PDF → figures/, tables/, metadata.json
2. AI Analysis → analysis_results.json
3. Generate Embeddings → processed_with_embeddings.json
4. Store in MongoDB → Query & Retrieve!
```

### One Command to Rule Them All

```bash
python main_pipeline.py --all
```


Your multimodal data is now in MongoDB with:
- ✅ AI-generated summaries
- ✅ 512-dim embeddings (text + image)
- ✅ Full-text search capability
- ✅ Vector similarity search
- ✅ Structured metadata

