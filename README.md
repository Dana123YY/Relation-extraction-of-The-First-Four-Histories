# Historical Figure GraphRAG System

A Graph-Retrieval Augmented Generation (GraphRAG) system for knowledge extraction and question answering on historical Chinese texts, specifically designed for China's "First Four Histories" (前四史).

This project is part of the research paper: *"Research on Graph-Retrieval Augmented Generation Based on Historical Text Knowledge Graphs"*.

Note: The pre-trained model weights (pytorch_model.bin) are excluded from this repository due to GitHub's file size limits. Please download the paraphrase-multilingual-MiniLM-L12-v2 model (e.g., from Hugging Face) and ensure the pytorch_model.bin file is placed in the following directory: models/paraphrase-multilingual-MiniLM-L12-v2/0_Transformer/

## 🌟 Features

- **Entity Recognition**: Automatic extraction of historical figures and entities from classical Chinese texts
- **Knowledge Graph Integration**: Leverages Neo4j graph database for relationship storage and retrieval
- **Semantic Matching**: Uses sentence transformers for intelligent triple ranking
- **Alias Expansion**: Handles alternative names and titles for historical figures
- **Multi-step Reasoning**: Combines graph retrieval with LLM reasoning for comprehensive answers
- **Traditional Chinese Support**: Full support for Traditional Chinese text processing

## 📋 Prerequisites

- Python 3.8+
- Neo4j Database (version 5.0+)
- OpenAI-compatible API (DeepSeek, OpenAI, etc.)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/historical-graphrag.git
cd historical-graphrag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download embedding model**

Download the multilingual sentence transformer model:
```bash
# Create models directory
mkdir -p models

# Download from Hugging Face
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').save('models/paraphrase-multilingual-MiniLM-L12-v2')"
```

4. **Set up Neo4j Database**

- Install Neo4j Desktop or Neo4j Server
- Create a database named `shijigraphy` (or your preferred name)
- Import the knowledge graph data (see [Dataset](#-dataset) section)

## ⚙️ Configuration

Create or modify `config.yaml` with your settings:

```yaml
# LLM API Configuration
llm_api_key: "your-api-key-here"
llm_base_url: "https://api.deepseek.com"
llm_model: "deepseek-chat"

# Neo4j Database Configuration
neo4j_uri: "bolt://localhost:7687"
neo4j_user: "neo4j"
neo4j_password: "your-password-here"
neo4j_database: "shijigraphy"

# Embedding Model Configuration
embedding_model_path: "./models/paraphrase-multilingual-MiniLM-L12-v2"

# Optional: Entity Alias Dictionary
alias_dict_path: "./data/entity_alias_dict.txt"

# Optional: Proxy Configuration (if needed)
proxy: ""
```

### Configuration Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `llm_api_key` | API key for LLM service | Yes |
| `llm_base_url` | Base URL for LLM API | Yes |
| `llm_model` | Model name (e.g., "deepseek-chat") | Yes |
| `neo4j_uri` | Neo4j connection URI | Yes |
| `neo4j_user` | Neo4j username | Yes |
| `neo4j_password` | Neo4j password | Yes |
| `neo4j_database` | Database name | No (default: "neo4j") |
| `embedding_model_path` | Path to embedding model | Yes |
| `alias_dict_path` | Path to entity alias dictionary | No |
| `proxy` | HTTP/HTTPS proxy | No |

## 📖 Usage

### Basic Usage

```python
import yaml
from graphrag import HistoricalGraphRAG

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize system
rag_system = HistoricalGraphRAG(config)

# Query the system
query = "曹操一生去过哪些地方？担任过什么官职？"
answer, graph_data = rag_system.query(query)

print(f"Answer: {answer}")
print(f"Graph nodes: {len(graph_data['nodes'])}")
print(f"Graph edges: {len(graph_data['edges'])}")

# Close connections
rag_system.close()
```

### Running the Example

```bash
python example.py
```

### Advanced Usage: Custom Pipeline

```python
# Step-by-step usage
entities, categories = rag_system.extract_entities("诸葛亮担任过什么官职？")
expanded_entities, alias_str = rag_system.expand_with_aliases(entities)
triples = rag_system.query_graph("诸葛亮担任过什么官职？", alias_str)
ranked_triples = rag_system.semantic_match("诸葛亮担任过什么官职？", triples)
context = rag_system.triples_to_text(ranked_triples)
answer = rag_system.generate_response("诸葛亮担任过什么官职？", context)
```

## 📊 Dataset

The project includes the following datasets:

### 1. Training and Test Data
- `train_data/`: Reasoning chains for extracting character relationships
- `test_data/`: Test cases for evaluation

### 2. Scoring Data
- `score_data/`: Step-by-step scoring dataset for reasoning chains

### 3. Knowledge Graph Data
- `KG_data/`: Automatically extracted knowledge graph from the "First Four Histories"

### 4. Entity Alias Dictionary
- `data/entity_alias_dict.txt`: Mapping of historical figures to their alternative names

**Format**: `MainName:Alias1,Alias2,Alias3`

**Example**:
```
諸葛亮:孔明,臥龍
曹操:孟德,阿瞞
```

## 🏗️ System Architecture

```
User Query
    ↓
Entity Recognition (LLM)
    ↓
Alias Expansion (Dictionary)
    ↓
Graph Query (Neo4j + Cypher)
    ↓
Semantic Ranking (Sentence Transformers)
    ↓
Context Generation
    ↓
Answer Generation (LLM)
    ↓
Final Answer + Graph Visualization
```

## 🔧 API Reference

### HistoricalGraphRAG Class

#### Methods

- `__init__(config)`: Initialize the system with configuration
- `query(text)`: Main interface - returns answer and graph data
- `extract_entities(text)`: Extract entities from text
- `expand_with_aliases(entities)`: Expand entities with known aliases
- `query_graph(text, alias_str)`: Query Neo4j graph database
- `semantic_match(query, triples, top_k)`: Rank triples by semantic similarity
- `triples_to_text(ranked_triples)`: Convert triples to natural language
- `generate_response(query, context)`: Generate final answer
- `close()`: Close database connections

## 📝 Graph Schema

### Node Types
- 人物 (Person)
- 社會集團 (Social Group)
- 地點 (Location)
- 時間 (Time)
- 職官 (Official Position)
- 爵位 (Noble Title)
- 文獻 (Document)

### Relationship Types
- 别称, 父母, 子女, 同胞, 祖, 孙, 同族, 婚恋
- 上级, 下级, 更代, 承继, 师从, 养亲
- 所属团体, 籍贯, 居, 辖, 逃奔, 卒于, 访, 占, 攻
- 任官, 出生时间, 死亡时间, 封爵
- 积极态度倾向, 消极态度倾向

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{your-paper-2024,
  title={Research on Graph-Retrieval Augmented Generation Based on Historical Text Knowledge Graphs},
  author={Yang Feng},
  year={2024},
  journal={Your Journal},
}
```

## 📄 License

This project is released for academic research purposes only.

**Usage Restrictions:**
- ❌ Commercial use is prohibited
- ✅ Academic research and non-commercial use allowed with proper citation

## 📧 Contact

For questions, discussions, or collaborations:

**Yang Fan **  
Email: yangf@stu.njau.edu.cn

## 🙏 Acknowledgments

- Neo4j for the graph database platform
- Sentence Transformers for multilingual embeddings
- LangChain for the LLM integration framework
- OpenCC for Chinese text conversion

## 📚 Related Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)

---

**Note**: This system is designed for research purposes on historical Chinese texts. Ensure you have proper Neo4j setup and API access before running the system.
