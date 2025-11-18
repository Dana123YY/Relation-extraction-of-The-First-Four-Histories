# Project Structure

## Recommended Directory Layout

```
├── historical-graphrag/
│   ├── LICENSE                    # License file
│   ├── .gitignore                # Git ignore rules
│   ├── requirements.txt          # Python dependencies
│   ├── config_example.yaml       # Example configuration
│   ├── config.yaml              # Your actual config (gitignored)
│   ├── example.py              # Usage example script
│
├── README.md                  # Project documentation
│
├── data/                    # Data directory
│   ├── entity_alias_dict.txt    # Entity alias dictionary
│   ├── train_data/              # Training data
│   ├── test_data/               # Test data
│   ├── score_data/              # Scoring dataset
│   └── KG_data/                 # Knowledge graph data
│
├── models/                  # Embedding models (gitignored)
│   └── paraphrase-multilingual-MiniLM-L12-v2/
```

## File Descriptions

### Core Files

- **graphrag.py**: Main GraphRAG system implementation
  - `HistoricalGraphRAG` class with all core functionality
  - Entity extraction, graph querying, semantic matching
  - LLM integration and answer generation

- **example.py**: Demonstration script showing basic usage

- **config.yaml**: Your personal configuration file
  - API keys, database credentials
  - Model paths, optional settings
  - **Important**: Keep this file secure and don't commit it

### Configuration

- **config_example.yaml**: Template configuration file
  - Safe to commit (no real credentials)
  - Shows all available options with examples

- **.gitignore**: Specifies files Git should ignore
  - Protects sensitive config files
  - Excludes large model files
  - Ignores Python cache and IDE files

### Documentation

- **README.md**: Comprehensive project documentation
  - Installation instructions
  - Usage examples
  - API reference
  - Dataset information

- **requirements.txt**: Python package dependencies
  - All required libraries with version constraints

## Setup Instructions

1. **Clone and install**:
   ```bash
   git clone <your-repo-url>
   cd historical-graphrag
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp config_example.yaml config.yaml
   # Edit config.yaml with your credentials
   ```

3. **Download models**:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; \
              SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').save('models/paraphrase-multilingual-MiniLM-L12-v2')"
   ```

4. **Prepare data**:
   - Place your entity alias dictionary in `data/entity_alias_dict.txt`

5. **Run**:
   ```bash
   python example.py
   ```

## Development Workflow

### Adding New Features

1. Modify `graphrag.py` with new methods
2. Update `example.py` with usage examples
3. Update `README.md` documentation
4. Add tests if applicable

## Notes

- Keep `config.yaml` out of version control
- Large files (models, data) should be hosted separately
- Use semantic versioning for releases
- Document all API changes in README.md
