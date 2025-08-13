# Tools Configuration

This directory contains the configuration and utilities for the RoboOS tools system.

## Configuration Files

### `config.yaml`
Contains all configuration parameters for the tools module in YAML format, including:

- **EMBEDDING_CONFIG**: Configuration for tool selection and embedding
- **TOOL_SELECTION_CONFIG**: General tool selection settings
- **CATEGORY_KEYWORDS**: Predefined tool categories and keywords
- **SYNONYM_MAPPINGS**: Synonym mappings for better semantic understanding
- **STOP_WORDS**: Stop words for keyword extraction

### Key Configuration Parameters

#### Embedding Configuration
- `method`: Embedding method ("tfidf", "enhanced_tfidf", "simple")
- `top_k`: Number of top tools to select
- `similarity_threshold`: Minimum similarity threshold (0.0 to 1.0)
- `debug`: Enable debug logging
- `use_category_filtering`: Enable category-based filtering
- `min_tools`: Minimum number of tools to return
- `max_tools`: Maximum number of tools to return
- `name_weight`: Weight for tool name matching
- `description_weight`: Weight for tool description matching
- `fuzzy_match`: Enable fuzzy matching
- `synonym_match`: Enable synonym matching

#### Tool Selection Configuration
- `default_tool_count`: Default number of tools to return
- `enable_caching`: Enable caching for tool embeddings
- `cache_expiration`: Cache expiration time in seconds
- `log_selection`: Log tool selection decisions

## Usage

### Importing Configuration
```python
from tools.embedding_utils import EMBEDDING_CONFIG, CATEGORY_KEYWORDS

# Access configuration values
method = EMBEDDING_CONFIG["method"]
top_k = EMBEDDING_CONFIG["top_k"]

# Or use the loader functions
from tools.embedding_utils import load_config
config = load_config()
embedding_config = config.get("embedding", {})
category_keywords = config.get("category_keywords", {})
```

### Using EnhancedEmbeddingManager
```python
from tools.embedding_utils import EnhancedEmbeddingManager

# Create manager with default config
manager = EnhancedEmbeddingManager()

# Or specify custom method
manager = EnhancedEmbeddingManager(method="enhanced_tfidf")
```

## Configuration Structure

The embedding configuration is now hardcoded in `tools/config.yaml` and loaded directly by the `embedding_utils.py` module. The configuration path is no longer configurable from the main config file.

## Benefits of New Structure

1. **Modularity**: Tool-specific configuration is now separate from main system config
2. **Maintainability**: Easier to manage and update tool-related settings
3. **Reusability**: Configuration can be imported by multiple tool modules
4. **Type Safety**: Python configuration provides better IDE support and validation
5. **Extensibility**: Easy to add new configuration parameters and categories

## Adding New Configuration

To add new configuration parameters:

1. Add them to the appropriate section in `config.yaml`
2. Update the `EnhancedEmbeddingManager.__init__()` method to load them
3. Use the new parameters in your code

Example:
```yaml
# In config.yaml
embedding:
  # ... existing config ...
  new_parameter: "new_value"
```

```python
# In embedding_utils.py
def __init__(self, method: str = None):
    # ... existing code ...
    self.new_parameter = EMBEDDING_CONFIG.get("new_parameter", "default_value")
```

## Configuration Loading

The configuration is automatically loaded when the module is imported. The `embedding_utils.py` module provides:

- `load_config()`: Load the full configuration
- `EMBEDDING_CONFIG`: Access embedding configuration section
- `TOOL_SELECTION_CONFIG`: Access tool selection configuration section
- `CATEGORY_KEYWORDS`: Access category keywords
- `SYNONYM_MAPPINGS`: Access synonym mappings
- `STOP_WORDS`: Access stop words

All configuration variables include fallback to default values if the YAML file is not found or has errors.
