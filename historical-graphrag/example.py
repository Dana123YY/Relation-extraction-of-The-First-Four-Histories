# -*- coding: utf-8 -*-
"""
Example usage of Historical Figure GraphRAG System
"""

import yaml
from graphrag import HistoricalGraphRAG


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize GraphRAG system
    print("Initializing GraphRAG system...")
    rag_system = HistoricalGraphRAG(config)
    
    # Example queries
    queries = [
        "曹操一生去过哪些地方？担任过什么官职？",
        "诸葛亮和谁有师从关系？",
        "刘备的祖先是谁？"
    ]
    
    print("\n" + "="*60)
    print("Historical Figure GraphRAG System - Example Usage")
    print("="*60 + "\n")
    
    for query in queries:
        print(f"\n【Query】{query}\n")
        
        # Get answer and graph data
        answer, graph_data = rag_system.query(query)
        
        print(f"【Answer】\n{answer}\n")
        print(f"【Graph Data】")
        print(f"  - Nodes: {len(graph_data['nodes'])}")
        print(f"  - Edges: {len(graph_data['edges'])}")
        print("-" * 60)
    
    # Close connections
    rag_system.close()
    print("\nGraphRAG system closed.")


if __name__ == '__main__':
    main()
