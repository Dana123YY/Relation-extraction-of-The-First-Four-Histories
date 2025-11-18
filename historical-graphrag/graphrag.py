# -*- coding: utf-8 -*-
"""
Historical Figure GraphRAG System
Main module for graph-retrieval augmented generation on historical texts
"""

import os
import re
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from opencc import OpenCC
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from openai import OpenAI


class HistoricalGraphRAG:
    """GraphRAG system for historical figure knowledge extraction and Q&A"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphRAG system
        
        Args:
            config: Configuration dictionary containing API keys, database info, etc.
        """
        self.config = config
        
        # Initialize OpenCC for Traditional/Simplified Chinese conversion
        self.cc = OpenCC('s2t')
        
        # Set proxy if configured
        if config.get('proxy'):
            os.environ["http_proxy"] = config['proxy']
            os.environ["https_proxy"] = config['proxy']
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config['embedding_model_path'])
        
        # Initialize Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            config['neo4j_uri'],
            auth=(config['neo4j_user'], config['neo4j_password'])
        )
        
        # Initialize Neo4j graph for LangChain
        self.graph = Neo4jGraph(
            url=config['neo4j_uri'],
            username=config['neo4j_user'],
            password=config['neo4j_password'],
            database=config.get('neo4j_database', 'neo4j')
        )
        
        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=config['llm_api_key'],
            base_url=config['llm_base_url']
        )
        
        # Load entity alias dictionary if provided
        self.entity_aliases = {}
        if config.get('alias_dict_path'):
            self._load_alias_dict(config['alias_dict_path'])
    
    def _load_alias_dict(self, path: str):
        """Load entity alias dictionary from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        main_name = parts[0].replace('(', '').replace(')', '')
                        aliases = parts[1].split(',')
                        self.entity_aliases[main_name] = aliases
        except Exception as e:
            print(f"Warning: Could not load alias dictionary: {e}")
    
    def _call_llm(self, prompt: str, max_attempts: int = 5) -> str:
        """
        Call LLM API with retry logic
        
        Args:
            prompt: Input prompt
            max_attempts: Maximum retry attempts
            
        Returns:
            LLM response text
        """
        for attempt in range(max_attempts):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.config['llm_model'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed after {max_attempts} attempts: {e}")
                    return ""
    
    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract named entities from text using LLM
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entity_list, category_list)
        """
        text = self.cc.convert(text)
        
        instruction = """假设你是古文专家，现在根据提供的文本抽取出人物名称实体，对抽取的实体按照(实体;实体类型)的格式进行抽取，如果只有人物实体类型没有具体实体，则按照(无;实体类型)的格式进行抽取。（要求抽取实体和实体类型尽可能精简，尽可能不抽取'无'实体）
比如：1.{输入：诸葛亮担任过什么官职。}
{回答：抽取实体：(诸葛亮;人物)}
2.{输入：拜韓信為相國，收趙兵未發者擊齊。}
{回答：抽取实体：(韓信;人物)}
以下为输入文本："""
        
        response = self._call_llm(instruction + text)
        response = self.cc.convert(response)
        
        # Extract entities using regex
        pattern = r'\(([^;]+);([^)]+)\)'
        matches = re.findall(pattern, response)
        
        entity_list = []
        category_list = []
        
        for entity, category in matches:
            entity = self.cc.convert(entity)
            category = self.cc.convert(category)
            entity_list.append(entity)
            category_list.append(category)
        
        return entity_list, category_list
    
    def expand_with_aliases(self, entities: List[str]) -> Tuple[List[str], str]:
        """
        Expand entity list with known aliases
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Tuple of (expanded_entity_list, alias_description_string)
        """
        expanded = list(entities)
        alias_str = "存在的人物别称有：\n"
        
        for ent in entities:
            for main_name, aliases in self.entity_aliases.items():
                if ent == main_name or ent in aliases:
                    expanded.append(main_name)
                    expanded.extend(aliases)
                    alias_str += f"主名称：{main_name}, 别名：{aliases}\n"
                    break
        
        return list(set(expanded)), alias_str
    
    def query_graph(self, text: str, alias_str: str) -> List[Tuple]:
        """
        Query Neo4j graph database for relevant relationships
        
        Args:
            text: Query text
            alias_str: String containing entity aliases
            
        Returns:
            List of triples (node1, relationship, node2)
        """
        # Initialize DeepSeek model for Cypher generation
        ds = ChatOpenAI(
            model=self.config['llm_model'],
            api_key=self.config['llm_api_key'],
            base_url=self.config['llm_base_url']
        )
        
        # Define Cypher generation prompt
        cypher_prompt = PromptTemplate.from_template("""
你是一个专业的 Neo4j Cypher 查询生成器。
根据用户的问题，生成对应的 Cypher 查询。
以下是图数据库的 Schema 信息：
- 节点标签：人物,社會集團,地點,時間,職官,爵位,文獻
- 关系类型：(人物,别称,人物),(人物,父母,人物),(人物,子女,人物),(人物,同胞,人物),(人物,祖,人物),(人物,孙,人物),(人物,同族,人物),(人物,婚恋,人物),(人物,上级,人物),(人物,下级,人物),(人物,更代,人物),(人物,承继,人物),(人物,师从,人物),(人物,养亲,人物),(人物,所属团体,社会集团),(人物,籍贯,地点),(人物,居,地点),(人物,辖,地点),(人物,逃奔,地点),(人物,卒于,地点),(人物,访,地点),(人物,占,地点),(人物,攻,地点),(人物,任官,职官),(人物,出生时间,时间),(人物,死亡时间,时间),(人物,封爵,爵位),(社会集团,从属,社会集团),(人物,积极态度倾向,人物),(人物,消极态度倾向,人物),(人物,积极态度倾向,社会集团),(人物,消极态度倾向,社会集团),(社会集团,积极态度倾向,社会集团),(社会集团,积极态度倾向,社会集团)。

示例：
问题：诸葛亮担任什么职位？
查询：MATCH (p:人物 {{name: "諸葛亮"}})-[r]->(q:職官) RETURN p, r, q

现在请将下面回答转为cyber查询，务必遵循注意事项的要求：
问题：{question}
注意：
1.存在的人物别称有{name_str}，请根据人物主名称与别称进行扩展，转为cyber并行检索。若无别称，则不增加别称检索；
2. cyber查询return中一定只能包含p、r、q三者，不一定三个字母都要包含，不能有其他杂项字母；
""")
        
        # Create QA chain
        qa_chain = GraphCypherQAChain.from_llm(
            llm=ds,
            graph=self.graph,
            verbose=True,
            cypher_prompt=cypher_prompt,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )
        
        # Execute query
        try:
            response = qa_chain.invoke({"query": text, "name_str": alias_str})
            generated_cypher = response["intermediate_steps"][0]['query']
            
            with self.neo4j_driver.session() as session:
                result = session.run(generated_cypher)
                triples = []
                
                for record in result:
                    # Parse node and relationship labels dynamically
                    pattern = re.compile(
                        r"(\w)=<Node.*?>[\s\n]*"
                        r"(\w)=<Relationship.*?>[\s\n]*"
                        r"(\w)=<Node.*?>"
                    )
                    matches = pattern.search(str(record))
                    
                    if matches:
                        p_tag, r_tag, q_tag = matches.groups()
                        triples.append((record[p_tag], record[r_tag], record[q_tag]))
                    else:
                        # Handle single node case
                        pattern2 = re.compile(r"(\w)=<Node.*?>[\s\n]*")
                        matches2 = pattern2.search(str(record))
                        if matches2:
                            p_tag = matches2.group(1)
                            triples.append((record[p_tag], None, None))
                
                return triples
        
        except Exception as e:
            print(f"Graph query failed: {e}")
            return []
    
    def semantic_match(self, query: str, triples: List[Tuple], top_k: int = 10) -> List[Tuple]:
        """
        Rank triples by semantic similarity to query
        
        Args:
            query: User query
            triples: List of (node1, relationship, node2) triples
            top_k: Number of top results to return
            
        Returns:
            List of (triple, score) tuples
        """
        if not triples:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        
        # Convert triples to natural language
        triple_descriptions = []
        for triple in triples:
            if triple[1] is not None:  # Has relationship
                desc = f"{triple[0]['name']} {triple[1].type} {triple[2]['name']}"
            else:  # Single node
                desc = triple[0]['name']
            triple_descriptions.append(desc)
        
        # Encode triple descriptions
        triple_embeddings = self.embedding_model.encode(triple_descriptions, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_scores = np.inner(query_embedding.cpu().numpy(), triple_embeddings.cpu().numpy())
        
        # Sort by similarity
        sorted_indices = np.argsort(cos_scores)[::-1]
        ranked_triples = [
            (triples[i], cos_scores[i])
            for i in sorted_indices[:top_k]
        ]
        
        return ranked_triples
    
    def triples_to_text(self, ranked_triples: List[Tuple]) -> str:
        """
        Convert ranked triples to natural language description
        
        Args:
            ranked_triples: List of (triple, score) tuples
            
        Returns:
            Natural language description
        """
        if not ranked_triples:
            return "图数据库未找到相关关系\n"
        
        rel_list = []
        for (s, r, o), score in ranked_triples:
            if r is not None:
                rel_list.append(f"三元组: {s['name']} -[{r.type}]-> {o['name']}")
        
        rel_content = "根据实体抽取的关系三元组如下：\n" + "\n".join(rel_list)
        
        # Convert to natural language using LLM
        instruction = "将下面关系三元组转换为非结构化的自然语言描述：\n" + rel_content
        natural_text = self._call_llm(instruction)
        
        return natural_text
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate final answer using retrieved context
        
        Args:
            query: User query
            context: Retrieved context from graph
            
        Returns:
            Generated answer
        """
        template = """你当前是一个古文知识专家，请学习以下内容：
{context}

根据所学内容以及你自身之前已知的知识回答提问，尽可能多的补充知识，富有条理的回答问题。
提问为：{query}"""
        
        chat_prompt = ChatPromptTemplate.from_messages([("system", template)])
        messages = chat_prompt.format_messages(context=context, query=query)
        
        return self._call_llm(messages[0].content)
    
    def query(self, text: str) -> Tuple[str, Dict]:
        """
        Main query interface - end-to-end GraphRAG pipeline
        
        Args:
            text: User query
            
        Returns:
            Tuple of (answer, graph_visualization_data)
        """
        # Convert to Traditional Chinese
        text = self.cc.convert(text)
        
        print("=== Step 1: Entity Recognition ===")
        entities, categories = self.extract_entities(text)
        print(f"Extracted entities: {entities}")
        
        print("\n=== Step 2: Alias Expansion ===")
        expanded_entities, alias_str = self.expand_with_aliases(entities)
        print(f"Expanded entities: {expanded_entities}")
        
        print("\n=== Step 3: Graph Query ===")
        triples = self.query_graph(text, alias_str)
        print(f"Retrieved {len(triples)} triples")
        
        print("\n=== Step 4: Semantic Ranking ===")
        ranked_triples = self.semantic_match(text, triples, top_k=10)
        
        print("\n=== Step 5: Context Generation ===")
        context = self.triples_to_text(ranked_triples)
        
        print("\n=== Step 6: Answer Generation ===")
        answer = self.generate_response(text, context)
        
        # Prepare graph visualization data
        nodes = []
        edges = []
        for (s, r, o), score in ranked_triples:
            if r is not None:
                nodes.extend([
                    {"id": s.element_id, "label": list(s.labels)[0], **dict(s)},
                    {"id": o.element_id, "label": list(o.labels)[0], **dict(o)}
                ])
                edges.append({
                    "source": s.element_id,
                    "target": o.element_id,
                    "relationship": r.type
                })
        
        graph_data = {"nodes": nodes, "edges": edges}
        
        return answer, graph_data
    
    def close(self):
        """Close database connections"""
        self.neo4j_driver.close()
