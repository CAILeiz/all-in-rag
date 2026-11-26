"""C4-03 · Text2SQL v2：知识库向量化 + 示例驱动

对应文档：docs/chapter4/03_text2sql.md（与 03 对照阅读）

和 03 的差别：

    03    给 LLM 一份 schema，让它现场生成 SQL
    03v2  先把「问题→SQL」的历史示例存进向量库，检索相似示例再拼 prompt

v2 的思路是 **few-shot 检索**：库里已有的相似问题，比空口描述 schema 更能约束
LLM 的写法（尤其是字段名、聚合口径、日期格式）。

这个脚本里的 demo() 只跑后半段（编码 + 直接执行 SQL），不连 LLM，
方便你先确认「向量维度和 SQL 执行」这两件事是通的。

运行（要 Milvus 才跑得完整，demo() 本身只用到嵌入模型 + sqlite）：
    python3 code/C4/03_text2sql_demo_v2.py
    RAG_LOG=quiet python3 code/C4/03_text2sql_demo_v2.py

4 个阶段：
    [1/4] 加载嵌入模型并测试编码
    [2/4] 建演示数据库
    [3/4] 执行 3 条 SQL 并看结果
    [4/4] 清理
"""
import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
from stage_log import Stage


class BGESmallEmbeddingFunction:
    """BGE-Small中文嵌入函数，用于Text2SQL知识库向量化"""
    
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dense_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, texts):
        """编码文本为密集向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            convert_to_numpy=True
        )
        
        return embeddings
    
    @property
    def dim(self):
        """返回向量维度"""
        return self.dense_dim


class SimpleKnowledgeBase:
    """简化的知识库，使用BGE-Small进行向量检索"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.collection_name = "text2sql_knowledge_base"
        self.milvus_client = None
        self.collection = None
        
        self.embedding_function = BGESmallEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )
        
        self.sql_examples = []
        self.table_schemas = []
        self.data_loaded = False
    
    def connect_milvus(self):
        """连接Milvus数据库"""
        connections.connect(uri=self.milvus_uri)
        self.milvus_client = MilvusClient(uri=self.milvus_uri)
        return True
    
    def create_collection(self):
        """创建Milvus集合"""
        if not self.milvus_client:
            self.connect_milvus()
        
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="sql", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim)
        ]
        
        schema = CollectionSchema(fields, description="Text2SQL知识库")
        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong")
        
        index_params = {"index_type": "AUTOINDEX", "metric_type": "IP", "params": {}}
        self.collection.create_index("embedding", index_params)
        
        return True
    
    def load_data(self):
        """加载知识库数据"""        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        self.load_sql_examples(data_dir)
        self.load_table_schemas(data_dir)
        self.vectorize_and_store()
        
        self.data_loaded = True
    
    def load_sql_examples(self, data_dir: str):
        """加载SQL示例"""
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json")
        
        default_examples = [
            {"question": "查询所有用户信息", "sql": "SELECT * FROM users", "description": "获取用户记录", "database": "sqlite"},
            {"question": "年龄大于30的用户", "sql": "SELECT * FROM users WHERE age > 30", "description": "年龄筛选", "database": "sqlite"},
            {"question": "统计用户总数", "sql": "SELECT COUNT(*) as user_count FROM users", "description": "用户计数", "database": "sqlite"},
            {"question": "查询库存不足的产品", "sql": "SELECT * FROM products WHERE stock < 50", "description": "库存筛选", "database": "sqlite"},
            {"question": "查询用户订单信息", "sql": "SELECT u.name, p.name, o.quantity FROM orders o JOIN users u ON o.user_id = u.id JOIN products p ON o.product_id = p.id", "description": "订单详情", "database": "sqlite"},
            {"question": "按城市统计用户", "sql": "SELECT city, COUNT(*) as count FROM users GROUP BY city", "description": "城市分组", "database": "sqlite"}
        ]
        
        if os.path.exists(sql_examples_path):
            with open(sql_examples_path, 'r', encoding='utf-8') as f:
                self.sql_examples = json.load(f)
        else:
            self.sql_examples = default_examples
            os.makedirs(data_dir, exist_ok=True)
            with open(sql_examples_path, 'w', encoding='utf-8') as f:
                json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
    
    def load_table_schemas(self, data_dir: str):
        """加载表结构信息"""
        schema_path = os.path.join(data_dir, "table_schemas.json")
        
        default_schemas = [
            {
                "table_name": "users",
                "description": "用户信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "name", "type": "VARCHAR", "description": "用户姓名"},
                    {"name": "age", "type": "INTEGER", "description": "用户年龄"},
                    {"name": "email", "type": "VARCHAR", "description": "邮箱地址"},
                    {"name": "city", "type": "VARCHAR", "description": "所在城市"},
                    {"name": "created_at", "type": "DATETIME", "description": "创建时间"}
                ]
            },
            {
                "table_name": "products",
                "description": "产品信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "product_name", "type": "VARCHAR", "description": "产品名称"},
                    {"name": "category", "type": "VARCHAR", "description": "产品类别"},
                    {"name": "price", "type": "DECIMAL", "description": "产品价格"},
                    {"name": "stock", "type": "INTEGER", "description": "库存数量"},
                    {"name": "description", "type": "TEXT", "description": "产品描述"}
                ]
            },
            {
                "table_name": "orders",
                "description": "订单信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "订单ID"},
                    {"name": "user_id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "product_id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "quantity", "type": "INTEGER", "description": "购买数量"},
                    {"name": "total_price", "type": "DECIMAL", "description": "总价格"},
                    {"name": "order_date", "type": "DATETIME", "description": "订单日期"}
                ]
            }
        ]
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.table_schemas = json.load(f)
        else:
            self.table_schemas = default_schemas
            os.makedirs(data_dir, exist_ok=True)
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(self.table_schemas, f, ensure_ascii=False, indent=2)
    
    def vectorize_and_store(self):
        """向量化数据并存储到Milvus"""
        self.create_collection()
        
        all_texts = []
        all_metadata = []
        
        for example in self.sql_examples:
            text = f"问题: {example['question']} SQL: {example['sql']} 描述: {example.get('description', '')}"
            all_texts.append(text)
            all_metadata.append({
                "content_type": "sql_example",
                "question": example['question'],
                "sql": example['sql'],
                "description": example.get('description', ''),
                "table_name": ""
            })
        
        for schema in self.table_schemas:
            columns_desc = ", ".join([f"{col['name']} ({col['type']}): {col.get('description', '')}" 
                                    for col in schema['columns']])
            text = f"表 {schema['table_name']}: {schema['description']} 字段: {columns_desc}"
            all_texts.append(text)
            all_metadata.append({
                "content_type": "table_schema",
                "question": "",
                "sql": "",
                "description": schema['description'],
                "table_name": schema['table_name']
            })
        
        embeddings = self.embedding_function.encode_text(all_texts)
        
        insert_data = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, all_metadata)):
            insert_data.append([
                metadata["content_type"],
                metadata["question"],
                metadata["sql"],
                metadata["description"],
                metadata["table_name"],
                embedding.tolist()
            ])
        
        self.collection.insert(insert_data)
        self.collection.flush()
        self.collection.load()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关的知识库信息"""
        if not self.data_loaded:
            self.load_data()
        
        query_embedding = self.embedding_function.encode_text([query])[0]
        
        search_params = {"metric_type": "IP", "params": {}}
        results = self.collection.search(
            [query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content_type", "question", "sql", "description", "table_name"]
        )[0]
        
        formatted_results = []
        for hit in results:
            result = {
                "score": float(hit.distance),
                "content_type": hit.entity.get("content_type"),
                "question": hit.entity.get("question"),
                "sql": hit.entity.get("sql"),
                "description": hit.entity.get("description"),
                "table_name": hit.entity.get("table_name")
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """降级搜索方法（简单文本匹配）"""
        results = []
        query_lower = query.lower()
        
        for example in self.sql_examples:
            question_lower = example['question'].lower()
            sql_lower = example['sql'].lower()
            
            score = 0
            for word in query_lower.split():
                if word in question_lower:
                    score += 2
                if word in sql_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "score": score,
                    "content_type": "sql_example",
                    "question": example['question'],
                    "sql": example['sql'],
                    "description": example.get('description', ''),
                    "table_name": ""
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def add_sql_example(self, question: str, sql: str, description: str = ""):
        """添加新的SQL示例"""
        new_example = {
            "question": question,
            "sql": sql,
            "description": description,
            "database": "sqlite"
        }
        self.sql_examples.append(new_example)
        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json")
        
        with open(sql_examples_path, 'w', encoding='utf-8') as f:
            json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
        
        if self.collection and self.data_loaded:
            text = f"问题: {question} SQL: {sql} 描述: {description}"
            embedding = self.embedding_function.encode_text([text])[0]
            
            insert_data = [[
                "sql_example",
                question,
                sql,
                description,
                "",
                embedding.tolist()
            ]]
            
            self.collection.insert(insert_data)
            self.collection.flush()
    
    def cleanup(self):
        """清理资源"""
        if self.collection:
            self.collection.release()
        
        if self.milvus_client and self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)


def demo():
    """简单演示：编码 → 建库 → 执行 SQL → 清理。"""
    stage = Stage(total=4, title="C4 · Text2SQL v2（示例检索）")
    stage.header("先把「问题→SQL」示例向量化，再检索相似示例约束 LLM")

    # 1. 模型测试
    with stage.step(
        "加载嵌入模型并测试编码",
        model="BAAI/bge-small-zh-v1.5",
        why="知识库里的「问题+SQL+描述」要先编码成向量，才能被检索到",
    ) as step:
        embedding_function = BGESmallEmbeddingFunction()
        test_texts = ["查询用户", "统计数据"]
        embeddings = embedding_function.encode_text(test_texts)
        step.detail(f"向量维度：{embeddings.shape}")
        step.detail("这两条文本的向量就是后面检索示例的「钥匙」")
        step.note(f"dim={embeddings.shape[1]}")

    # 2. 数据库查询演示
    db_path = "demo.db"

    with stage.step(
        "建演示数据库",
        table="users(id, name, age, city)",
        why="表越简单越好：这一步只验证「SQL 能跑通」，不验证 LLM",
    ) as step:
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)")
        users_data = [(1, "张三", 25, "北京"), (2, "李四", 32, "上海"), (3, "王五", 35, "深圳")]
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users_data)
        conn.commit()
        step.detail("插入 3 行用户数据")
        step.note(f"{db_path}")

    test_sqls = [
        ("查询所有用户", "SELECT * FROM users"),
        ("年龄大于30的用户", "SELECT * FROM users WHERE age > 30"),
        ("统计用户总数", "SELECT COUNT(*) FROM users"),
    ]

    with stage.step(
        "执行 SQL 并看结果",
        count=len(test_sqls),
        why="第 3 条是聚合查询 —— 这类问题向量检索给不出答案，只能走 SQL",
    ) as step:
        for i, (question, sql) in enumerate(test_sqls, 1):
            step.detail("")
            step.detail(f"问题 {i}: {question}")
            step.detail(f"  SQL：{sql}")
            cursor.execute(sql)
            rows = cursor.fetchall()
            if rows:
                step.detail(f"  返回 {len(rows)} 行")
                for j, row in enumerate(rows[:2], 1):
                    step.detail(f"    {j}. {row}")
                if len(rows) > 2:
                    step.detail(f"    ... 还有 {len(rows) - 2} 行")
            else:
                step.detail("  无数据返回")
        step.note(f"{len(test_sqls)} 条 SQL")

    with stage.step("清理", why="删掉临时库，避免污染下一次运行") as step:
        conn.close()
        os.remove(db_path)
        step.detail(f"已删除 {db_path}")
        step.note("已清理")

    stage.summary()

    print()
    print("=" * 72)
    print("  观察点")
    print("=" * 72)
    print("1. 03 靠 schema 现场生成；03v2 靠「检索相似示例」来约束写法")
    print("2. few-shot 检索对字段名、聚合口径的帮助，比纯 schema 描述更直接")
    print("3. 示例库要持续维护：业务新增表，就得补新的「问题→SQL」对")
    print("4. 检索出来的示例不能照抄，要连同当前 schema 一起喂给 LLM")
    print()
    print("提示：这个 demo 不调 LLM，所以即使 Ollama 没开也能跑完")
    print("=" * 72)


if __name__ == "__main__":
    demo()