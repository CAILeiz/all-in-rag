import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

import json
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DragonImage:
    """龙类图像数据类"""
    img_id: str
    path: str
    title: str
    description: str
    category: str
    location: str
    environment: str
    combat_details: Dict[str, Any] = None
    scene_info: Dict[str, Any] = None

class DragonDataset:
    """龙类图像数据集管理类"""
    def __init__(self, data_dir: str, metadata_path: str):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[DragonImage] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """加载图像元数据"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for img_data in data:
                # 确保图片路径是完整的
                if not img_data['path'].startswith(self.data_dir):
                    img_data['path'] = os.path.join(self.data_dir, img_data['path'].split('/')[-1])
                self.images.append(DragonImage(**img_data))
    
    def get_image_paths(self) -> List[str]:
        """获取所有图像路径"""
        return [img.path for img in self.images]
    
    def get_metadata_by_path(self, path: str) -> DragonImage:
        """根据路径获取元数据"""
        for img in self.images:
            if img.path == path:
                return img
        return None
    
    def get_text_content(self, img: DragonImage) -> str:
        """获取图像的文本描述内容"""
        parts = [
            img.title, img.description,
            img.location, img.environment
        ]
        if img.combat_details:
            parts.extend(img.combat_details.get('combat_style', []))
            parts.extend(img.combat_details.get('abilities_used', []))
        if img.scene_info:
            parts.append(img.scene_info.get('time_of_day', ''))
        return ' '.join(filter(None, parts))

class Encoder:
    """编码器类，用于将图像和文本编码为向量"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str = None, text: str = None) -> list[float]:
        """编码查询（支持图像+文本或仅文本）"""
        with torch.no_grad():
            if image_path and text:
                query_emb = self.model.encode(image=image_path, text=text)
            elif image_path:
                query_emb = self.model.encode(image=image_path)
            elif text:
                query_emb = self.model.encode(text=text)
            else:
                raise ValueError("必须提供图像路径或文本内容")
        return query_emb.tolist()[0]

    def encode_multimodal(self, image_path: str, text: str) -> list[float]:
        """编码多模态内容（图像+文本）"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

def visualize_results(query_image_path: str, retrieved_results: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的结果创建一个全景图用于可视化"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    if query_image_path and os.path.exists(query_image_path):
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, result in enumerate(retrieved_results):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        img_path = result['image_path']
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号和相似度
        cv2.putText(panoramic_image, f"{i+1}", (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(panoramic_image, f"{result['distance']:.3f}", (start_col + 10, start_row + img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return np.hstack([query_display_area, panoramic_image])

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3/dragon"
METADATA_PATH = "../../data/C4/metadata/dragon.json"
COLLECTION_NAME = "multimodal_dragon_demo"
MILVUS_URI = "http://localhost:19530"

OUTPUT_FIELDS = ["img_id", "image_path", "title", "description",
                 "category", "location", "environment"]

stage = Stage(total=8, title="C3 · 多模态龙类检索（work）")
stage.header("多模态向量入库 → 三种查询方式对比（图+文 / 纯文 / 纯图）")

# ---------------------------------------------------------------- 1. 数据集
with stage.step(
    "加载数据集",
    data_dir=DATA_DIR,
    metadata=METADATA_PATH,
    why="每张图除了向量，还带 title/description/category 等元数据，检索后一并取回",
) as step:
    dataset = DragonDataset(DATA_DIR, METADATA_PATH)
    step.detail(f"共 {len(dataset.images)} 张龙类图像")
    for img in dataset.images[:3]:
        step.detail(f"  · {img.title}（{img.category}）")
    step.note(f"{len(dataset.images)} 张")

# ---------------------------------------------------------------- 2. 编码器
with stage.step(
    "加载编码器 + 连接 Milvus",
    model=MODEL_NAME,
    uri=MILVUS_URI,
) as step:
    encoder = Encoder(MODEL_NAME, MODEL_PATH)
    milvus_client = MilvusClient(uri=MILVUS_URI)
    step.note("已连接")

# ---------------------------------------------------------------- 3. Collection
with stage.step(
    "创建 Collection",
    name=COLLECTION_NAME,
    why="相比 C3-04 只存 image_path，这里存了完整的业务元数据",
) as step:
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
        step.detail(f"已删除同名旧 Collection: {COLLECTION_NAME}")

    # 维度从真实输出里取，避免硬编码
    sample_text = dataset.get_text_content(dataset.images[0])
    sample_path = dataset.images[0].path
    dim = len(encoder.encode_multimodal(sample_path, sample_text))

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="多模态龙类图像检索")
    milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    step.detail(f"9 个字段：id / vector(dim={dim}) / img_id / image_path / title")
    step.detail("           description / category / location / environment")
    step.note(f"dim={dim}")

# ---------------------------------------------------------------- 4. 编码入库
with stage.step(
    "批量编码并插入",
    func="Encoder.encode_multimodal(img, text)",
    count=len(dataset.images),
    why="和 04 的纯图像编码不同：这里把元数据文字也一起编码进向量",
) as step:
    data_to_insert = []
    for img_data in tqdm(dataset.images, desc="生成多模态嵌入"):
        text_content = dataset.get_text_content(img_data)
        vector = encoder.encode_multimodal(img_data.path, text_content)
        data_to_insert.append({
            "vector": vector,
            "img_id": img_data.img_id,
            "image_path": img_data.path,
            "title": img_data.title,
            "description": img_data.description,
            "category": img_data.category,
            "location": img_data.location,
            "environment": img_data.environment,
        })

    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    step.detail(f"insert_count = {result['insert_count']}")
    step.note(f"{result['insert_count']} 条")

# ---------------------------------------------------------------- 5. 索引
with stage.step(
    "建 HNSW 索引并加载",
    index_type="HNSW",
    metric="COSINE",
    params="M=16, efConstruction=256",
) as step:
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    milvus_client.load_collection(collection_name=COLLECTION_NAME)
    step.detail("HNSW 索引已建，Collection 已加载进内存")
    step.note("索引就绪")

# ---------------------------------------------------------------- 6. 图+文
with stage.step(
    "查询①：图像 + 文本",
    image="query.png",
    text="悬崖上的巨龙",
    why="图给视觉线索，文给语义方向，两者融合后最准",
) as step:
    query_image_path = os.path.join(DATA_DIR, "query.png")
    query_text = "悬崖上的巨龙"
    query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=OUTPUT_FIELDS,
        limit=6,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]

    retrieved_results = []
    for i, hit in enumerate(search_results, 1):
        step.detail(
            f"Top {i}: {hit['distance']:.4f}  {hit['entity']['title']}"
            f"（{hit['entity']['category']}）"
        )
        retrieved_results.append({
            "image_path": hit["entity"]["image_path"],
            "distance": hit["distance"],
        })
    step.note(f"Top{len(retrieved_results)}")

# ---------------------------------------------------------------- 7. 纯文 / 纯图
with stage.step(
    "查询②③：纯文本 vs 纯图像",
    text_query="悬崖上的巨龙",
    image_query="query.png",
    why="同一意图换一种输入方式，看排序怎么变",
) as step:
    text_query = "悬崖上的巨龙"
    text_query_vector = encoder.encode_query(text=text_query)
    text_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[text_query_vector],
        output_fields=OUTPUT_FIELDS,
        limit=3,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]
    step.detail("--- 纯文本查询 ---")
    for i, hit in enumerate(text_results, 1):
        step.detail(f"  Top {i}: {hit['distance']:.4f}  {hit['entity']['title']}")

    step.detail("")
    image_query_path = os.path.join(DATA_DIR, "query.png")
    image_query_vector = encoder.encode_query(image_path=image_query_path)
    image_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[image_query_vector],
        output_fields=OUTPUT_FIELDS,
        limit=3,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]
    step.detail("--- 纯图像查询 ---")
    for i, hit in enumerate(image_results, 1):
        step.detail(f"  Top {i}: {hit['distance']:.4f}  {hit['entity']['title']}")

    step.detail("")
    step.detail("三种查询的 Top1 是否一致？一致说明图文编码空间已经对齐")
    step.note("3 组结果")

# ---------------------------------------------------------------- 8. 可视化
with stage.step(
    "可视化并清理",
    output="data/C4/multimodal_search.png",
) as step:
    if retrieved_results:
        panoramic_image = visualize_results(query_image_path, retrieved_results)
        combined_image_path = "../../data/C4/multimodal_search.png"
        cv2.imwrite(combined_image_path, panoramic_image)
        step.detail(f"结果图已保存：{combined_image_path}")

    milvus_client.release_collection(collection_name=COLLECTION_NAME)
    milvus_client.drop_collection(COLLECTION_NAME)
    step.detail(f"已释放并删除 Collection: {COLLECTION_NAME}")
    step.note("已清理")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 图+文查询通常最准：视觉和语义两个信号投票")
print("2. 纯文本查询依赖向量里「学到的图文对应关系」，描述越具体越准")
print("3. 纯图像查询最容易受背景干扰（比如都拍到了天空/悬崖）")
print("4. 生产建议：默认走图+文；只有用户没输入文字时才退回纯图像")
print()
print("提示：本脚本会删掉演示用 Collection，重复跑不会累积脏数据")
print("=" * 72)
