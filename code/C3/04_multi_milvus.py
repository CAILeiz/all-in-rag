"""C3-04 · Milvus 多模态检索：把图片存进向量库

对应文档：docs/chapter3/09_milvus.md

这一节的要点：**向量库不只存向量，还存「能拿回来用的东西」。**
这里存三样：
    id          自增主键，插入时由 Milvus 生成
    vector      768 维多模态向量
    image_path  原图路径 ← 检索命中后靠它把图取回来展示

流程：编码入库 → 建 HNSW 索引 → 用「一张图 + 一句话」联合查询

前置条件：
    1. Milvus 必须在跑（默认 localhost:19530）
       cd code && docker compose -f docker-compose.local.yml up -d
    2. 模型权重已下载：python3 code/C3/download_model.py

运行：
    python3 code/C3/04_multi_milvus.py
    RAG_LOG=quiet python3 code/C3/04_multi_milvus.py

7 个阶段：
    [1/7] 加载编码器 + 连接 Milvus
    [2/7] 创建 Collection（含字段 Schema）
    [3/7] 批量编码图像
    [4/7] 插入数据
    [5/7] 建 HNSW 索引并加载到内存
    [6/7] 多模态查询（图 + 文）
    [7/7] 可视化结果并清理
"""
import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、固定缓存目录、修代理与证书

bootstrap.anchor_cwd(__file__)  # 工作目录钉在脚本所在目录，相对路径才不会失效
from stage_log import Stage

import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from visual_bge.visual_bge.modeling import Visualized_BGE

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"
QUERY_TEXT = "一条龙"


class Encoder:
    """把图像（可选叠加文本）编码为向量。"""

    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(
    query_image_path: str,
    retrieved_images: list,
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
    """把「查询图 + 检索到的图」拼成一张全景图。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(
        resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
    )
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(
        bordered_query, (img_width, img_height)
    )
    cv2.putText(
        query_display_area, "Query", (10, panoramic_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
    )

    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(
            resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        panoramic_image[start_row:start_row + img_height,
                        start_col:start_col + img_width] = bordered_retrieved

        cv2.putText(
            panoramic_image, str(i), (start_col + 10, start_row + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        )

    return np.hstack([query_display_area, panoramic_image])


stage = Stage(total=7, title="C3 · Milvus 多模态检索")
stage.header("图像编码入库 → HNSW 索引 → 图+文 联合查询")

# ---------------------------------------------------------------- 1. 初始化
with stage.step(
    "加载编码器 + 连接 Milvus",
    model=MODEL_NAME,
    uri=MILVUS_URI,
    why="Milvus 是服务端型向量库，连不上会在这里直接失败",
) as step:
    encoder = Encoder(MODEL_NAME, MODEL_PATH)
    milvus_client = MilvusClient(uri=MILVUS_URI)
    step.detail("编码器就绪，Milvus 连接成功")
    step.note("已连接")

# ---------------------------------------------------------------- 2. Collection
with stage.step(
    "创建 Collection",
    name=COLLECTION_NAME,
    why="先按 schema 建表，再往里插数据；字段名一旦定下就别随意改",
) as step:
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
        step.detail(f"已删除同名旧 Collection: {COLLECTION_NAME}")

    image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
    if not image_list:
        raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")

    # 维度从模型实际输出里取，避免硬编码 768 写错
    dim = len(encoder.encode_image(image_list[0]))

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="多模态图文检索")
    milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    step.detail(f"字段：id(自增) / vector(dim={dim}) / image_path(VARCHAR 512)")
    step.detail(f"待入库图片：{len(image_list)} 张")
    step.note(f"dim={dim}")

# ---------------------------------------------------------------- 3. 编码
with stage.step(
    "批量编码图像",
    func="Encoder.encode_image()",
    count=len(image_list),
    why="每张图过一次 Visualized BGE，这是最慢的一步",
) as step:
    data_to_insert = []
    for image_path in image_list:
        vector = encoder.encode_image(image_path)
        data_to_insert.append({"vector": vector, "image_path": image_path})
        step.detail(f"已编码 {os.path.basename(image_path)}")
    step.note(f"{len(data_to_insert)} 条向量")

# ---------------------------------------------------------------- 4. 插入
with stage.step(
    "插入数据",
    api="milvus_client.insert(...)",
    why="自增主键由 Milvus 生成，插入时不用（也不能）自己指定 id",
) as step:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    step.detail(f"insert_count = {result['insert_count']}")
    step.note(f"{result['insert_count']} 条")

# ---------------------------------------------------------------- 5. 索引
with stage.step(
    "建 HNSW 索引并加载",
    index_type="HNSW",
    metric="COSINE",
    params="M=16, efConstruction=256",
    why="不建索引也能搜，但数据量一大就退化成暴力扫描",
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
    step.detail("HNSW：图结构近邻索引，M 控制每个节点的连边数")
    step.detail("Collection 已加载到内存，可以开始检索")
    step.note("索引就绪")

# ---------------------------------------------------------------- 6. 检索
with stage.step(
    "多模态查询",
    query_image="data/C3/dragon/query.png",
    query_text=QUERY_TEXT,
    why="图 + 文一起编码，比单用图更准（回顾 01 的相似度对比）",
) as step:
    query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
    query_vector = encoder.encode_query(image_path=query_image_path, text=QUERY_TEXT)

    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["image_path"],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]

    retrieved_images = []
    for i, hit in enumerate(search_results, 1):
        step.detail(
            f"Top {i}: 距离={hit['distance']:.4f}  {os.path.basename(hit['entity']['image_path'])}"
        )
        retrieved_images.append(hit["entity"]["image_path"])
    step.note(f"Top{len(retrieved_images)} 命中")

# ---------------------------------------------------------------- 7. 可视化
with stage.step(
    "可视化结果并清理",
    output="data/C3/search_result.png",
    why="演示脚本用完就删 Collection，避免下次运行残留脏数据",
) as step:
    if not retrieved_images:
        step.detail("没有检索到任何图像。")
    else:
        panoramic_image = visualize_results(query_image_path, retrieved_images)
        combined_image_path = os.path.join(DATA_DIR, "search_result.png")
        cv2.imwrite(combined_image_path, panoramic_image)
        step.detail(f"结果图已保存：{combined_image_path}（自动弹出预览）")
        Image.open(combined_image_path).show()

    milvus_client.release_collection(collection_name=COLLECTION_NAME)
    milvus_client.drop_collection(COLLECTION_NAME)
    step.detail(f"已释放并从 Milvus 删除 Collection: {COLLECTION_NAME}")
    step.note("已清理")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. Top1 就是 query.png 自己（距离 0.9466）—— 说明检索链路是通的")
print("2. 后面几条按「龙」的视觉+语义相似度排序，不是按文件名")
print("3. HNSW 的 ef（查询时）和 M/efConstruction（建索引时）是两套参数：")
print("   建索引时决定图的质量，查询时决定搜索的广度")
print("4. Collection 的字段设计决定你能「取回什么」—— 这里存 image_path，")
print("   所以检索结果能直接渲染成图片；只存 vector 的话就取不回原图")
print()
print("提示：连不上 Milvus 就先 docker compose -f code/docker-compose.local.yml up -d")
print("=" * 72)
