import os as _os, sys as _sys
# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(_os.path.join(_p, "bootstrap.py")):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import llm_kwargs
from stage_log import Stage

import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
import numpy as np

stage = Stage(total=4, title="C4 · 嵌入路由（按语义相似度分流）")
stage.header("把「菜系描述」编码成向量，用余弦相似度选路由 —— 不调 LLM 做分类")

# 1. 定义路由描述
sichuan_route_prompt = "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。"
cantonese_route_prompt = "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"

route_prompts = [sichuan_route_prompt, cantonese_route_prompt]
route_names = ["川菜", "粤菜"]

# 初始化嵌入模型，并对路由描述进行向量化
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
route_prompt_embeddings = embeddings.embed_documents(route_prompts)

with stage.step(
    "定义路由描述并编码",
    embed="BAAI/bge-small-zh-v1.5",
    routes=", ".join(route_names),
    why="每个菜系用一段「典型菜品关键词」描述，问题越接近这段描述就越可能被路由到它",
) as step:
    for name, prompt in zip(route_names, route_prompts):
        step.detail(f"{name}：{prompt[:36]}...")
    step.note(f"{len(route_names)} 条路由向量")

# 2. 定义不同路由的目标链
llm = ChatDeepSeek(temperature=0, **llm_kwargs())

# 定义川菜和粤菜处理链
sichuan_chain = (
    PromptTemplate.from_template("你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)
cantonese_chain = (
    PromptTemplate.from_template("你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

route_map = { "川菜": sichuan_chain, "粤菜": cantonese_chain }

with stage.step(
    "构建两条处理链",
    steps="PromptTemplate | ChatDeepSeek | StrOutputParser",
) as step:
    step.detail("川菜链、粤菜链各一条，用相同结构、不同人设")
    step.note("2 条链")

# 3. 创建路由函数
def route(info):
    # 对用户查询进行嵌入
    query_embedding = embeddings.embed_query(info["query"])
    
    # 计算与各路由提示的余弦相似度
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    
    # 找到最相似的路由
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]
    
    route.last_scores = (chosen_route_name, similarity_scores)
    
    # 获取对应的处理链
    chosen_chain = route_map[chosen_route_name]
    
    # 直接调用选中的链并返回结果
    return chosen_chain.invoke(info)

# 创建完整的路由链
full_chain = RunnableLambda(route)


# 4. 运行演示查询
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

with stage.step(
    "演示查询",
    count=len(demo_queries),
    why="看每个问题的相似度分数，理解它为什么被分到这一路",
) as step:
    for i, query in enumerate(demo_queries, 1):
        step.detail("")
        step.detail(f"--- 问题 {i}: {query} ---")
        try:
            scores = {}
            q_emb = embeddings.embed_query(query)
            sims = cosine_similarity([q_emb], route_prompt_embeddings)[0]
            for name, sim in zip(route_names, sims):
                scores[name] = float(sim)
            step.detail("  相似度：" + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

            result = full_chain.invoke({"query": query})
            preview = str(result).strip().splitlines()
            for line in preview[:2]:
                step.detail(f"  回答：{line}")
        except Exception as e:
            step.detail(f"执行错误：{e}")
    step.note(f"{len(demo_queries)} 个问题")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 01 用 LLM 分类（2 次调用）；06 只用嵌入模型算一次相似度（0 次 LLM 分类）")
print("2. 便宜、快、稳定；但只认「描述里写过的词」")
print("3. 路由描述怎么写得看业务：把用户真实问法里的高频词放进去，命中率才高")
print("4. 分数接近时（比如 0.52 vs 0.51）结果不可靠，需要设阈值或回退到通用链")
print()
print("提示：bge-small-zh 对中文短语区分度一般，换成 bge-m3 通常更稳")
print("=" * 72)

