"""C1 · 基于 LangChain 的四步 RAG（带分阶段日志）

对应文档：docs/chapter1/03_get_start_rag.md 第三节

运行时会在终端打印 6 个阶段：
    [1/6] 文档开始分片      → 用了什么切分函数、切成几块
    [2/6] 开始构建向量      → 用了什么嵌入模型、多少维
    [3/6] 开始初始化向量库  → 向量库类型
    [4/6] 开始写入向量库    → 写入多少条、库里共几条
    [5/6] llm 相似检索      → 问题、k、命中哪几块、相似度多少
    [6/6] llm 生成回答      → prompt 多长、调了哪个模型、耗时多少

想关掉这些日志：RAG_LOG=quiet python3 01_langchain_example.py
"""

import os as _os, sys as _sys

# 向上找到含 bootstrap.py 的 code/ 目录，加入 sys.path（脚本可能在多级子目录里）
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.exists(
    _os.path.join(_p, "bootstrap.py")
):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import bootstrap  # noqa: F401  仅为了副作用：加载 code/.env、修代理与证书
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效
from llm_config import llm_kwargs
from stage_log import Stage

import os

# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

stage = Stage(total=6, title="C1 · RAG 流水线")
stage.header("LangChain 版 · 数据准备 → 索引构建 → 检索 → 生成")

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# ---------------------------------------------------------------- 1. 数据准备
with stage.step(
    "文档开始分片",
    func="RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)",
    file=markdown_path,
) as step:
    # 加载本地markdown文件
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    step.detail(
        f"加载完成：{len(docs)} 个 Document，共 {sum(len(d.page_content) for d in docs)} 字符"
    )

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(docs)

    lens = [len(c.page_content) for c in chunks]
    step.note(
        f"{len(chunks)} 个 chunk（长度 {min(lens)}~{max(lens)}，均 {sum(lens) // len(lens)}）"
    )

# ---------------------------------------------------------------- 2. 构建向量
with stage.step(
    "开始构建向量",
    model="BAAI/bge-small-zh-v1.5",
    device="cpu",
    normalize="True（L2 归一化，内积即余弦）",
) as step:
    # 中文嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    probe = embeddings.embed_query(chunks[0].page_content)
    step.note(f"向量维度 {len(probe)}，每条 chunk 生成 1 个向量")

# ---------------------------------------------------------------- 3. 初始化向量库
with stage.step(
    "开始初始化向量库",
    type="InMemoryVectorStore（存在进程内存，脚本结束即消失）",
) as step:
    # 构建向量存储
    vectorstore = InMemoryVectorStore(embeddings)
    step.note("实例已创建，此时库内为空")

# ---------------------------------------------------------------- 4. 写入向量库
with stage.step(
    "开始写入向量库",
    payload=f"{len(chunks)} 个 chunk → 逐个编码后写入",
) as step:
    vectorstore.add_documents(chunks)
    step.note(f"库内条数 = {len(chunks)}（= chunk 数）")

# ---------------------------------------------------------------- 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

# 配置大语言模型
# 默认使用 AIHubmix；配置了 LLM_BASE_URL（例如本地 Ollama）时优先走本地/自建端点
llm = ChatOpenAI(temperature=0.7, max_tokens=4096, **llm_kwargs())

# 用户查询
question = "文中举了哪些例子？"

# ---------------------------------------------------------------- 5. 相似检索
with stage.step(
    "llm 相似检索",
    question=question,
    k=3,
    method="InMemoryVectorStore.similarity_search（余弦相似度）",
) as step:
    # 在向量存储中查询相关文档
    hits = vectorstore.similarity_search_with_score(question, k=3)
    # step.note(f"hits：{hits}")
    retrieved_docs = [doc for doc, _ in hits]
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    for i, (doc, score) in enumerate(hits, 1):
        head = doc.page_content[:56].replace("\n", " ")
        step.detail(f"top{i} score={score:.4f}  {head}…")
    step.note(f"命中 {len(hits)} 块，拼成上下文 {len(docs_content)} 字符")

# ---------------------------------------------------------------- 6. 生成回答
with stage.step(
    "llm 生成回答",
    model=llm_kwargs()["model"],
    endpoint=llm_kwargs()["base_url"],
    temperature=0.7,
) as step:
    final_prompt = prompt.format(question=question, context=docs_content)
    step.detail(
        f"最终 prompt = {len(final_prompt)} 字符（上下文 {len(docs_content)} + 模板）"
    )

    answer = llm.invoke(final_prompt)

    usage = answer.response_metadata.get("token_usage", {})
    step.note(
        f"prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"total={usage.get('total_tokens')}"
    )

stage.summary()

print()
print("=" * 72)
print("  最终回答（answer.content）")
print("=" * 72)
print(answer.content)
print("=" * 72)
print("原始返回对象（含 additional_kwargs / response_metadata / id 等）：")
print(answer)
