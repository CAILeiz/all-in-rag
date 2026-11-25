"""C1 拆解实验：把「四步构建 RAG」拆成 4 个可以单独观察的阶段。

`01_langchain_example.py` 一口气跑完四步，只能看到最后一段回答；
这个脚本把中间过程全部打印出来，方便你验证每一步到底发生了什么。

用法（在 code/C1 目录下）：

    python3 lab_walkthrough.py              # 四个阶段全跑
    python3 lab_walkthrough.py 1            # 只看阶段1：数据准备（加载+分块）
    python3 lab_walkthrough.py 2            # 阶段2：索引构建（看向量长什么样）
    python3 lab_walkthrough.py 3            # 阶段3：检索（看命中了哪几块、相似度多少）
    python3 lab_walkthrough.py 4            # 阶段4：生成（看拼出来的 prompt 和最终回答）
    python3 lab_walkthrough.py 3 --k 5      # 阶段3 换 k=5

阶段之间的数据不落盘，全跑时会复用同一份 chunks / 向量，只加载一次模型。
"""
import argparse
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import bootstrap  # noqa: F401  加载 code/.env（缓存目录、LLM 端点）
bootstrap.anchor_cwd(__file__)   # 工作目录钉在脚本所在目录，相对路径才不会失效


from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import llm_kwargs

MD_PATH = "../../data/C1/markdown/easy-rl-chapter1.md"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
QUESTION = "文中举了哪些例子？"

PROMPT = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------ 阶段 1

def stage1_load_and_chunk(chunk_size=4000):
    banner("阶段 1 · 数据准备：加载 markdown → 切成 chunks")

    loader = UnstructuredMarkdownLoader(MD_PATH)
    docs = loader.load()
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"① 加载完成")
    print(f"   文件        : {MD_PATH}")
    print(f"   Document 数 : {len(docs)}（一个 md 文件通常就是 1 个 Document）")
    print(f"   总字符数    : {total_chars}")
    print(f"   metadata    : {docs[0].metadata}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    lens = [len(c.page_content) for c in chunks]
    print(f"\n② 分块完成（RecursiveCharacterTextSplitter，chunk_size={chunk_size} / overlap=200）")
    print(f"   块数        : {len(chunks)}")
    print(f"   长度 min/avg/max : {min(lens)} / {sum(lens)//len(lens)} / {max(lens)}")
    for i, c in enumerate(chunks):
        head = c.page_content[:46].replace("\n", " ")
        print(f"   chunk[{i}] len={len(c.page_content):5d}  {head}…")

    # 顺带证明 chunk_size 的影响 —— 这是课后练习之一
    print(f"\n③ 换个 chunk_size 再切一次（对比用）")
    for cs in (500, 1500, 4000):
        small = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=200).split_documents(docs)
        print(f"   chunk_size={cs:5d} → {len(small):3d} 块，"
              f"平均 {sum(len(c.page_content) for c in small)//len(small):5d} 字符")
    print("\n   注意：块切得越小，检索越精准但上下文越碎；切得越大，上下文完整但会混入噪声。")
    return chunks


# ------------------------------------------------------------------ 阶段 2

def stage2_index(chunks):
    banner("阶段 2 · 索引构建：chunk → 向量 → 放进向量库")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"① 嵌入模型就绪：{EMBED_MODEL}")
    print("   normalize_embeddings=True → 每个向量 L2 范数都是 1，内积即余弦相似度")

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    dim = len(vectors[0])
    import math
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    print(f"\n② 向量化完成")
    print(f"   向量维度    : {dim}")
    print(f"   向量条数    : {len(vectors)}（每个 chunk 一条）")
    print(f"   第1条前6维  : {[round(v, 4) for v in vectors[0][:6]]}")
    print(f"   第1条L2范数 : {norm:.4f}")

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)
    print(f"\n③ 入库完成：InMemoryVectorStore（存在进程内存里，脚本结束就没了）")
    print(f"   库内条数    : 等于 chunk 数 = {len(chunks)}")
    print("\n   为什么需要向量？因为向量空间里『语义相近 → 距离相近』，")
    print("   所以问题『举了哪些例子』才能匹配到讲述例子的那一段，即使字面上不重合。")
    return embeddings, vectorstore


# ------------------------------------------------------------------ 阶段 3

def stage3_retrieve(vectorstore, k=3, chunks=None):
    banner(f"阶段 3 · 检索：用问题去向量库里找 top-{k}")

    print(f"问题: {QUESTION}\n")
    hits = vectorstore.similarity_search_with_score(QUESTION, k=k)
    print(f"命中 {len(hits)} 块（score 越大越相似，这里是余弦相似度）：\n")
    for i, (doc, score) in enumerate(hits, 1):
        preview = doc.page_content[:110].replace("\n", " ")
        idx = next((j for j, c in enumerate(chunks or []) if c.page_content == doc.page_content), "?")
        marks = [w for w in ("餐馆", "广告", "挖油", "游戏") if w in doc.page_content]
        print(f"  [{i}] score={score:.4f}  len={len(doc.page_content)}  → 对应 chunk[{idx}]")
        if marks:
            print(f"        ★ 这一块里确实出现了例子关键词：{marks}")
        print(f"      {preview}…")

    context = "\n\n".join(doc.page_content for doc, _ in hits)
    print(f"\n拼成上下文：{len(hits)} 块 → 共 {len(context)} 字符")
    print("\n对比一下 k 的影响：")
    for kk in (1, 3, 6):
        c = "\n\n".join(d.page_content for d in vectorstore.similarity_search(QUESTION, k=kk))
        print(f"   k={kk} → 上下文 {len(c):6d} 字符")

    # 反例：问一个文档里根本没写的东西
    # 自适应点评：看 top1 是不是真的命中了「例子」那一段
    top1_mark = [w for w in ("餐馆", "广告", "挖油") if w in hits[0][0].page_content]
    print()
    if len(top1_mark) >= 2:
        print("↑ top-1 就命中了讲例子的段落，检索质量不错。")
        print("  块小的时候每个向量只代表一个主题，检索更『准』。")
    else:
        print("↑ 注意：讲例子的那段没有排第一，排第一的是开头总述。")
        print(f"  当前块很大（每块约 {sum(len(c.page_content) for c in chunks)//len(chunks) if chunks else 0} 字符），")
        print("  一块里混了很多主题，向量被『稀释』，相似度区分度就下来了。")
        print("  试试 python3 lab_walkthrough.py 3 --chunk-size 500 对比一下。")

    print("\n反例：问一个 markdown 里没有的内容，看看检索会返回什么")
    off = "Transformer 的自注意力公式是什么？"
    off_hits = vectorstore.similarity_search_with_score(off, k=1)
    print(f"   『{off}』 top1 score={off_hits[0][1]:.4f}")
    print("   → 分数并没有低到能被自动过滤，所以『答不上来』这件事只能靠 prompt 约束。")
    return context


# ------------------------------------------------------------------ 阶段 4

def stage4_generate(context):
    banner("阶段 4 · 生成：context + question 填进 prompt，交给 LLM")

    final_prompt = PROMPT.format(question=QUESTION, context=context)
    print(f"① 最终 prompt 长度：{len(final_prompt)} 字符")
    print("---- prompt 头 300 字符 ----")
    print(final_prompt[:300] + "\n…")
    print("---- prompt 尾 120 字符 ----")
    print("…" + final_prompt[-120:])

    llm = ChatOpenAI(temperature=0.7, max_tokens=4096, **llm_kwargs())
    print(f"\n② 调用 LLM：{llm_kwargs()['base_url']}  model={llm_kwargs()['model']}")
    resp = llm.invoke(final_prompt)

    print("\n③ 回答（.content）")
    print("-" * 78)
    print(resp.content)
    print("-" * 78)
    usage = resp.response_metadata.get("token_usage", {})
    print(f"\n④ 元数据：prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}")
    print("   原始对象还带 additional_kwargs / response_metadata / id 等字段，")
    print("   只想看答案时取 .content 即可（这就是课后练习 1 的做法）。")


def main():
    ap = argparse.ArgumentParser(description="C1 四步拆解实验")
    ap.add_argument("stage", nargs="?", type=int, choices=[1, 2, 3, 4], help="只跑某个阶段")
    ap.add_argument("--k", type=int, default=3, help="检索返回几块，默认 3")
    ap.add_argument("--chunk-size", type=int, default=4000,
                    help="分块大小，默认 4000（脚本 01 用的就是这个默认值）")
    args = ap.parse_args()

    chunks = None
    store = None
    ctx = None

    if args.stage in (None, 1):
        chunks = stage1_load_and_chunk(args.chunk_size)

    if args.stage in (None, 2):
        chunks = chunks or stage1_load_and_chunk(args.chunk_size)
        _, store = stage2_index(chunks)

    if args.stage in (None, 3):
        if store is None:
            chunks = chunks or stage1_load_and_chunk(args.chunk_size)
            _, store = stage2_index(chunks)
        ctx = stage3_retrieve(store, args.k, chunks)

    if args.stage in (None, 4):
        if ctx is None:  # 全跑时复用阶段 3 的结果，避免重复检索
            if store is None:
                chunks = chunks or stage1_load_and_chunk(args.chunk_size)
                _, store = stage2_index(chunks)
            ctx = stage3_retrieve(store, args.k, chunks)
        stage4_generate(ctx)


if __name__ == "__main__":
    main()
