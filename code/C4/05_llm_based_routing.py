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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableBranch

llm = ChatDeepSeek(temperature=0, **llm_kwargs())

stage = Stage(total=2, title="C4 · LLM 路由（按菜系分流）")
stage.header("分类器先判断菜系，再交给对应大厨链回答")

# 1. 设置不同菜系的处理链
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

cantonese_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain = cantonese_prompt | llm | StrOutputParser()

# 定义备用通用链
general_prompt = ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
general_chain = general_prompt | llm | StrOutputParser()


# 2. 创建路由链
classifier_prompt = ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 定义路由分支
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain  # 默认选项
)

# 组合成完整路由链
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch
print()
with stage.step(
    "构建路由链",
    classifier="LLM 分类为 ['川菜','粤菜','其他']",
    branches="川菜链 / 粤菜链 / 通用链（兜底）",
    why="RunnableBranch 按顺序匹配，第一个为 True 的分支胜出，最后一个是 default",
) as step:
    step.detail("川菜链  ：prompt 加「川菜大厨」人设")
    step.detail("粤菜链  ：prompt 加「粤菜大厨」人设")
    step.detail("通用链  ：不限定菜系，兜底")
    step.note("3 条分支")


# 3. 运行演示查询
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]

with stage.step(
    "演示查询",
    count=len(demo_questions),
    why="注意看每个问题的路由决策是否和预期一致（麻婆豆腐→川菜、白切鸡→粤菜）",
) as step:
    for i, item in enumerate(demo_questions, 1):
        question = item["question"]
        step.detail("")
        step.detail(f"--- 问题 {i}: {question} ---")
        try:
            topic = classifier_chain.invoke({"question": question})
            step.detail(f"路由决策：{topic}")

            result = full_router_chain.invoke(item)
            preview = str(result).strip().splitlines()
            for line in preview[:3]:
                step.detail(f"  回答：{line}")
        except Exception as e:
            step.detail(f"执行错误：{e}")
    step.note(f"{len(demo_questions)} 个问题")

stage.summary()

print()
print("=" * 72)
print("  观察点")
print("=" * 72)
print("1. 路由靠的是 LLM 分类，不是关键词匹配 —— 所以口径不统一时会误判")
print("2. 分类失败（返回不在列表里的词）会落到通用链，不会报错")
print("3. 这条链路是「一次分类 + 一次回答」= 2 次 LLM 调用，成本翻倍")
print("4. 生产里常把分类结果缓存，或改用小模型/规则先粗筛")
print()
print("提示：分类器温度设 0，是为了让同一个问题每次分级稳定")
print("=" * 72)

