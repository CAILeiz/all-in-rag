"""C4-03 · Text2SQL 演示（自然语言 → SQL → 查库）

对应文档：docs/chapter4/03_text2sql.md

要点：不是所有问题都该走向量检索。「统计每个城市的用户数量」这类需求，
正确答案是一句 SQL，而不是一段语义相似的文本。

    LLM 负责：自然语言 → SQL
    数据库负责：执行 SQL、返回精确结果

这里用 sqlite 造了一个电商小库（users / products / orders），
让 agent 先看表结构（schema）再生成 SQL。

运行（走 Ollama，不需要联网）：
    python3 code/C4/03_text2sql_demo.py
    RAG_LOG=quiet python3 code/C4/03_text2sql_demo.py

4 个阶段：
    [1/4] 建演示数据库（3 张表 + 19 行数据）
    [2/4] 初始化 Text2SQL Agent（连库 + 加载知识库）
    [3/4] 跑 6 个自然语言问题
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
import sys
import sqlite3

# 添加text2sql模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))

from stage_log import Stage
from text2sql.text2sql_agent import SimpleText2SQLAgent


def setup_demo():
    """设置演示环境：建库 → 初始化 agent → 连库 → 加载知识库。"""
    # 检查 API 密钥（llm_config 已从 code/.env 读好，这里只做存在性校验）
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None, None, "缺少 DEEPSEEK_API_KEY / LLM_API_KEY"

    db_path = create_demo_database()

    agent = SimpleText2SQLAgent(api_key=api_key)

    if not agent.connect_database(db_path):
        return None, None, "数据库连接失败"

    try:
        agent.load_knowledge_base()
    except Exception as e:
        return None, None, f"知识库加载失败: {e}"

    return agent, db_path, None


def create_demo_database():
    """创建演示数据库"""
    db_path = "text2sql_demo.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            city TEXT
        )
    """)
    
    # 创建产品表
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)
    
    # 创建订单表
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            total_price REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    
    # 插入示例数据
    users_data = [
        (1, '张三', 'zhangsan@email.com', 25, '北京'),
        (2, '李四', 'lisi@email.com', 32, '上海'),
        (3, '王五', 'wangwu@email.com', 28, '广州'),
        (4, '赵六', 'zhaoliu@email.com', 35, '深圳'),
        (5, '陈七', 'chenqi@email.com', 29, '杭州'),
    ]
    
    products_data = [
        (1, 'iPhone 15', '电子产品', 7999.0, 50),
        (2, 'MacBook Pro', '电子产品', 12999.0, 20),
        (3, 'Nike运动鞋', '服装', 599.0, 100),
        (4, '办公椅', '家具', 899.0, 30),
        (5, '台灯', '家具', 199.0, 80),
        (6, 'iPad', '电子产品', 3999.0, 40),
        (7, 'Adidas外套', '服装', 399.0, 60),
    ]
    
    orders_data = [
        (1, 1, 1, 1, '2024-01-15', 7999.0),
        (2, 2, 3, 2, '2024-01-16', 1198.0),
        (3, 3, 5, 1, '2024-01-17', 199.0),
        (4, 1, 2, 1, '2024-01-18', 12999.0),
        (5, 4, 4, 1, '2024-01-19', 899.0),
        (6, 5, 6, 1, '2024-01-20', 3999.0),
        (7, 2, 7, 1, '2024-01-21', 399.0),
    ]
    
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users_data)
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders_data)
    
    conn.commit()
    conn.close()
    
    return db_path


def run_demo_queries(agent, step):
    """运行演示查询，把过程写进 stage 日志。"""
    demo_questions = [
        "查询所有用户的姓名和邮箱",
        "年龄大于30的用户有哪些",
        "哪些产品的库存少于50",
        "查询来自北京的用户的所有订单",
        "统计每个城市的用户数量",
        "查询价格在500-8000之间的产品"
    ]
    
    success_count = 0

    for i, question in enumerate(demo_questions, 1):
        step.detail("")
        step.detail(f"问题 {i}: {question}")

        try:
            result = agent.query(question)

            if result["success"]:
                sql_oneline = " ".join(result["sql"].split())
                step.detail(f"  SQL：{sql_oneline[:110]}")

                if isinstance(result["results"], dict) and "rows" in result["results"]:
                    count = result["results"]["count"]
                    step.detail(f"  返回 {count} 行")
                    for j, row in enumerate(result["results"]["rows"][:2], 1):
                        row_str = " | ".join(f"{k}={v}" for k, v in row.items())
                        step.detail(f"    {j}. {row_str}")
                    if count > 2:
                        step.detail(f"    ... 还有 {count - 2} 行")
                else:
                    step.detail(f"  结果：{result['results']}")

                success_count += 1
            else:
                step.detail(f"  失败：{result['error']}")
                step.detail(f"  SQL：{' '.join(str(result['sql']).split())[:110]}")

        except Exception as e:
            step.detail(f"  执行错误：{e}")

    return success_count, len(demo_questions)


def cleanup(agent, db_path, step):
    """清理资源。"""
    if agent:
        agent.cleanup()
        step.detail("agent 已关闭")
    if db_path and os.path.exists(db_path):
        os.remove(db_path)
        step.detail(f"已删除演示数据库：{db_path}")


def main():
    """主函数：4 个阶段串起来。"""
    stage = Stage(total=4, title="C4 · Text2SQL（自然语言 → SQL）")
    stage.header("结构化问题用 SQL 更准，不用向量检索")

    with stage.step(
        "建演示数据库",
        tables="users / products / orders",
        why="schema 越清晰，LLM 生成的 SQL 越准",
    ) as step:
        db_path = create_demo_database()
        step.detail("users    5 行（id/name/email/age/city）")
        step.detail("products 7 行（id/name/category/price/stock）")
        step.detail("orders   7 行（id/user_id/product_id/quantity/order_date/total_price）")
        step.detail("orders 有两个外键指向 users/products")
        step.note(f"{db_path}")

    with stage.step(
        "初始化 Agent + 加载知识库",
        why="知识库给 LLM 补充表关系、字段含义等上下文，减少瞎猜字段名",
    ) as step:
        agent, db_path, err = setup_demo()
        if err:
            step.detail(f"初始化失败：{err}")
            step.note("已中止")
            stage.summary()
            raise SystemExit(1)
        step.detail("已连接 sqlite，知识库已加载")
        step.note("就绪")

    try:
        with stage.step(
            "跑演示查询",
            why="注意 SQL 里的 GROUP BY / 区间条件 —— 这些向量检索做不到",
        ) as step:
            ok, total = run_demo_queries(agent, step)
            step.note(f"{ok}/{total} 成功")
    finally:
        with stage.step("清理", why="删掉 demo 库，保证下次从干净状态开始") as step:
            cleanup(agent, db_path, step)
            step.note("已清理")

    stage.summary()

    print()
    print("=" * 72)
    print("  观察点")
    print("=" * 72)
    print("1. 「统计每个城市的用户数量」→ SQL 的 GROUP BY，向量检索无法胜任")
    print("2. agent 会先读 schema，再生成 SQL；schema 描述越全，SQL 越准")
    print("3. 生成 SQL 后必须校验：只允许 SELECT，禁止 DROP/DELETE/UPDATE")
    print("4. 生产里通常还要加 LIMIT、超时、只读账号，防止慢查询打挂库")
    print()
    print("提示：LLM 偶尔会写错字段名，报错信息本身就是下一次重试的输入")
    print("=" * 72)


if __name__ == "__main__":
    main() 