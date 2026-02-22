from datetime import datetime,date
import os
from typing import TypedDict,Dict,Any,List
from langgraph.graph import StateGraph,START,END
import json

from genesis.checkpoint import CheckpointManager
from memorydb import BrainDB, MemoryDB
from portfolio import Portfolio
from environment import MarketEnvironment
from chat import ChatOpenAICompatible
from reflection import trading_reflection
import prompts
from agent import *




# 状态容器
class TradeState(TypedDict):
    _llm: Any
    _portfolio: Portfolio
    _brain: BrainDB


    cur_date: str
    cur_date_obj: date
    symbol: str
    market_info: tuple

    # 记忆
    short_memory: List[str]
    short_memory_id: List[int]

    mid_memory: List[str]
    mid_memory_id: List[int]

    long_memory: List[str]
    long_memory_id: List[int]

    reflection_memory: List[str]
    reflection_memory_id: List[int]

    # 记忆ID的历史记录
    memory_history: Dict[str, List[int]]

    # 命中的记忆
    hit_memory_ids: List[int]

    # 分析师报告
    fundamental_report: Dict[str, Any]
    sentiment_report: Dict[str, Any]
    technical_report: Dict[str, Any]

    # 决策与执行
    trader_decision: Dict[str, Any] # 包括决策和原因
    risk_approved: bool # 是否放行
    risk_reason: str # 风险原因


    # 进化反馈（亏损就被激活）
    pnl_feedback: float # 盈亏反馈
    portfolio_snapshot: Dict[str, Any] # 钱包快照

    trace: List[Dict[str,Any]] # 日志

    error_trace: Dict[str, Any] # 文本梯度
    workflow_config: Dict[str, Any]

# 转换日期
def ensure_date(d) -> date:
    if isinstance(d, date):
        return d
    # 兼容 "YYYY-MM-DD"
    return datetime.strptime(str(d), "%Y-%m-%d").date()

# 调用API
def call_llm_json(llm,prompt:str) -> Dict[str,Any]:
    out = llm(prompt)
    return out



# 建立图
def build_graph(cfg: Dict[str,Any]) -> Any:
    g = StateGraph(TradeState)

    # 读取配置
    graph_cfg = cfg.get("graph", {})
    nodes_cfg = graph_cfg.get("nodes", {})
    edges_cfg = graph_cfg.get("edges", [])
    entry_node = graph_cfg.get("entry", "memory_loader")

    # 注册启用节点
    active_nodes = set()

    EXECUTIVE_NODES = {
        "memory_loader": memory_loader_node,
        "trader": trader_node,
        "risk": risk_manager_node,
        "execute": execute_node,
        "gradient": gradient_node,
        "update": update_node
    }

    

    for node_name, node_info in nodes_cfg.items():
        if not node_info.get("enabled", False) and node_name in EXECUTIVE_NODES:
            return False  # 动态剔除被禁用的节点
        
        if not node_info.get("enabled", False):
            continue

        else:
            # 呼叫工厂，动态生成分析师节点
            g.add_node(node_name, create_dynamic_node(node_name))

        active_nodes.add(node_name)
    # 边
    if entry_node in active_nodes:
        g.add_edge(START, entry_node)
    else:
        logging.error(f"严重错误：指定的入口节点 {entry_node} 未启用或不存在！")

    for source,target in edges_cfg:
        if source not in active_nodes:
            continue
        if target == "__ROUTE_AFTER_RISK__":
            g.add_conditional_edges(source, route_after_risk, {"execute": "execute", END: END})

        elif target == "__ROUTE_AFTER_EXECUTE__":
            g.add_conditional_edges(source, route_after_execute, {"gradient": "gradient", END: END})

        elif target == "__END__":
            g.add_edge(source, END)

        else:
            if target in active_nodes:
                g.add_edge(source, target)
            else:
                logging.warning(f"警告：边配置中的目标节点 {target} 未启用或不存在，已跳过该边连接！")

    return g.compile()

# TODO 初始化大脑 后需要改的
def build_brain(cfg: Dict[str,Any],symbol: str) -> BrainDB:
    config = cfg.get("brain", {})
    return BrainDB.from_config(config)

# 验证拓扑结构
def validate_graph_topology(cfg: Dict[str,Any]) -> bool:
    graph_cfg = cfg.get("graph", {})
    nodes_cfg = graph_cfg.get("nodes", {})
    edges_cfg = graph_cfg.get("edges", [])
    entry = graph_cfg.get("entry", "memory_loader")

    # 获取所有存活的节点
    active_nodes = {name for name, info in nodes_cfg.items() if info.get("enabled", False)}

    # 必要节点被删除
    required_nodes = {"memory_loader", "trader", "risk", "execute", "gradient", "update"}
    if not required_nodes.issubset(active_nodes):
        logging.error("安全审查失败：大模型删除了致命的核心高管节点！")
        return False

    # 入口被删除
    if entry not in active_nodes:
        logging.error("安全审查失败：入口节点不存在或被禁用！")
        return False

    # 检查边的合法性
    adjacency_list = {node: [] for node in active_nodes}
    for source,target in edges_cfg:
        if source in active_nodes:
            target_clean = target if not target.startswith("__") else "MAGIC_ROUTE"
            adjacency_list[source].append(target_clean)

    # DFS
    visited = set()
    def dfs(current_node):
        if current_node == "trader":
            return True
        if current_node == "MAGIC_ROUTE":
            return False

        visited.add(current_node)
        for neighbor in adjacency_list.get(current_node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        return False

    if not dfs(entry):
        logging.error("安全审查失败：无法从入口节点访问到交易员节点，存在安全风险！")
        return False

    return True

# 每日循环
def run_one_epoch(brain: BrainDB,env: MarketEnvironment,symbol: str,workflow_config: Dict[str,Any]) -> Dict[str,Any]:
    llm = ChatOpenAICompatible(
        end_point="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        model="glm-4",
        other_parameters={"temperature": 0.1}
    ).guardrail_endpoint()

    app = build_graph()
    env.reset()
    trace_all=[]
    cfg=workflow_config
    memory_history = {}

    portfolio = Portfolio(symbol = symbol,lookback_window_size=workflow_config.get("lookback_window_size", 5))
    # 交易循环
    while True:
        market_info = env.step()
        cur_date = market_info[0]
        cur_date_obj = ensure_date(cur_date)
        terminated = market_info[-1]

        state: TradeState = {
            "_llm": llm,
            "_brain":brain,
            "_portfolio": portfolio,
            "symbol":symbol,
            "cur_date":cur_date,
            "cur_date_obj": cur_date_obj,
            "market_info":market_info,
            "short_memory": [], "short_memory_id": [],
            "mid_memory": [], "mid_memory_id": [],
            "long_memory": [], "long_memory_id": [],
            "reflection_memory": [], "reflection_memory_id": [],
            "memory_history": memory_history,
            "hit_memory_ids": [],
            "trace": [],
            "workflow_config":cfg,
            "fundamental_report": {},
            "sentiment_report": {},
            "technical_report": {},
            "trader_decision": {},
            "risk_approved": True,
            "risk_reason": "",
            "pnl_feedback": 0.0,
            "error_trace": {},
            "portfolio_snapshot": {},
        }
        out = app.invoke(state)
        cfg = out.get("workflow_config",cfg)
        memory_history = out.get("memory_history", memory_history)
        trace_all.append(out.get("trace",[]))

        if terminated:
            break

    return {"final_config":cfg,"trace":trace_all}


if __name__ == "__main__":
    ckpt = CheckpointManager("./checkpoints/run_001")

    brain, env, p_state, cfg, meta = ckpt.load_all()

    # 如果没有 checkpoint，就正常初始化
    symbol = "AAPL"
    if brain is None:
        brain = build_brain(symbol)
    if env is None:
        env = MarketEnvironment(env_data_pkl=..., start_date=..., end_date=..., symbol=symbol)
    result = run_one_epoch(brain=brain,env=env,symbol=,workflow_config=)