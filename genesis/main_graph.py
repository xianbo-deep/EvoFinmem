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

    # 节点
    enabled = set(cfg.get("graph",{}).get("enabled_nodes", []))
    parallel = bool(cfg.get("graph",{}).get("parallel_analysts", False))
    ver = cfg.get("graph", {}).get("verifiers", {})

    # 加载记忆
    g.add_node("memory_loader", memory_loader_node)
    # 分析
    g.add_node("fundamental",fundamental_analyst_node)
    g.add_node("sentiment",sentiment_analyst_node)
    g.add_node("technical", technical_analyst_node)

    # 决策
    g.add_node("trader",trader_node)
    g.add_node("risk",risk_manager_node)
    g.add_node("execute",execute_node)

    # 迭代
    g.add_node("gradient",gradient_node)
    g.add_node("update",update_node)

    # 边
    g.add_edge(START, "memory_loader")
    g.add_edge("memory_loader", "fundamental")
    g.add_edge("fundamental", "sentiment")
    g.add_edge("sentiment", "technical")
    g.add_edge("technical", "trader")

    g.add_edge("trader","risk")

    # 路由边：是否撤回交易
    g.add_conditional_edges("risk", route_after_risk, {"execute": "execute", END: END})


    # 路由边：是否更新架构
    g.add_conditional_edges("execute",route_after_execute,{"gradient": "gradient", END: END})

    g.add_edge("gradient", "update")
    g.add_edge("update", END)

    return g.compile()

# 初始化大脑 后需要改的
def build_brain(symbol: str) -> BrainDB:
    cfg = {
        "general": {
            "agent_name": "company_agent",
            "trading_symbol": symbol,
        },
        "agent": {
            "agent_1": {
                "embedding": {
                    "detail": {
                        # 这里要与你 embedding.py 需要的参数对齐
                        # 你的 embedding.py 读 OPENAI_API_KEY，所以最少也要确保环境变量存在
                        "model": "text-embedding-ada-002",
                    }
                }
            }
        },
        "short": {
            "jump_threshold_upper": 80,
            "importance_score_initialization": "sample",  # 依据你的 importance_score.py 支持的 type
            "decay_params": {"recency_factor": 5.0, "importance_factor": 0.97},
            "clean_up_threshold_dict": {"recency_threshold": -1e9, "importance_threshold": -1e9},
        },
        "mid": {
            "jump_threshold_upper": 90,
            "jump_threshold_lower": 20,
            "importance_score_initialization": "sample",
            "decay_params": {"recency_factor": 10.0, "importance_factor": 0.985},
            "clean_up_threshold_dict": {"recency_threshold": -1e9, "importance_threshold": -1e9},
        },
        "long": {
            "jump_threshold_lower": 30,
            "importance_score_initialization": "sample",
            "decay_params": {"recency_factor": 30.0, "importance_factor": 0.995},
            "clean_up_threshold_dict": {"recency_threshold": -1e9, "importance_threshold": -1e9},
        },
        "reflection": {
            "importance_score_initialization": "sample",
            "decay_params": {"recency_factor": 15.0, "importance_factor": 0.99},
            "clean_up_threshold_dict": {"recency_threshold": -1e9, "importance_threshold": -1e9},
        },
    }
    return BrainDB.from_config(cfg)


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