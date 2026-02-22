from dataclasses import dataclass
from datetime import datetime,date
import os
from typing import TypedDict,Dict,Any,List
from langgraph.graph import StateGraph,START,END
import json
import logging
from memorydb import BrainDB, MemoryDB
from portfolio import Portfolio

import prompts

def call_llm_json(llm,prompt:str) -> Dict[str,Any]:
    out = llm(prompt)
    return out



# 转换日期
def ensure_date(d) -> date:
    if isinstance(d, date):
        return d
    # 兼容 "YYYY-MM-DD"
    return datetime.strptime(str(d), "%Y-%m-%d").date()



# 状态容器
class TradeState(TypedDict):
    _llm: Any
    _portfolio: Portfolio
    _brain: BrainDB

    # 日期
    cur_date: str
    cur_date_obj: date
    # 股票名称
    symbol: str
    # 市场信息
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
    dynamic_results: Dict[str, Any]

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

# 加载记忆
def memory_loader_node(state: TradeState) -> Dict[str,Any]:
    d = ensure_date(state["cur_date"])
    state["cur_date_obj"] = d

    symbol = state["symbol"]
    market = state["market_info"]
    query_text = f"{symbol} | {d.isoformat()} | market:{market}"

    graph = state.get("workflow_config", {}).get("graph", {})
    params = graph.get("nodes", {}).get("memory_loader", {}).get("params", {})
    k_short = params.get("topk_short", 5)
    k_mid = params.get("topk_mid", 3)
    k_long = params.get("topk_long", 3)
    k_ref = params.get("topk_reflection", 3)

    brain = state["_brain"]
    short, sid = brain.query_short(query_text, k_short, symbol)
    mid, midid = brain.query_mid(query_text, k_mid, symbol)
    long, lid = brain.query_long(query_text, k_long, symbol)
    ref, rid = brain.query_reflection(query_text, k_ref, symbol)

    return {
        "cur_date_obj": d,
        "short_memory": short, "short_memory_id": sid,
        "mid_memory": mid, "mid_memory_id": midid,
        "long_memory": long, "long_memory_id": lid,
        "reflection_memory": ref, "reflection_memory_id": rid,
        # 为 EvoMAC 审计准备：记录“今天检索命中的所有 ids”
        "hit_memory_ids": list(set(sid + midid + lid + rid)),
    }
# 交易员
def trader_node(state: TradeState) -> Dict[str,Any]:
    # TODO 获取分析师报告
    reports_dict = state.get("dynamic_results", {})
    reports_str = ""
    for analyst_name, report_content in reports_dict.items():
        reports_str += f"\n【{analyst_name}分析师报告】:\n{json.dumps(report_content, ensure_ascii=False)}\n"
    if not reports_str:
        reports_str = "今日无任何分析师报告。"

    # 填充模板
    cfg = state.get("workflow_config", {}).get("graph", {}).get("nodes", {}).get("trader", {})
    prompt_template = cfg.get("prompt", "你是交易员。请根据以下报告做出买卖决策：\n{reports}\n请给出 JSON。")
    prompt = prompt_template.format(
        symbol=state["symbol"],
        cur_date=state["cur_date"],
        portfolio=state.get("portfolio_snapshot", {}),
        reports=reports_str,
        reflection_memory=state.get("reflection_memory", [])
    )

    decision = call_llm_json(state["_llm"], prompt)
    reason_text = decision.get("reason", "")
    if reason_text and "_brain" in state:
        # BrainDB 要求的 date 是 datetime.date 对象，可能需要转换一下字符串
        cur_date_obj = state.get("cur_date_obj") or ensure_date(state["cur_date"])

        state["_brain"].add_memory_reflection(
            symbol=state["symbol"],
            date=cur_date_obj,
            text=reason_text
        )
    new_trace = state.get("trace", []) + [{"agent": "trader", "output": decision}]
    return {"trader_decision": decision,"trace": new_trace}


# 风险管理
def risk_manager_node(state: TradeState) -> Dict[str,Any]:
    cfg = state.get("workflow_config",{})
    risk_limits = cfg.get("risk_limits",{"max_position_frac": 0.3, "max_drawdown": 0.1})

    prompt = prompts.RISK_MANAGER_PROMPT.format(
        symbol=state["symbol"],
        cur_date=state["cur_date"],
        order=state.get("trader_decision", {}),
        portfolio=state.get("portfolio_snapshot", {}),
        risk_limits=risk_limits,
    )

    rep = call_llm_json(state["_llm"], prompt)
    approved = bool(rep.get("approved", True))
    reason = rep.get("reason", "")

    new_trace = state.get("trace", []) +  [{"agent": "risk", "output": rep}]
    return {"risk_approved": approved, "risk_reason": reason,"trace": new_trace}

# 执行
def execute_node(state: TradeState) -> Dict[str,Any]:
    _, cur_price, *_ = state["market_info"]
    port: Portfolio = state["_portfolio"]
    cur_date = state["cur_date"]

    # 更新当前价格到钱包
    port.update_market_info(cur_price, cur_date)


    # 获取动作
    decision = state.get("trader_decision",{})

    # 风控
    if not state.get("risk_approved", False):
        decision = {"action": "hold", "position_frac": 0.0}

    action = decision.get("action","hold")
    direction = 1 if action == "buy" else (-1 if action == "sell" else 0)

    # 交易，改变持仓
    port.record_action({"direction":direction})

    # 每日快照
    port.update_portfolio_series()



    # 命中的记忆ID
    hit_ids = state.get("hit_memory_ids", [])

    # 存入历史账本
    memory_history = dict(state.get("memory_history", {}))
    memory_history[cur_date] = hit_ids

    # 盈亏率，前n天
    fb = port.get_feedback_response()
    pnl_feedback = 0.0

    if fb:
        pnl_feedback = float(fb["feedback"])
        target_date = fb.get("date")

        if target_date and target_date in memory_history and pnl_feedback != 0:
            past_hit_ids = memory_history[target_date]

            if past_hit_ids:
                # 给大功臣加分，或者让背锅侠扣分
                state["_brain"].update_access_count_with_feed_back(
                    symbol=state["symbol"],
                    ids=past_hit_ids,
                    feedback=int(pnl_feedback)
                )



    # 快照
    snapshot = {
        "holding_shares": port.holding_shares, # 持仓份额
        "market_price": port.market_price,     # 市场标价
        "day_count": port.day_count,           # 运行天数
    }

    # 衰减
    state["_brain"].step()

    new_trace = state.get("trace", []) + [{"agent": "execute", "action": action, "direction": direction, "feedback": fb}]
    return {"pnl_feedback": pnl_feedback, "portfolio_snapshot": snapshot,"memory_history": memory_history,"trace": new_trace}

# 文本梯度
def gradient_node(state: TradeState) -> Dict[str,Any]:
    prompt = prompts.GRADIENT_AUDITOR_PROMPT.format(
        cur_date=state["cur_date"],
        symbol=state["symbol"],
        pnl_feedback=state.get("pnl_feedback", 0.0),
        trace=state.get("trace", []),
    )
    diag = call_llm_json(state["_llm"], prompt)
    new_trace = state.get("trace",[]) + [{"agent": "gradient", "output": diag}]
    return {"error_trace": diag,"trace":new_trace}


# 更新
def update_node(state: TradeState) -> Dict[str,Any]:
    prompt = prompts.UPDATING_TRADING_AGENT.format(
        error_trace=state.get("error_trace", {}),
        workflow_config=state.get("workflow_config", {}),
    )
    upd = call_llm_json(state["_llm"],prompt)
    new_cfg = upd.get("new_config", state.get("workflow_config", {}))
    new_trace = state.get("trace", []) + [{"agent": "update", "output": upd}]
    return {"workflow_config": new_cfg,"trace":new_trace}

# 路由边
def route_after_risk(state: TradeState) -> str:
    if state.get("risk_approved",True):
        return "execute"
    return END


def route_after_execute(state: TradeState) -> str:
    # 亏损触发进化
    if state.get("pnl_feedback", 0.0) < 0:
        return "gradient"
    return END


def create_dynamic_node(node_name: str):
    def universal_node(state: TradeState) -> Dict[str, Any]:
        node_cfg = state["workflow_config"]["graph"]["nodes"].get(node_name, {})

        # 动态获取它需要吃什么数据
        deps = node_cfg.get("data_dependencies", [])
        format_kwargs = {}
        for key in deps:
            # 兼容性寻找：先去 state 第一层找，找不到再去 dynamic_results 里找
            if key in state:
                format_kwargs[key] = state[key]
            elif key in state.get("dynamic_results", {}):
                format_kwargs[key] = state["dynamic_results"][key]
            else:
                format_kwargs[key] = ""

        # Prompt
        prompt_dict = node_cfg.get("prompt", {})
        try:
            role = prompt_dict.get("role", "").format(**format_kwargs)
            task = prompt_dict.get("task", "").format(**format_kwargs)
            constraints_str = "\n".join([f"- {c}" for c in prompt_dict.get("constraints", [])])
            final_prompt = f"# Role\n{role}\n\n# Task\n{task}\n\n# Constraints\n{constraints_str}"
        except KeyError as e:
            logging.error(f"节点 {node_name} 填空失败: {e}")
            return {"dynamic_results": state.get("dynamic_results", {})}


        rep = call_llm_json(state["_llm"], final_prompt)

        #  记忆处理
        write_tiers = node_cfg.get("memory_write_tiers", [])
        if write_tiers and "_brain" in state and rep:
            text_to_save = rep.get("output", json.dumps(rep, ensure_ascii=False))
            d = state.get("cur_date_obj") or ensure_date(state["cur_date"])
            for tier in write_tiers:
                if tier == "short": state["_brain"].add_memory_short(state["symbol"], d, text_to_save)

        # 把任何输出，存进动态字典
        current_results = state.get("dynamic_results", {})
        current_results[node_name] = rep

        return {"dynamic_results": current_results, "trace": state.get("trace", []) + [{"agent": node_name}]}

    universal_node.__name__ = f"{node_name}_node"
    return universal_node

