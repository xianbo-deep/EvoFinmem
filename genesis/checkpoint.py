import os,json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CheckpointPaths:
    root: str
    brain_dir: str
    env_path: str
    portfolio_path: str
    config_path: str
    latest_path: str

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def build_paths(root: str) -> CheckpointPaths:
    ensure_dir(root)
    brain_dir = os.path.join(root,"brain")
    ensure_dir(brain_dir)

    return CheckpointPaths(
        root=root,
        brain_dir=brain_dir,
        env_path=os.path.join(root,"env"),
        portfolio_path=os.path.join(root,"portfolio.json"),
        config_path=os.path.join(root, "config.json"),
        latest_path=os.path.join(root, "latest.json"),
    )

def save_json(path:str,obj:Any):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)

def load_json(path:str) -> Any:
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)


class CheckpointManager:
    def __init__(self,root:str):
        self.paths = build_paths(root)

    def save_all(
    self,
    *,
    brain,
    env,
    portfolio_state: Dict,
    workflow_config: Dict,
    meta: Optional[Dict] = None,
    force: bool = True,
    ):

        # brain
        brain.save_checkpoint(self.paths.brain_dir,force=force)

        # 环境市场
        env.save_checkpoint(self.paths.env_path)

        # 钱包
        save_json(self.paths.portfolio_path, portfolio_state)

        # 工作流配置
        save_json(self.paths.config_path,workflow_config)

        latest = {"ok": True, "meta": meta or {}}
        save_json(self.paths.latest_path, latest)

    def load_all(self, *, brain, env):
        if os.path.exists(self.paths.brain_dir):
            from memorydb import BrainDB
            brain = BrainDB.load_checkpoint(self.paths.brain_dir)
        if os.path.exists(self.paths.env_path):
            from environment import MarketEnvironment
            env = MarketEnvironment.load_checkpoint(self.paths.env_path)

        portfolio_state = None
        if os.path.exists(self.paths.portfolio_path):
            portfolio_state = load_json(self.paths.portfolio_path)

        workflow_config = None
        if os.path.exists(self.paths.config_path):
            workflow_config = load_json(self.paths.config_path)

        meta = None
        if os.path.exists(self.paths.latest_path):
            meta = load_json(self.paths.latest_path)

        return brain, env, portfolio_state, workflow_config, meta