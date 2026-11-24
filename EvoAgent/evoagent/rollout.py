"""Deterministic canary and shadow assignment with automatic rollback."""
import hashlib
from typing import Dict, Optional


# 进化系统或人工会产出新版本的 llm-review prompt（candidate）。不能一次全量上线，需要：
# Canary：一小部分任务走新版本，其余走 stable
# Shadow：在后台对比 stable vs candidate 的结果（可选）
# 自动回滚：candidate 失败率过高 → 自动 rollback
# 自动晋升：shadow 验证通过 → 自动 promote（store 层逻辑）
# rollout.py 负责分配通道 + 记录结果 + 触发告警，具体计数和状态变更在 store.py 的 deployments 表。
class ReleaseManager:
    def __init__(self, store):
        self.store = store

    def configure(self, tenant_id: str, skill_name: str, config: Dict[str, object]) -> dict:
        canary = int(config.get("canary_percent", 0))
        shadow = int(config.get("shadow_percent", 0))
        if not 0 <= canary <= 100 or not 0 <= shadow <= 100:
            raise ValueError("canary_percent and shadow_percent must be between 0 and 100")
        if config.get("candidate_version") is None:
            raise ValueError("candidate_version is required")
        self.store.save_deployment(tenant_id, skill_name, config)
        return self.store.get_deployment(tenant_id, skill_name)

    def assignment(self, tenant_id: str, skill_name: str, key: str) -> Dict[str, object]:
        deployment = self.store.get_deployment(tenant_id, skill_name)
        if not deployment or deployment["status"] != "running":
            return {"lane": "stable", "shadow": False, "deployment": None}
        bucket = int(hashlib.sha256(
            ("%s:%s:%s" % (tenant_id, skill_name, key)).encode("utf-8")
        ).hexdigest()[:8], 16) % 100
        return {
            "lane": "canary" if bucket < deployment["canary_percent"] else "stable",
            "shadow": bucket < deployment["shadow_percent"],
            "deployment": deployment,
        }

    def observe(
        self, tenant_id: str, skill_name: str, failed: bool,
        lane: str = "canary",
    ) -> Optional[dict]:
        if lane != "canary":
            return self.store.get_deployment(tenant_id, skill_name)
        result = self.store.record_deployment_result(tenant_id, skill_name, failed)
        if result and result["status"] == "rolled_back":
            self.store.create_alert(
                tenant_id, "rollout:%s" % skill_name, "critical",
                "Canary %s was automatically rolled back after exceeding its error budget." % skill_name,
            )
        return result

    def observe_shadow(
        self, tenant_id: str, skill_name: str, task_id: str, lane: str,
        primary: Dict[str, object], candidate: Optional[Dict[str, object]],
        candidate_failed: bool = False,
    ) -> Optional[dict]:
        primary_keys = set(primary.get("finding_keys", []))
        candidate_keys = set((candidate or {}).get("finding_keys", []))
        union = primary_keys | candidate_keys
        disagreement = len(primary_keys ^ candidate_keys) / len(union) if union else 0.0
        result = self.store.record_shadow_observation(
            tenant_id, skill_name, task_id, lane, primary, candidate,
            disagreement, candidate_failed,
        )
        if result and result["status"] == "promoted":
            self.store.create_alert(
                tenant_id, "rollout-promoted:%s" % skill_name, "info",
                "Candidate %s was automatically promoted after shadow verification." % skill_name,
            )
        return result
