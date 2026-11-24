# MultiEvo
多体智能动态协作和Agentic RL部分待更新。面向研发过程PR的风险治理与安全修复场景，构建自进化Multi-Agent代码审查系统，完成风险发现、证据复核、安全修复、结果验证与自动评测，并通过反馈驱动Prompt/Skill受控自进化。

技术栈：Python、Harness、Agent Runtime、Multi-Agent、Redis Streams、PostgreSQL、SQLite、Prometheus。
主要工作：
1. 高可用Runtime Harness：自研有界Runtime Harness，统一管理任务状态，记录全链路Trace；在容错与控制上实现Checkpoint、死	信队列、断点续跑等；观测记录Agent交互消息与状态流转生成全链路Trace，支持执行链路回放、评测复现与细粒度失败归因。
2. Multi-Agent协作：设计Lead主导的「委派-并行执行-盲审-返工综合」四层协作；Security与Correctness/Reliability 在最小特权工具	白名单下并行执行有界Tool Loop；Critic在隐藏候选来源后独立质疑，Lead将revision request下发返工，有效降低单Agent幻觉与误判	风险。
3. 反馈驱动的自进化：将误报/漏报等回流为结构化反馈，自动聚类失败轨迹并演进Prompt/Skill候选版本，支持去重、回滚；搭建含100个	受控PR Diff的Evaluation Harness基准集，按仓库隔离Validation、Holdout集合，通过提升与非退化门禁后自动激活。端到端评测发现，	演进后多Agent风险检测F1从81.9%提升至91.3%，高风险召回率从84.2%提升至94.7%，干净PR准确率达91.7%。
4. 门禁与安全设计：构建「格式-证据-置信度-发布」四重门禁过滤；高危缺陷要求 AST/扫描/调用链实证，并附带修复与测试方案；Skill自	进化采用声明式 SKILL.md，从架构上避免反馈注入导致的任意代码执行风险。
5. 上下文压缩与Memory管理：按风险优先级对超大Diff进行自适应压缩，在Token预算内动态组织工具描述、历史Observation与Critic	反馈；构建「工作-情景-语义」三层记忆体系，打通长短期记忆，支持按租户/仓库隔离检索、任务归档及过期清理。
