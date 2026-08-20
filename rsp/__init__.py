"""Risk-sensitive post-training: SFT, DPO and risk-aware GRPO."""

from rsp.metrics import TailReport, evaluate_rollouts, paired_bootstrap
from rsp.rewards import RewardBreakdown, extract_answer, gsm8k_reward
from rsp.risk import RiskConfig, batch_advantages, cvar, group_advantages

__version__ = "0.1.0"
