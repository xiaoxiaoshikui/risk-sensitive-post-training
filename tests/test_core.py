import numpy as np, pytest
from rsp.risk import RiskConfig, group_advantages, batch_advantages, cvar, value_at_risk
from rsp.rewards import gsm8k_reward, extract_answer
from rsp.metrics import evaluate_rollouts, paired_bootstrap

def test_mean_estimator_is_zero_centred():
    r = np.array([0., 0., 1., 1.])
    a = group_advantages(r, RiskConfig("mean"))
    assert abs(a.sum()) < 1e-9

def test_degenerate_group_returns_zero():
    r = np.ones(8)
    assert np.allclose(group_advantages(r, RiskConfig("cvar")), 0.0)

def test_cvar_support_is_tail_only():
    r = np.array([0., 0., 1., 1., 1., 1., 1., 1.])
    a = group_advantages(r, RiskConfig("cvar", alpha=0.25, normalize=False))
    assert np.all(a[r > 0] == 0.0)          # non-tail rollouts get no signal
    assert np.all(a[r == 0] <= 0.0)         # tail rollouts are pushed down

def test_cvar_lambda_interpolates_to_mean():
    r = np.array([0., 0., 1., 1.])
    m = group_advantages(r, RiskConfig("mean"))
    c0 = group_advantages(r, RiskConfig("cvar", lambda_risk=0.0))
    assert np.allclose(m, c0)

def test_entropic_recovers_mean_as_beta_vanishes():
    r = np.array([0., 0.3, 1., 0.7])
    m = group_advantages(r, RiskConfig("mean"))
    e = group_advantages(r, RiskConfig("entropic", beta=1e-9))
    assert np.allclose(m, e, atol=1e-6)

def test_entropic_baseline_is_below_mean():
    # Risk-tilted baseline weights bad outcomes more, so it sits below the mean,
    # which makes every rollout look relatively better than under plain GRPO.
    r = np.array([0., 0., 1., 1.])
    e = group_advantages(r, RiskConfig("entropic", beta=2.0, normalize=False))
    assert e.sum() > 0

def test_alpha_too_small_for_group_is_rejected():
    with pytest.raises(ValueError):
        batch_advantages(np.zeros((2, 4)) + np.arange(4), RiskConfig("cvar", alpha=0.1))

def test_cvar_functional():
    r = np.array([0., 1., 2., 3.])
    assert cvar(r, 0.5) == 0.5
    assert value_at_risk(r, 0.5) == 1.0

def test_reward_requires_exactly_one_answer_tag():
    assert extract_answer("<answer>42</answer>") == ("42", True)
    assert extract_answer("<answer>1</answer><answer>2</answer>")[1] is False
    assert extract_answer("the answer is 42")[1] is False

def test_reward_scoring():
    b = gsm8k_reward("<answer>72</answer>", "blah #### 72")
    assert b.correct and b.format_ok and abs(b.reward - 1.1) < 1e-9
    b2 = gsm8k_reward("72", "#### 72")
    assert not b2.format_ok and b2.reward == 0.0   # correct number, contract broken
    b3 = gsm8k_reward("<answer>71</answer>", "#### 72")
    assert not b3.correct and b3.format_ok

def test_number_canonicalisation():
    assert gsm8k_reward("<answer>1,200.0</answer>", "#### 1200").correct

def test_tail_report_separates_mean_from_tail():
    # Two policies, identical mean, very different tails.
    flat = np.tile(np.array([1, 1, 0, 0]), (20, 1))
    split = np.concatenate([np.ones((10, 4)), np.zeros((10, 4))])
    a, b = evaluate_rollouts(flat), evaluate_rollouts(split)
    assert abs(a.mean_acc - b.mean_acc) < 1e-9
    assert a.worst_decile > b.worst_decile
    assert b.zero_solve_rate == 0.5 and a.zero_solve_rate == 0.0

def test_paired_bootstrap_detects_no_difference():
    x = np.random.default_rng(0).random(200)
    out = paired_bootstrap(x, x.copy(), np.mean)
    assert abs(out["delta"]) < 1e-9 and out["p_value"] > 0.5
