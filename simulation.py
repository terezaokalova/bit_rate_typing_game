#!/usr/bin/env python3
"""
Optimal Alphabet Size (N*) Simulation for Keyboard-Based Bit Rate Game.

Sweeps alphabet size N to maximize expected Shenoy achieved bit rate,
averaged across a panel of three evaluators. Uses Monte Carlo sampling
of stochastic human agent parameters.

Usage:
    cd /Users/tereza/sciencecorp/maximizing_bit_rate_game
    python analysis/optimal_n.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Configuration
SEED = 80
N_RANGE = np.arange(3, 41)           # N = 3..40 inclusive
N_MC = 2000                           # Monte Carlo samples per N
SESSION_DURATION = 60.0               # seconds
DEFAULT_PIPELINE_FACTOR = 1.20
PIPELINE_FACTORS = [1.0, 1.10, 1.15, 1.20, 1.25]

# Plausibility clipping bounds
CLIP_P0 = (0.8, 1.0)
CLIP_A = (0.08, 0.5)
CLIP_B = (0.01, 0.3)
CLIP_C = (0.005, 0.08)

FIGURE_DIR = Path(__file__).resolve().parent / "figures"


# Agent profile dataclasses
@dataclass(frozen=True)
class ParamDist:
    """Normal distribution specification for a single agent parameter."""
    mu: float
    sigma: float


@dataclass(frozen=True)
class AgentProfile:
    """Stochastic agent profile with parameter distributions.

    Each parameter is Normal(mu, sigma). During simulation, parameters are
    sampled and clipped to physical plausibility bounds.
    """
    name: str
    color: str
    a: ParamDist    # base sensorimotor latency (s)
    b: ParamDist    # per-bit processing cost (s/bit)
    p0: ParamDist   # peak accuracy at N=2
    c: ParamDist    # linear accuracy decay per bit
    k: ParamDist    # logistic steepness
    x0: ParamDist   # logistic inflection point (bits)

    def sample(self, rng: np.random.Generator, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample all parameters, clipped to plausibility bounds."""
        return {
            "a":  np.clip(rng.normal(self.a.mu,  self.a.sigma,  n_samples), *CLIP_A),
            "b":  np.clip(rng.normal(self.b.mu,  self.b.sigma,  n_samples), *CLIP_B),
            "p0": np.clip(rng.normal(self.p0.mu, self.p0.sigma, n_samples), *CLIP_P0),
            "c":  np.clip(rng.normal(self.c.mu,  self.c.sigma,  n_samples), *CLIP_C),
            "k":  rng.normal(self.k.mu,  self.k.sigma,  n_samples),
            "x0": rng.normal(self.x0.mu, self.x0.sigma, n_samples),
        }


# Agent definitions
EMMA = AgentProfile(
    name="Emma", color="#E74C3C",
    a=ParamDist(0.250, 0.020), b=ParamDist(0.100, 0.015),
    p0=ParamDist(0.970, 0.010), c=ParamDist(0.032, 0.005),
    k=ParamDist(1.8, 0.3), x0=ParamDist(3.5, 0.3),
)
CALVIN = AgentProfile(
    name="Calvin", color="#3498DB",
    a=ParamDist(0.160, 0.015), b=ParamDist(0.055, 0.010),
    p0=ParamDist(0.960, 0.010), c=ParamDist(0.015, 0.004),
    k=ParamDist(1.2, 0.2), x0=ParamDist(4.2, 0.3),
)
ELIZABETH = AgentProfile(
    name="Elizabeth", color="#2ECC71",
    a=ParamDist(0.200, 0.015), b=ParamDist(0.075, 0.012),
    p0=ParamDist(0.970, 0.010), c=ParamDist(0.023, 0.004),
    k=ParamDist(1.5, 0.25), x0=ParamDist(3.8, 0.3),
)
TYPICAL = AgentProfile(
    name="Typical User", color="#95A5A6",
    a=ParamDist(0.230, 0.025), b=ParamDist(0.110, 0.020),
    p0=ParamDist(0.960, 0.015), c=ParamDist(0.035, 0.006),
    k=ParamDist(2.0, 0.3), x0=ParamDist(3.2, 0.3),
)
FAST_SLOPPY = AgentProfile(
    name="Fast-Sloppy", color="#E67E22",
    a=ParamDist(0.140, 0.015), b=ParamDist(0.045, 0.010),
    p0=ParamDist(0.940, 0.012), c=ParamDist(0.020, 0.005),
    k=ParamDist(1.3, 0.2), x0=ParamDist(3.8, 0.3),
)
SLOW_ACCURATE = AgentProfile(
    name="Slow-Accurate", color="#9B59B6",
    a=ParamDist(0.280, 0.020), b=ParamDist(0.120, 0.015),
    p0=ParamDist(0.985, 0.005), c=ParamDist(0.018, 0.004),
    k=ParamDist(1.0, 0.2), x0=ParamDist(4.5, 0.3),
)

PANEL_MEMBERS = [EMMA, CALVIN, ELIZABETH]
ALL_AGENTS = [EMMA, CALVIN, ELIZABETH, TYPICAL, FAST_SLOPPY, SLOW_ACCURATE]


# Core computation (vectorized across MC samples)
def selection_rate(a: np.ndarray, b: np.ndarray, n: int,
                   pipeline_factor: float = DEFAULT_PIPELINE_FACTOR) -> np.ndarray:
    """Pipelined selection rate via Hick's Law: R = pipeline / (a + b*log2(N))."""
    return pipeline_factor / (a + b * np.log2(n))


def accuracy_linear(p0: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Linear accuracy model: p(N) = clip(p0 - c*log2(N), 0.5, 1.0)."""
    return np.clip(p0 - c * np.log2(n), 0.5, 1.0)


def accuracy_logistic(p0: np.ndarray, k: np.ndarray, x0: np.ndarray,
                      n: int) -> np.ndarray:
    """Logistic accuracy model with N-dependent floor.

    p(N) = p_floor + (p0 - p_floor) / (1 + exp(k*(log2(N) - x0)))
    where p_floor = max(1/N, 0.05).
    """
    p_floor = max(1.0 / n, 0.05)
    logit = 1.0 / (1.0 + np.exp(k * (np.log2(n) - x0)))
    return np.clip(p_floor + (p0 - p_floor) * logit, 0.5, 1.0)


def bit_rate(n: int, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Shenoy achieved bit rate: B = log2(N-1) * R * max(2p-1, 0)."""
    return np.log2(n - 1) * r * np.maximum(2.0 * p - 1.0, 0.0)

# Simulation results container
@dataclass
class SimResult:
    """Simulation results for one agent across all N values."""
    agent_name: str
    n_values: np.ndarray
    # Linear accuracy model
    b_mean_lin: np.ndarray
    b_p5_lin: np.ndarray
    b_p95_lin: np.ndarray
    n_star_lin: int
    peak_b_lin: float
    p_at_nstar_lin: float
    # Logistic accuracy model
    b_mean_log: np.ndarray
    b_p5_log: np.ndarray
    b_p95_log: np.ndarray
    n_star_log: int
    peak_b_log: float
    p_at_nstar_log: float


# Simulation drivers
def simulate_agent(agent: AgentProfile, rng: np.random.Generator,
                   pipeline_factor: float = DEFAULT_PIPELINE_FACTOR) -> SimResult:
    """Run full MC simulation for a single agent across N=3..40.

    Samples parameters once, then sweeps N to compute bit rate curves
    under both accuracy models.
    """
    params = agent.sample(rng, N_MC)
    n_n = len(N_RANGE)

    b_lin = np.empty((n_n, N_MC))
    b_log = np.empty((n_n, N_MC))
    p_lin_arr = np.empty((n_n, N_MC))
    p_log_arr = np.empty((n_n, N_MC))

    for i, n in enumerate(N_RANGE):
        r = selection_rate(params["a"], params["b"], n, pipeline_factor)
        p_l = accuracy_linear(params["p0"], params["c"], n)
        p_g = accuracy_logistic(params["p0"], params["k"], params["x0"], n)
        b_lin[i] = bit_rate(n, r, p_l)
        b_log[i] = bit_rate(n, r, p_g)
        p_lin_arr[i] = p_l
        p_log_arr[i] = p_g

    mean_lin = b_lin.mean(axis=1)
    mean_log = b_log.mean(axis=1)
    idx_lin = int(np.argmax(mean_lin))
    idx_log = int(np.argmax(mean_log))

    return SimResult(
        agent_name=agent.name,
        n_values=N_RANGE,
        b_mean_lin=mean_lin,
        b_p5_lin=np.percentile(b_lin, 5, axis=1),
        b_p95_lin=np.percentile(b_lin, 95, axis=1),
        n_star_lin=int(N_RANGE[idx_lin]),
        peak_b_lin=float(mean_lin[idx_lin]),
        p_at_nstar_lin=float(p_lin_arr[idx_lin].mean()),
        b_mean_log=mean_log,
        b_p5_log=np.percentile(b_log, 5, axis=1),
        b_p95_log=np.percentile(b_log, 95, axis=1),
        n_star_log=int(N_RANGE[idx_log]),
        peak_b_log=float(mean_log[idx_log]),
        p_at_nstar_log=float(p_log_arr[idx_log].mean()),
    )


def simulate_panel_average(rng: np.random.Generator,
                           pipeline_factor: float = DEFAULT_PIPELINE_FACTOR) -> SimResult:
    """Compute panel-average bit rate by independently sampling each panel member.

    For each MC iteration i, independently sample parameters from Emma, Calvin,
    and Elizabeth, compute each agent's B(N), then average the three.

    NOTE: The 2000-sample confidence intervals represent uncertainty in the
    EXPECTED bit rate (i.e., the mean of the distribution). The actual scoring
    event is a single run per person, so real-world variance will be much higher.
    """
    n_n = len(N_RANGE)
    all_params = [agent.sample(rng, N_MC) for agent in PANEL_MEMBERS]

    b_avg_lin = np.zeros((n_n, N_MC))
    b_avg_log = np.zeros((n_n, N_MC))
    p_avg_lin = np.zeros((n_n, N_MC))
    p_avg_log = np.zeros((n_n, N_MC))

    for i, n in enumerate(N_RANGE):
        for params in all_params:
            r = selection_rate(params["a"], params["b"], n, pipeline_factor)
            p_l = accuracy_linear(params["p0"], params["c"], n)
            p_g = accuracy_logistic(params["p0"], params["k"], params["x0"], n)
            b_avg_lin[i] += bit_rate(n, r, p_l)
            b_avg_log[i] += bit_rate(n, r, p_g)
            p_avg_lin[i] += p_l
            p_avg_log[i] += p_g

    b_avg_lin /= len(PANEL_MEMBERS)
    b_avg_log /= len(PANEL_MEMBERS)
    p_avg_lin /= len(PANEL_MEMBERS)
    p_avg_log /= len(PANEL_MEMBERS)

    mean_lin = b_avg_lin.mean(axis=1)
    mean_log = b_avg_log.mean(axis=1)
    idx_lin = int(np.argmax(mean_lin))
    idx_log = int(np.argmax(mean_log))

    return SimResult(
        agent_name="Panel Average",
        n_values=N_RANGE,
        b_mean_lin=mean_lin,
        b_p5_lin=np.percentile(b_avg_lin, 5, axis=1),
        b_p95_lin=np.percentile(b_avg_lin, 95, axis=1),
        n_star_lin=int(N_RANGE[idx_lin]),
        peak_b_lin=float(mean_lin[idx_lin]),
        p_at_nstar_lin=float(p_avg_lin[idx_lin].mean()),
        b_mean_log=mean_log,
        b_p5_log=np.percentile(b_avg_log, 5, axis=1),
        b_p95_log=np.percentile(b_avg_log, 95, axis=1),
        n_star_log=int(N_RANGE[idx_log]),
        peak_b_log=float(mean_log[idx_log]),
        p_at_nstar_log=float(p_avg_log[idx_log].mean()),
    )


# Plotting
PANEL_AVG_COLOR = "#2C3E50"

def _style_axes(ax: plt.Axes, title: str) -> None:
    """Apply consistent professional styling to axes."""
    ax.set_xlabel("Alphabet Size N", fontsize=12)
    ax.set_ylabel("Bit Rate (bits/s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _plot_agent_curve(ax: plt.Axes, res: SimResult, color: str,
                      label: str, lw: float = 1.5, alpha_band: float = 0.12) -> None:
    """Plot mean + CI band for one agent (linear model)."""
    ax.plot(res.n_values, res.b_mean_lin, color=color, linewidth=lw, label=label)
    ax.fill_between(res.n_values, res.b_p5_lin, res.b_p95_lin,
                    color=color, alpha=alpha_band)
    ax.axvline(res.n_star_lin, color=color, linestyle="--", alpha=0.45, linewidth=0.8)


def plot_panel_bitrate(results: Dict[str, SimResult],
                       panel_result: SimResult) -> None:
    """B(N) for Emma, Calvin, Elizabeth, and panel average (linear model)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")

    for agent in PANEL_MEMBERS:
        _plot_agent_curve(ax, results[agent.name], agent.color, agent.name)

    _plot_agent_curve(ax, panel_result, PANEL_AVG_COLOR, "Panel Average",
                      lw=2.5, alpha_band=0.10)

    _style_axes(ax, "Bit Rate vs Alphabet Size \u2014 Panel Members (Linear Model)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bitrate_vs_n_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_agents(results: Dict[str, SimResult],
                    panel_result: SimResult) -> None:
    """B(N) for all six agents + panel average (linear model)."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300, facecolor="white")

    for agent in ALL_AGENTS:
        _plot_agent_curve(ax, results[agent.name], agent.color, agent.name,
                          alpha_band=0.08)

    _plot_agent_curve(ax, panel_result, PANEL_AVG_COLOR, "Panel Average",
                      lw=2.5, alpha_band=0.10)

    _style_axes(ax, "Bit Rate vs Alphabet Size \u2014 All Agents (Linear Model)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bitrate_vs_n_all_agents.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_linear_vs_logistic(panel_result: SimResult) -> None:
    """Overlay linear vs logistic accuracy model for panel average."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")

    ax.plot(panel_result.n_values, panel_result.b_mean_lin, color=PANEL_AVG_COLOR,
            linewidth=2.0, label=f"Linear (N*={panel_result.n_star_lin})")
    ax.fill_between(panel_result.n_values, panel_result.b_p5_lin,
                    panel_result.b_p95_lin, color=PANEL_AVG_COLOR, alpha=0.12)

    ax.plot(panel_result.n_values, panel_result.b_mean_log, color="#E74C3C",
            linewidth=2.0, linestyle="-.",
            label=f"Logistic (N*={panel_result.n_star_log})")
    ax.fill_between(panel_result.n_values, panel_result.b_p5_log,
                    panel_result.b_p95_log, color="#E74C3C", alpha=0.10)

    ax.axvline(panel_result.n_star_lin, color=PANEL_AVG_COLOR, linestyle="--",
               alpha=0.6, linewidth=1.0)
    ax.axvline(panel_result.n_star_log, color="#E74C3C", linestyle="--",
               alpha=0.6, linewidth=1.0)

    _style_axes(ax, "Panel Average: Linear vs Logistic Accuracy Model")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "linear_vs_logistic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Console output
def print_summary_table(results: Dict[str, SimResult],
                        panel: SimResult) -> None:
    """Print the main summary table of optimal N* and peak bit rates."""
    header = (
        f"{'Agent':<18}| {'Lin N*':>6} | {'Log N*':>6} | "
        f"{'Peak B(lin)':>11} | {'Peak B(log)':>11} | "
        f"{'p(N*) lin':>9} | {'p(N*) log':>9} | "
        f"{'Eff lin':>8} | {'Eff log':>8}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    def _row(name: str, res: SimResult) -> None:
        eff_lin = 2.0 * res.p_at_nstar_lin - 1.0
        eff_log = 2.0 * res.p_at_nstar_log - 1.0
        print(
            f"{name:<18}| {res.n_star_lin:>6} | {res.n_star_log:>6} | "
            f"{res.peak_b_lin:>11.3f} | {res.peak_b_log:>11.3f} | "
            f"{res.p_at_nstar_lin:>9.4f} | {res.p_at_nstar_log:>9.4f} | "
            f"{eff_lin:>8.4f} | {eff_log:>8.4f}"
        )

    for agent in ALL_AGENTS:
        _row(agent.name, results[agent.name])
    _row("Panel Average", panel)
    print(sep)


def print_pipeline_sensitivity() -> None:
    """Print how panel-average N* and peak B shift with pipeline factor."""
    print("\n── Pipeline Factor Sensitivity (Panel Average, Linear Model) ──")
    print(f"{'Pipeline':>10} | {'N*':>4} | {'Peak B':>8} | {'Delta N*':>8} | {'Delta B':>8}")
    print("-" * 55)

    ref_n: int | None = None
    ref_b: float | None = None
    for pf in PIPELINE_FACTORS:
        rng = np.random.default_rng(SEED)
        res = simulate_panel_average(rng, pipeline_factor=pf)
        if ref_n is None:
            ref_n = res.n_star_lin
            ref_b = res.peak_b_lin
        print(
            f"{pf:>10.2f} | {res.n_star_lin:>4} | {res.peak_b_lin:>8.3f} | "
            f"{res.n_star_lin - ref_n:>+8} | {res.peak_b_lin - ref_b:>+8.3f}"
        )


def print_breakdown(panel_result: SimResult) -> None:
    """Print expected R, p, Sc, Si, Sc-Si, B at panel-average optimal N*."""
    n_star = panel_result.n_star_lin
    print(f"\n── Breakdown at Panel-Average Optimal N*={n_star} (Linear Model) ──")

    rng = np.random.default_rng(SEED)
    r_means, p_means, b_means = [], [], []
    for agent in PANEL_MEMBERS:
        params = agent.sample(rng, N_MC)
        r = selection_rate(params["a"], params["b"], n_star)
        p = accuracy_linear(params["p0"], params["c"], n_star)
        b = bit_rate(n_star, r, p)
        r_means.append(float(r.mean()))
        p_means.append(float(p.mean()))
        b_means.append(float(b.mean()))

    r_avg = np.mean(r_means)
    p_avg = np.mean(p_means)
    s_avg = r_avg * SESSION_DURATION
    sc_avg = s_avg * p_avg
    si_avg = s_avg * (1.0 - p_avg)

    print(f"  Mean R (sel/s):      {r_avg:.3f}")
    print(f"  Mean p (accuracy):   {p_avg:.4f}")
    print(f"  Mean S (total sel):  {s_avg:.1f}")
    print(f"  Mean Sc (correct):   {sc_avg:.1f}")
    print(f"  Mean Si (incorrect): {si_avg:.1f}")
    print(f"  Mean Sc - Si:        {sc_avg - si_avg:.1f}")
    print(f"  Mean B (bits/s):     {np.mean(b_means):.3f}")
    print(f"  Shenoy Efficiency:   {2.0 * p_avg - 1.0:.4f}")

# Main
def main() -> None:
    """Run the full simulation, print results, and generate figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Simulate individual agents
    rng = np.random.default_rng(SEED)
    results: Dict[str, SimResult] = {}
    for agent in ALL_AGENTS:
        results[agent.name] = simulate_agent(agent, rng)

    # Simulate panel average (fresh RNG for independent sampling)
    panel = simulate_panel_average(np.random.default_rng(SEED))

    # Console output
    print_summary_table(results, panel)
    print_pipeline_sensitivity()
    print_breakdown(panel)

    # Generate figures
    plot_panel_bitrate(results, panel)
    plot_all_agents(results, panel)
    plot_linear_vs_logistic(panel)

    print(f"\nFigures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
