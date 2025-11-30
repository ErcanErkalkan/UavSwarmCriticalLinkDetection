#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACLD-UAV: Adaptive Critical Link Detection for UAV Swarms
Tam Python iskelet proje (tek dosya).
- UAVSwarm: sürü dinamiği
- ChannelModel: Nakagami-m / jamming
- ACLD: kritik link tespiti
- Basit Tarjan köprü bulma
- Demo simülasyon ve runtime benchmark
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Optional, Set

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy.special import gammaincc  # normalized upper incomplete gamma
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# 1. Kanal Modeli (Nakagami-m + Jamming)
# ============================================================================

@dataclass
class ChannelModel:
    """
    Nakagami-m fading + path-loss modeli.
    Makaledeki default değerlerle başlatılır.
    """
    m: float = 2.5          # Nakagami shape param
    snr0_db: float = 22.0   # referans SNR(d0)'daki ortalama (dB)
    gamma0_db: float = 10.0 # hedef SNR (dB)
    d0: float = 100.0       # referans mesafe (m)
    xi: float = 2.4         # path-loss exponent (plExp)
    jammer_db: float = 0.0  # ortalama jammer gücü (SNR'ya göre dB cinsinden)

    def __post_init__(self):
        self.snr0_lin = 10.0 ** (self.snr0_db / 10.0)
        self.gamma0_lin = 10.0 ** (self.gamma0_db / 10.0)
        self.jammer_lin = 10.0 ** (self.jammer_db / 10.0) if self.jammer_db > 0 else 0.0

    def _nakagami_success_prob_exact(self, d: np.ndarray) -> np.ndarray:
        """
        P[SINR >= gamma0] – SciPy varsa makaledeki formülle.
        """
        # Ortalama SINR(d) = SNR0 * (d0/d)^xi / (1+J)
        with np.errstate(divide='ignore'):
            sinr_bar = self.snr0_lin * (self.d0 / np.maximum(d, 1e-3)) ** self.xi / (1.0 + self.jammer_lin)
        arg = self.m * (self.gamma0_lin / np.maximum(sinr_bar, 1e-9))
        # gammaincc(a,x) = Γ(a,x)/Γ(a)
        return gammaincc(self.m, arg)

    def _nakagami_success_prob_approx(self, d: np.ndarray) -> np.ndarray:
        """
        SciPy yoksa makul bir approx:
        Rayleigh benzeri tail: P ~ exp(-c * (d/d0)^xi)
        """
        c = (self.gamma0_lin / self.snr0_lin) * (1.0 + self.jammer_lin)
        with np.errstate(over='ignore'):
            return np.exp(-c * (np.maximum(d, 1e-3) / self.d0) ** self.xi)

    def success_probability(self, d: np.ndarray) -> np.ndarray:
        """
        d: [m] (scalar veya ndarray)
        Çıktı: P[link up] (aynı shape)
        """
        d = np.asarray(d, dtype=float)
        if _HAS_SCIPY:
            return self._nakagami_success_prob_exact(d)
        else:
            return self._nakagami_success_prob_approx(d)


# ============================================================================
# 2. UAV Sürüsü Dinamiği
# ============================================================================

class UAVSwarm:
    """
    Basitleştirilmiş UAV sürüsü:
    - pos: (n,3)
    - vel: (n,3)
    - acc: (n,3)
    - jerk: (n,3)
    Adaptif eşik için gereken terimler ve temel dinamikler var.
    """
    def __init__(
        self,
        n: int,
        dt: float = 0.01,
        cthr_base: float = 100.0,
        v_max: float = 23.0,
        a_max: float = 60.0,
        j_max: float = 100.0,
        alpha: float = 0.35,
        beta: float = 0.10,
        gamma: float = 0.20,
        delta: float = 0.10,
        zeta: float = 0.05,
        density: str = "medium",
        seed: Optional[int] = None,
    ) -> None:
        self.n = n
        self.dt = dt
        self.cthr_base = cthr_base
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta

        self.rng = np.random.default_rng(seed)

        if density == "sparse":
            scale = 1.5 * cthr_base
        elif density == "dense":
            scale = 0.3 * cthr_base
        else:
            scale = 0.8 * cthr_base

        # Başlangıç durumları
        self.pos = self.rng.uniform(-scale, scale, size=(n, 3))
        self.vel = self._rand_bounded(self.v_max * 0.5, size=(n, 3))
        self.acc = self._rand_bounded(self.a_max * 0.1, size=(n, 3))
        self.jerk = np.zeros((n, 3), dtype=float)

    def _rand_bounded(self, bound: float, size: Tuple[int, int]) -> np.ndarray:
        return self.rng.uniform(-bound, bound, size=size)

    # ---------------- dynamics ----------------

    def step_dynamics(self) -> None:
        """
        Çok basit bir model:
        - Kontrol ivmesi hafif random walk
        - hızı güncelle, saturate et
        - pozisyonu güncelle
        """
        noise = self.rng.normal(scale=self.a_max * 0.05, size=(self.n, 3))
        new_acc = self.acc + noise

        # ivme sınırı (satır bazlı scale)
        norms_a = np.linalg.norm(new_acc, axis=1, keepdims=True)  # (n,1)
        scale_a = np.ones_like(norms_a)
        mask_a = norms_a > self.a_max
        scale_a[mask_a] = self.a_max / np.maximum(norms_a[mask_a], 1e-9)
        new_acc *= scale_a

        self.jerk = (new_acc - self.acc) / self.dt
        self.acc = new_acc

        # hız
        self.vel += self.acc * self.dt
        norms_v = np.linalg.norm(self.vel, axis=1, keepdims=True)  # (n,1)
        scale_v = np.ones_like(norms_v)
        mask_v = norms_v > self.v_max
        scale_v[mask_v] = self.v_max / np.maximum(norms_v[mask_v], 1e-9)
        self.vel *= scale_v

        # pozisyon
        self.pos += self.vel * self.dt

    # ---------------- yardımcı fonksiyonlar ----------------

    def pairwise_distances(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tüm UAV çiftleri arasındaki Öklid uzaklıkları.
        """
        if pos is None:
            pos = self.pos
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _heading_angles(self) -> np.ndarray:
        """
        2D heading (xy düzleminde) – radian.
        """
        vx = self.vel[:, 0]
        vy = self.vel[:, 1]
        return np.arctan2(vy, vx)

    def adaptive_cthr(self, d_air: np.ndarray) -> np.ndarray:
        """
        Eq.(cthr_adp): cthr_ij(t) = cthr_base * [ .. ]
        d_air: anlık UAV–UAV mesafe matrisi (n,n)
        """
        n = self.n
        v_max = self.v_max
        a_max = self.a_max
        j_max = self.j_max

        # global dist istatistikları (üst üçgen)
        iu = np.triu_indices(n, k=1)
        d_vals = d_air[iu]
        mean_d = float(np.mean(d_vals))
        sigma_d = float(np.std(d_vals))

        # relative terimler
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        dv_norm = np.linalg.norm(dv, axis=-1)

        da = self.acc[:, None, :] - self.acc[None, :, :]
        da_norm = np.linalg.norm(da, axis=-1)

        dj = self.jerk[:, None, :] - self.jerk[None, :, :]
        dj_norm = np.linalg.norm(dj, axis=-1)

        head = self._heading_angles()
        dtheta = head[:, None] - head[None, :]
        dtheta = np.abs((dtheta + np.pi) % (2 * np.pi) - np.pi)  # [-pi,pi] -> |.| <= pi

        # if mean_d ~ 0 ise patlamasın
        disp_term = (sigma_d / mean_d) if mean_d > 1e-6 else 0.0

        term = (
            1.0
            + self.alpha * disp_term
            - self.beta * (dv_norm / max(v_max, 1e-6))
            - self.gamma * (dtheta / np.pi)
            + self.delta * (da_norm / max(a_max, 1e-6))
            - self.zeta * (dj_norm / max(j_max, 1e-6))
        )

        cthr = self.cthr_base * term
        # negatifleri engelle (çok agresif parametrelerde olabilir):
        return np.maximum(cthr, 0.1 * self.cthr_base)


# ============================================================================
# 3. Laplasyen ve Cebirsel Bağlantı
# ============================================================================

def build_weight_matrices(d_air: np.ndarray, theta_plus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Makaledeki row-stochastic -> symmetrize sürecini uygular:
    - tildeW: yönlü row-stochastic ağırlıklar
    - W: simetriklaştırılmış ağırlık matrisi
    - L: Laplasyen = D - W
    """
    n = d_air.shape[0]
    tildeW = np.zeros((n, n), dtype=float)

    for i in range(n):
        # Komşu: d_ij <= theta_plus_ij ve i != j
        mask = (d_air[i] <= theta_plus[i]) & (np.arange(n) != i)
        neigh = np.where(mask)[0]
        if neigh.size == 0:
            continue
        inv_d2 = 1.0 / np.maximum(d_air[i, neigh] ** 2, 1e-6)
        s = inv_d2.sum()
        tildeW[i, neigh] = inv_d2 / s

    W = 0.5 * (tildeW + tildeW.T)
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return tildeW, W, L


def algebraic_connectivity(L: np.ndarray) -> float:
    """
    λ₂(L) – ikinci en küçük özdeğer (sym Laplacian).
    """
    vals = np.linalg.eigvalsh(L)
    vals.sort()
    if len(vals) < 2:
        return 0.0
    return float(vals[1])


# ============================================================================
# 4. ACLD Algoritması
# ============================================================================

class ACLD:
    """
    Adaptive Critical Link Detection algoritması.
    - Tahmini adjacency (t+Δt)
    - Bileşenler
    - Riskli linkler (inter / intra)
    """

    def __init__(self, swarm: UAVSwarm, channel: ChannelModel, p_min: float = 0.8) -> None:
        self.swarm = swarm
        self.channel = channel
        self.p_min = p_min

    # ---------- KNeighbors & Components ----------

    @staticmethod
    def _kneighbors(adjacency: np.ndarray,
                    d_hat: np.ndarray,
                    theta_minus: np.ndarray,
                    seed: int,
                    k_max: int) -> set:
        n = adjacency.shape[0]
        visited = {seed}
        frontier = {seed}
        for _ in range(k_max):
            nxt = set()
            for u in frontier:
                neighbors = np.nonzero(adjacency[u])[0]
                for v in neighbors:
                    if v not in visited and d_hat[u, v] <= theta_minus[u, v]:
                        visited.add(v)
                        nxt.add(v)
            if not nxt:
                break
            frontier = nxt
        return visited

    @classmethod
    def predict_components(cls,
                           adjacency: np.ndarray,
                           d_hat: np.ndarray,
                           theta_minus: np.ndarray,
                           k_max: int = 5) -> List[List[int]]:
        n = adjacency.shape[0]
        unseen = set(range(n))
        components: List[List[int]] = []
        while unseen:
            i = next(iter(unseen))
            comp = cls._kneighbors(adjacency, d_hat, theta_minus, i, k_max)
            components.append(sorted(comp))
            unseen -= comp
        return components

    @staticmethod
    def catalogue_risky_links(components: List[List[int]],
                              adjacency: np.ndarray,
                              d_hat: np.ndarray,
                              theta_minus: np.ndarray,
                              theta_plus: np.ndarray
                              ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        comp_of: Dict[int, int] = {}
        for idx, comp in enumerate(components):
            for u in comp:
                comp_of[u] = idx

        inter: List[Tuple[int, int]] = []
        intra: List[Tuple[int, int]] = []

        n = adjacency.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if not adjacency[i, j]:
                    continue
                dij = d_hat[i, j]
                if theta_minus[i, j] < dij <= theta_plus[i, j]:
                    if comp_of[i] != comp_of[j]:
                        inter.append((i, j))
                    else:
                        intra.append((i, j))
        return inter, intra

    # ---------- Predicted adjacency & main step ----------

    def _build_predicted_adjacency(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tahmini adjacency (t+Δt) ve risk bantları.
        """
        n = self.swarm.n
        dt = self.swarm.dt

        # Tahmini pozisyon (sabit hız modeli)
        pos_hat = self.swarm.pos + self.swarm.vel * dt
        d_hat = self.swarm.pairwise_distances(pos_hat)

        # Anlık d_air ile adaptif eşik
        d_air = self.swarm.pairwise_distances()
        cthr = self.swarm.adaptive_cthr(d_air)
        theta_minus = 0.9 * cthr
        theta_plus = cthr

        # Link-up olasılığı
        p_up = self.channel.success_probability(np.maximum(d_hat, 1e-3))
        adj = (d_hat <= theta_plus) & (p_up >= self.p_min)
        np.fill_diagonal(adj, 0)
        # simetrik
        adj = np.logical_or(adj, adj.T)
        return adj.astype(np.int8), d_hat, theta_minus, theta_plus

    def step(self,
             update_dynamics: bool = True,
             k_max: int = 5) -> Tuple[np.ndarray, List[List[int]],
                                     List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Bir kontrol adımı:
        - opsiyonel dinamik güncelle
        - tahmini adjacency + komponentler + riskli linkler
        """
        if update_dynamics:
            self.swarm.step_dynamics()

        adj, d_hat, theta_minus, theta_plus = self._build_predicted_adjacency()
        comps = self.predict_components(adj, d_hat, theta_minus, k_max=k_max)
        inter, intra = self.catalogue_risky_links(comps, adj, d_hat, theta_minus, theta_plus)
        return adj, comps, inter, intra


# ============================================================================
# 5. Basit Tarjan Köprü Algoritması (Snapshot Baseline)
# ============================================================================

def tarjan_bridges(adj: np.ndarray) -> List[Tuple[int, int]]:
    """
    Tarjan'ın O(n+m) köprü bulma algoritması (undirected graph).
    adj: {0,1} adjacency (n,n)
    """
    n = adj.shape[0]
    graph: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                graph[i].append(j)
                graph[j].append(i)

    time_dfs = 0
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges: List[Tuple[int, int]] = []

    def dfs(u: int):
        nonlocal time_dfs
        time_dfs += 1
        disc[u] = low[u] = time_dfs
        for v in graph[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    bridges.sort()
    return bridges


# ============================================================================
# 6. Simülasyon Yardımcıları
# ============================================================================

def simulate_episode(
    n: int = 150,
    T: float = 20.0,
    dt: float = 0.01,
    density: str = "medium",
    jammer_db: float = 0.0,
    seed: int = 0,
    return_timeseries: bool = False,  # <<< YENİ PARAM
):
    """
    Tek episode:
    - λ₂(t) zaman serisi
    - partition sayısı
    - ACLD'nin her adımda kaç inter-riskli link bulduğu
    Basit bir metrik seti döner.

    return_timeseries=True ise zaman serilerini de döndürür.
    """
    steps = int(T / dt)

    swarm = UAVSwarm(
        n=n,
        dt=dt,
        density=density,
        alpha=0.35,
        beta=0.10,
        gamma=0.20,
        delta=0.10,
        zeta=0.05,
        seed=seed,
    )
    channel = ChannelModel(jammer_db=jammer_db)
    acld = ACLD(swarm, channel, p_min=0.8)

    lambda2_vals: List[float] = []
    partitions = 0
    num_risky_inter: List[int] = []

    for _ in range(steps):
        # ACLD adımı
        adj, comps, inter, intra = acld.step(update_dynamics=True, k_max=5)

        # Laplasyen ve λ₂
        d_air = swarm.pairwise_distances()
        cthr = swarm.adaptive_cthr(d_air)
        theta_plus = cthr
        _, _, L = build_weight_matrices(d_air, theta_plus)
        lam2 = algebraic_connectivity(L)
        lambda2_vals.append(lam2)
        if lam2 < 1e-3:   # pratik threshold
            partitions += 1

        num_risky_inter.append(len(inter))

    lambda2_arr = np.asarray(lambda2_vals)
    risky_arr = np.asarray(num_risky_inter)

    summary = {
        "lambda2_mean": float(lambda2_arr.mean()),
        "lambda2_min": float(lambda2_arr.min()),
        "lambda2_std": float(lambda2_arr.std()),
        "partitions": float(partitions),
        "risky_inter_mean": float(risky_arr.mean()),
    }

    if return_timeseries:
        ts = {
            "lambda2": lambda2_arr,
            "risky_inter": risky_arr,
        }
        return summary, ts

    return summary



def bench_runtime(
    n_list: Iterable[int] = (50, 150, 500, 1000),
    repeats: int = 200,
    dt: float = 0.01,
    density: str = "medium",
) -> Dict[int, float]:
    """
    Runtime benchmark:
    Her n için birkaç adımın medyan süresini ölçer.
    (Bu Python versiyonu C++/OpenMP kadar hızlı olmayacak,
     ama O(n^2) ölçeklemeyi görebilirsin.)
    """
    results: Dict[int, float] = {}

    for n in n_list:
        swarm = UAVSwarm(n=n, dt=dt, density=density, seed=0)
        channel = ChannelModel()
        acld = ACLD(swarm, channel, p_min=0.8)

        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            acld.step(update_dynamics=True, k_max=5)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        med = float(np.median(times))
        p25, p75 = float(np.percentile(times, 25)), float(np.percentile(times, 75))
        results[n] = med
        print(f"n={n:4d} -> median {med:.3f} ms  [IQR {p25:.3f}–{p75:.3f}]")

    return results


# ============================================================================
# 7. Mini Parametre Süpürme (alpha, beta, gamma)
# ============================================================================

def simple_param_sweep(
    n: int = 50,
    T: float = 10.0,
    dt: float = 0.1,
    n_trials: int = 30,
    rng_seed: int = 1234,
) -> Tuple[Tuple[float, float, float], Dict[str, float]]:
    """
    Makaledeki LHS/BEI yerine, hafif bir random sweep:
    (alpha, beta, gamma) ~ uniform aralıklar.
    En yüksek reward = E[λ₂] - 0.1 * partitions veren üçlü seçilir.
    """
    rng = np.random.default_rng(rng_seed)
    best_params = None
    best_reward = -1e9
    best_stats = {}

    n_steps = int(T / dt)

    def run_once(alpha, beta, gamma, seed):
        swarm = UAVSwarm(
            n=n,
            dt=dt,
            density="medium",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=0.10,
            zeta=0.05,
            seed=seed,
        )
        channel = ChannelModel()
        acld = ACLD(swarm, channel, p_min=0.8)

        lambda2_vals = []
        partitions = 0
        for _ in range(n_steps):
            adj, comps, inter, intra = acld.step(update_dynamics=True, k_max=5)
            d_air = swarm.pairwise_distances()
            cthr = swarm.adaptive_cthr(d_air)
            theta_plus = cthr
            _, _, L = build_weight_matrices(d_air, theta_plus)
            lam2 = algebraic_connectivity(L)
            lambda2_vals.append(lam2)
            if lam2 < 1e-3:
                partitions += 1

        lam2_arr = np.asarray(lambda2_vals)
        return {
            "lambda2_mean": float(lam2_arr.mean()),
            "partitions": float(partitions),
        }

    for trial in range(n_trials):
        alpha = rng.uniform(0.1, 0.5)
        beta = rng.uniform(0.05, 0.4)
        gamma = rng.uniform(0.05, 0.3)

        stats = run_once(alpha, beta, gamma, seed=trial)
        reward = stats["lambda2_mean"] - 0.1 * stats["partitions"]

        print(
            f"[{trial+1:02d}/{n_trials}] "
            f"alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f} "
            f"-> λ2̄={stats['lambda2_mean']:.4f}, parts={stats['partitions']:.0f}, "
            f"reward={reward:.4f}"
        )

        if reward > best_reward:
            best_reward = reward
            best_params = (alpha, beta, gamma)
            best_stats = stats

    print("\nBest params:", best_params, "  best_stats:", best_stats, "  reward:", best_reward)
    return best_params, best_stats


# ============================================================================
# 8. CLI / Demo
# ============================================================================

def _demo():
    print("=== ACLD Demo: tek episode, n=150, T=20 s ===")
    stats = simulate_episode(n=150, T=20.0, dt=0.01, density="medium", jammer_db=0.0, seed=0)
    for k, v in stats.items():
        print(f"{k:20s}: {v:.4f}")


def _demo_jamming():
    print("=== ACLD Demo: jamming 15 dB, n=150, T=20 s ===")
    stats = simulate_episode(n=150, T=20.0, dt=0.01, density="medium", jammer_db=15.0, seed=42)
    for k, v in stats.items():
        print(f"{k:20s}: {v:.4f}")

# ============================================================================
# 9. Plot Yardımcıları
# ============================================================================

def plot_lambda2_timeseries(
    ts_dicts: Dict[str, Dict[str, np.ndarray]],
    dt: float,
    title: str = "Algebraic connectivity over time",
    filename: Optional[str] = None,
):
    """
    ts_dicts: {"label": {"lambda2": np.array, "risky_inter": np.array}, ...}
    dt: zaman adımı
    """
    if not _HAS_MPL:
        print("[WARN] Matplotlib yüklü değil, grafik üretilemiyor.")
        return

    plt.figure()
    for label, ts in ts_dicts.items():
        lam = ts["lambda2"]
        t = np.arange(len(lam)) * dt
        plt.plot(t, lam, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel(r"$\lambda_2(L(t))$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[INFO] λ2 zaman serisi grafiği kaydedildi: {filename}")
    else:
        plt.show()


def plot_runtime_vs_n(
    runtime_results: Dict[int, float],
    title: str = "ACLD runtime vs swarm size",
    filename: Optional[str] = None,
):
    if not _HAS_MPL:
        print("[WARN] Matplotlib yüklü değil, grafik üretilemiyor.")
        return

    ns = sorted(runtime_results.keys())
    med = [runtime_results[n] for n in ns]

    plt.figure()
    plt.plot(ns, med, marker="o")
    plt.xlabel("Number of UAVs (n)")
    plt.ylabel("Median step runtime [ms]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Eğer makalede log-log kullandıysan:
    # plt.xscale("log")
    # plt.yscale("log")

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[INFO] Runtime grafiği kaydedildi: {filename}")
    else:
        plt.show()

# ============================================================================
# 10. Makale-vari Toplu Deney Koşucu
# ============================================================================

def run_paper_like_experiments(
    out_prefix: str = "acld_results",
    n: int = 150,
    T: float = 20.0,
    dt: float = 0.01,
    n_runs_per_scenario: int = 10,
):
    """
    Örnek “makale stili” deney seti:
    - Senaryo 1: Jammer yok
    - Senaryo 2: Orta jammer (15 dB)
    - Senaryo 3: Yüksek jammer (25 dB)
    Her senaryo için n_runs_per_scenario adet episode koşar,
    λ₂ zaman serilerini ve özet istatistikleri kaydeder.
    """
    scenarios = {
        "no_jammer": 0.0,
        "jam_15dB": 15.0,
        "jam_25dB": 25.0,
    }

    print("=== [PAPER MODE] ACLD-UAV deneyleri başlıyor ===")
    print(f"n={n}, T={T}, dt={dt}, runs/scenario={n_runs_per_scenario}")
    print(f"Senaryolar: {', '.join([f'{k}={v} dB' for k, v in scenarios.items()])}")
    print("===============================================")

    all_stats = {}
    all_ts_for_plot = {}

    for name, jammer_db in scenarios.items():
        print(f"\n=== Scenario: {name} (jammer_db={jammer_db} dB) ===")

        scenario_lambda2 = []
        scenario_risky = []
        scenario_summaries = []

        for run_id in range(n_runs_per_scenario):
            print(f"  -> Run {run_id+1}/{n_runs_per_scenario} başlıyor...", end="", flush=True)

            summary, ts = simulate_episode(
                n=n,
                T=T,
                dt=dt,
                density="medium",
                jammer_db=jammer_db,
                seed=run_id,
                return_timeseries=True,
            )
            scenario_summaries.append(summary)
            scenario_lambda2.append(ts["lambda2"])
            scenario_risky.append(ts["risky_inter"])

            print(
                f" bitti. "
                f"λ2̄={summary['lambda2_mean']:.4f}, "
                f"λ2_min={summary['lambda2_min']:.4f}, "
                f"σ(λ2)={summary['lambda2_std']:.4f}, "
                f"partitions={summary['partitions']:.0f}, "
                f"risky_inter̄={summary['risky_inter_mean']:.4f}"
            )

        # Aynı uzunlukta olduklarını varsayıyoruz (T/dt sabit)
        lambda2_arr = np.vstack(scenario_lambda2)
        risky_arr = np.vstack(scenario_risky)

        print(f"[{name}] Tüm run'lar tamamlandı. lambda2_arr.shape={lambda2_arr.shape}, "
              f"risky_arr.shape={risky_arr.shape}")

        # Ortalama zaman serisi
        mean_lambda2 = lambda2_arr.mean(axis=0)
        mean_risky = risky_arr.mean(axis=0)

        all_ts_for_plot[name] = {
            "lambda2": mean_lambda2,
            "risky_inter": mean_risky,
        }

        # Özet istatistiklerin ortalaması
        lambda2_means = [s["lambda2_mean"] for s in scenario_summaries]
        partitions = [s["partitions"] for s in scenario_summaries]

        stats = {
            "lambda2_mean_mean": float(np.mean(lambda2_means)),
            "lambda2_mean_std": float(np.std(lambda2_means)),
            "partitions_mean": float(np.mean(partitions)),
            "partitions_std": float(np.std(partitions)),
        }
        all_stats[name] = stats
        print(f"[{name}] λ2̄ (episode ort.): {stats['lambda2_mean_mean']:.4f} ± {stats['lambda2_mean_std']:.4f}, "
              f"partitions: {stats['partitions_mean']:.2f} ± {stats['partitions_std']:.2f}")

        # NumPy binary olarak kaydet
        npz_name = f"{out_prefix}_{name}_timeseries.npz"
        np.savez_compressed(
            npz_name,
            lambda2=lambda2_arr,
            risky=risky_arr,
        )
        print(f"[INFO] Zaman serileri NPZ olarak kaydedildi: {npz_name}")

    # λ₂ zaman serisi figürü üret
    print("\n[INFO] λ2 zaman serisi grafiği üretiliyor...")
    plot_lambda2_timeseries(
        all_ts_for_plot,
        dt=dt,
        title="Effect of jamming on algebraic connectivity",
        filename=f"{out_prefix}_lambda2_vs_time.png",
    )

    # Runtime grafiği
    print("[INFO] Runtime benchmark çalıştırılıyor...")
    runtime_res = bench_runtime()
    plot_runtime_vs_n(runtime_res, filename=f"{out_prefix}_runtime_vs_n.png")

    # Özet sonuçları CSV olarak kaydet
    import csv
    csv_name = f"{out_prefix}_summary.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "lambda2_mean_mean", "lambda2_mean_std",
                         "partitions_mean", "partitions_std"])
        for name, s in all_stats.items():
            writer.writerow([
                name,
                s["lambda2_mean_mean"],
                s["lambda2_mean_std"],
                s["partitions_mean"],
                s["partitions_std"],
            ])
    print(f"[INFO] Özet istatistikler CSV olarak kaydedildi: {csv_name}")

    print("\n=== [PAPER MODE] Tüm deneyler tamamlandı. ===")
    for name, s in all_stats.items():
        print(f"  - {name}: λ2̄={s['lambda2_mean_mean']:.4f} ± {s['lambda2_mean_std']:.4f}, "
              f"partitions={s['partitions_mean']:.2f} ± {s['partitions_std']:.2f}")

    return all_stats

# ============================================================================
# 11. Baseline yardımcıları: adjacency, ek algoritmalar, metrikler
# ============================================================================

def build_adjacency_from_swarm_and_channel(
    swarm: UAVSwarm,
    channel: ChannelModel,
    p_min: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mevcut sürü durumundan (t anı) tahmini adjacency matrisi üretir.
    - d_air: anlık mesafeler
    - cthr/theta_plus: adaptif eşik
    - adj: Nakagami-m guard (P_up >= p_min) uygulanmış komşuluk
    """
    d_air = swarm.pairwise_distances()
    cthr = swarm.adaptive_cthr(d_air)
    theta_plus = cthr

    p_up = channel.success_probability(np.maximum(d_air, 1e-3))
    adj = (d_air <= theta_plus) & (p_up >= p_min)
    np.fill_diagonal(adj, 0)
    adj = np.logical_or(adj, adj.T)

    return adj.astype(np.int8), d_air, theta_plus


def connected_components_from_adj(adj: np.ndarray) -> List[List[int]]:
    """
    Basit bağlı bileşen bulucu (undirected graph, adjacency {0,1}).
    """
    n = adj.shape[0]
    unseen = set(range(n))
    comps: List[List[int]] = []

    while unseen:
        start = unseen.pop()
        comp = [start]
        stack = [start]
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and v in unseen:
                    unseen.remove(v)
                    comp.append(v)
                    stack.append(v)
        comps.append(sorted(comp))

    return comps


def ground_truth_critical_edges(adj: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Ground-truth kritik kenarlar:
    - Tanım: t+Δt anındaki grafikte Tarjan köprüleri.
    """
    return set(tarjan_bridges(adj))


def eb_appx_top_eta(
    adj: np.ndarray,
    eta: float = 0.05,
    max_sources: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> Set[Tuple[int, int]]:
    """
    EB-APPX (top-η) için çok basit bir edge-betweenness approx:
    - Rastgele bazı kaynaklardan BFS ağacı çıkar
    - Ağaç kenarlarını say → skor
    - Skoru en yüksek η*m kenarı döndür.
    """
    n = adj.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    m = len(edges)
    if m == 0:
        return set()

    num_sources = min(max_sources, n)
    sources = rng.choice(n, size=num_sources, replace=False)

    scores: Dict[Tuple[int, int], float] = {e: 0.0 for e in edges}
    from collections import deque

    for s in sources:
        parent = {s: None}
        order: List[int] = []
        q = deque([s])

        while q:
            u = q.popleft()
            order.append(u)
            for v in range(n):
                if adj[u, v] and v not in parent:
                    parent[v] = u
                    q.append(v)

        # BFS ağacındaki kenarlara +1 ver
        for v in order:
            u = parent.get(v)
            if u is None:
                continue
            e = (u, v) if u < v else (v, u)
            scores[e] += 1.0

    num_pick = max(1, int(eta * m))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return set(e for e, _ in ranked[:num_pick])


def cut_kec_flag_edges(adj: np.ndarray, frac: float = 0.05) -> Set[Tuple[int, int]]:
    """
    CUT/KEC için basit bir "key edge" heuristiği:
    - Kenar skor = (deg(i)+deg(j)) * (deg(i)*deg(j))
    - Skoru en yüksek frac*m kenarı "kritik" say.
    """
    n = adj.shape[0]
    deg = np.sum(adj, axis=1)

    scored_edges: List[Tuple[Tuple[int, int], float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                score = (deg[i] + deg[j]) * (deg[i] * deg[j] + 1e-6)
                scored_edges.append(((i, j), score))

    m = len(scored_edges)
    if m == 0:
        return set()

    num_pick = max(1, int(frac * m))
    scored_edges.sort(key=lambda kv: kv[1], reverse=True)
    return set(e for e, _ in scored_edges[:num_pick])


# ---------------- Metrikler ----------------

def compute_pew(
    flags_per_step: List[Set[Tuple[int, int]]],
    critical_per_step: List[Set[Tuple[int, int]]],
    partition_events: List[bool],
) -> float:
    """
    PEW (%): Partition Early Warning
    - Partition olan adımlarda (λ2 < threshold) kritik kenarlardan en az
      birini flag'leyebilen adımların oranı.
    """
    assert len(flags_per_step) == len(critical_per_step) == len(partition_events)
    hits = 0
    total = 0

    for k, is_part in enumerate(partition_events):
        if not is_part:
            continue
        Ck = critical_per_step[k]
        if not Ck:
            continue
        total += 1
        if flags_per_step[k] & Ck:
            hits += 1

    if total == 0:
        return 0.0
    return 100.0 * hits / total


def compute_fp_rate(
    flags_per_step: List[Set[Tuple[int, int]]],
    critical_per_step: List[Set[Tuple[int, int]]],
) -> float:
    """
    FP (%): False Positive Rate
    - Flag'lenen tüm kenarlar içinde kritik OLMAYAN kenarların oranı.
    """
    flagged = 0
    tp = 0

    for Fk, Ck in zip(flags_per_step, critical_per_step):
        flagged += len(Fk)
        tp += len(Fk & Ck)

    if flagged == 0:
        return 0.0

    fp = flagged - tp
    return 100.0 * fp / flagged


def compute_edge_f1(
    flags_per_step: List[Set[Tuple[int, int]]],
    critical_per_step: List[Set[Tuple[int, int]]],
) -> float:
    """
    Edge-level F1:
    - TP, FP, FN kenar sayıları üzerinden mikro ortalama F1.
    """
    tp = 0
    flagged = 0
    gt = 0

    for Fk, Ck in zip(flags_per_step, critical_per_step):
        tp += len(Fk & Ck)
        flagged += len(Fk)
        gt += len(Ck)

    if flagged == 0 or gt == 0:
        return 0.0

    precision = tp / flagged
    recall = tp / gt
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_detection_latency(
    flags_per_step: List[Set[Tuple[int, int]]],
    critical_per_step: List[Set[Tuple[int, int]]],
    partition_events: List[bool],
    dt: float,
) -> float:
    """
    Detection latency (ms):
    - Her partition olayı için:
        * Eğer kritik kenar adım k'da oluşuyorsa,
        * En erken t<=k adımında bu kenarlardan en az birini flag'leyen
          algoritmanın gecikmesi = (k - t)*dt (ms)
        * Eğer sadece t>k adımlarında flag'liyorsa: negatif gecikme
          (post-factum) = -(t-k)*dt (ms)
    - Çıkış: partition olayları üzerindeki ortalama latency (ms).
    """
    steps = len(flags_per_step)
    latencies_ms: List[float] = []

    for k in range(steps):
        if not partition_events[k]:
            continue
        Ck = critical_per_step[k]
        if not Ck:
            continue

        # t <= k erken uyarılar
        t_hit_pre = None
        for t in range(0, k + 1):
            if flags_per_step[t] & Ck:
                t_hit_pre = t
                break

        if t_hit_pre is not None:
            latencies_ms.append((k - t_hit_pre) * dt * 1000.0)
            continue

        # t > k post-factum
        t_hit_post = None
        for t in range(k + 1, steps):
            if flags_per_step[t] & Ck:
                t_hit_post = t
                break

        if t_hit_post is not None:
            latencies_ms.append(-(t_hit_post - k) * dt * 1000.0)

    if not latencies_ms:
        return 0.0

    return float(np.mean(latencies_ms))

# ============================================================================
# 12. Tek episode için tüm baselines + metrikler
# ============================================================================

def simulate_episode_with_baselines(
    n: int,
    T: float,
    dt: float,
    density: str = "medium",
    jammer_db: float = 0.0,
    p_min: float = 0.8,
    eta: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Bir episode içinde ACLD ve baselines'ları karşılaştırır.
    Çıkış: her yöntem için {pew, fp, lat, f1, tstep} sözlüğü.
    """
    steps = int(T / dt)

    swarm = UAVSwarm(
        n=n,
        dt=dt,
        density=density,
        alpha=0.35,
        beta=0.10,
        gamma=0.20,
        delta=0.10,
        zeta=0.05,
        seed=seed,
    )
    channel = ChannelModel(jammer_db=jammer_db)
    acld = ACLD(swarm, channel, p_min=p_min)
    rng = np.random.default_rng(seed)

    methods = ["ACLD", "TARJAN_t", "TARJAN_tplus", "EBAPPX", "CUTKEC"]

    # Her adımda hangi kenarları flag'ledi?
    flags: Dict[str, List[Set[Tuple[int, int]]]] = {m: [] for m in methods}
    # Her adımda algoritma runtime'ı (ms)
    times_ms: Dict[str, List[float]] = {m: [] for m in methods}
    # Ground-truth kritik kenarlar ve partition olayları
    critical_edges_per_step: List[Set[Tuple[int, int]]] = []
    partition_events: List[bool] = []

    for _ in range(steps):
        # 1) t anındaki graph (adj_t): Tarjan@t, EB-APPX, CUT/KEC için
        adj_t, d_air_t, theta_plus_t = build_adjacency_from_swarm_and_channel(
            swarm, channel, p_min=p_min
        )

        # TARJAN@t
        t0 = time.perf_counter()
        bridges_t = tarjan_bridges(adj_t)
        t1 = time.perf_counter()
        flags["TARJAN_t"].append(set(bridges_t))
        times_ms["TARJAN_t"].append((t1 - t0) * 1000.0)

        # EB-APPX(top-η)
        t0 = time.perf_counter()
        eb_edges = eb_appx_top_eta(adj_t, eta=eta, rng=rng)
        t1 = time.perf_counter()
        flags["EBAPPX"].append(eb_edges)
        times_ms["EBAPPX"].append((t1 - t0) * 1000.0)

        # CUT/KEC
        t0 = time.perf_counter()
        cut_edges = cut_kec_flag_edges(adj_t, frac=eta)
        t1 = time.perf_counter()
        flags["CUTKEC"].append(cut_edges)
        times_ms["CUTKEC"].append((t1 - t0) * 1000.0)

        # 2) ACLD: t anından t+Δt için tahmin
        t0 = time.perf_counter()
        adj_pred, d_hat, theta_minus, theta_plus_pred = acld._build_predicted_adjacency()
        comps_pred = ACLD.predict_components(adj_pred, d_hat, theta_minus, k_max=5)
        inter_pred, intra_pred = ACLD.catalogue_risky_links(
            comps_pred, adj_pred, d_hat, theta_minus, theta_plus_pred
        )
        t1 = time.perf_counter()
        # Makalede daha önemli olan inter-component riskli kenarlar:
        flags["ACLD"].append(set(inter_pred))
        times_ms["ACLD"].append((t1 - t0) * 1000.0)

        # 3) Sürüyü t+Δt anına taşı
        swarm.step_dynamics()

        # 4) t+Δt anındaki graph (adj_tp): ground-truth + TARJAN@(t+Δt)
        adj_tp, d_air_tp, theta_plus_tp = build_adjacency_from_swarm_and_channel(
            swarm, channel, p_min=p_min
        )

        # Ground-truth kritik kenarlar = Tarjan köprüleri
        Ck = ground_truth_critical_edges(adj_tp)
        critical_edges_per_step.append(Ck)

        # TARJAN@(t+Δt) (oracle)
        t0 = time.perf_counter()
        bridges_tp = tarjan_bridges(adj_tp)
        t1 = time.perf_counter()
        flags["TARJAN_tplus"].append(set(bridges_tp))
        times_ms["TARJAN_tplus"].append((t1 - t0) * 1000.0)

        # 5) Partition olayı (λ2 < threshold) – makaledekiyle aynı mantık
        cthr_tp = swarm.adaptive_cthr(d_air_tp)
        _, _, L_tp = build_weight_matrices(d_air_tp, cthr_tp)
        lam2 = algebraic_connectivity(L_tp)
        partition_events.append(lam2 < 1e-3)

    # --- Metrikleri episode bazında hesapla ---
    episode_metrics: Dict[str, Dict[str, float]] = {}

    for m in methods:
        pew = compute_pew(flags[m], critical_edges_per_step, partition_events)
        fp = compute_fp_rate(flags[m], critical_edges_per_step)
        f1 = compute_edge_f1(flags[m], critical_edges_per_step)
        lat = compute_detection_latency(
            flags[m], critical_edges_per_step, partition_events, dt
        )
        tstep = float(np.mean(times_ms[m])) if times_ms[m] else 0.0

        episode_metrics[m] = {
            "pew": pew,
            "fp": fp,
            "f1": f1,
            "lat": lat,
            "tstep": tstep,
        }

    return episode_metrics


# ============================================================================
# 13. Çoklu episode: n={150,1000} için ortalama ± %95 CI tablosu
# ============================================================================

def _mean_and_ci(values: List[float]) -> Tuple[float, float]:
    """
    Ortalama ve 95% CI (normal approx, 1.96 * s / sqrt(N)).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / math.sqrt(arr.size)
    return mean, ci


def run_baseline_table_experiment(
    n_list: Iterable[int] = (150, 1000),
    T: float = 5.0,
    dt: float = 0.02,
    density: str = "medium",
    jammer_db: float = 15.0,   # istersen 0.0 yap → jammer yok
    n_episodes: int = 5,
    eta: float = 0.05,
    p_min: float = 0.8,
):
    """
    Makaledeki Tablo~\\ref{tab:baseline_compare} için sayıları üretir.
    Her n için:
      - PEW, FP, Latency, F1, Time/step metriklerinin
        ortalama ± 95% CI çıktısını verir.
    """
    methods = ["ACLD", "TARJAN_t", "TARJAN_tplus", "EBAPPX", "CUTKEC"]
    metric_keys = ["pew", "fp", "lat", "f1", "tstep"]

    for n in n_list:
        print(f"\n=== Baseline experiment for n={n} ===")
        # Her yöntem/metrik için episode değerleri
        agg: Dict[str, Dict[str, List[float]]] = {
            m: {k: [] for k in metric_keys} for m in methods
        }

        for ep in range(n_episodes):
            print(f"  -> episode {ep+1}/{n_episodes} (seed={ep})")
            ep_metrics = simulate_episode_with_baselines(
                n=n,
                T=T,
                dt=dt,
                density=density,
                jammer_db=jammer_db,
                p_min=p_min,
                eta=eta,
                seed=ep,
            )
            for m in methods:
                for k in metric_keys:
                    agg[m][k].append(ep_metrics[m][k])

        # Konsola küçük bir tablo bastıralım
        print("\nMethod        PEW% (±CI)   FP% (±CI)    Lat(ms ±CI)   F1 (±CI)    Time/step(ms ±CI)")
        for m in methods:
            pew_mean, pew_ci = _mean_and_ci(agg[m]["pew"])
            fp_mean, fp_ci = _mean_and_ci(agg[m]["fp"])
            lat_mean, lat_ci = _mean_and_ci(agg[m]["lat"])
            f1_mean, f1_ci = _mean_and_ci(agg[m]["f1"])
            t_mean, t_ci = _mean_and_ci(agg[m]["tstep"])

            print(
                f"{m:10s}  "
                f"{pew_mean:5.1f}±{pew_ci:4.1f}  "
                f"{fp_mean:5.1f}±{fp_ci:4.1f}  "
                f"{lat_mean:7.2f}±{lat_ci:5.2f}  "
                f"{f1_mean:4.2f}±{f1_ci:4.2f}  "
                f"{t_mean:6.2f}±{t_ci:4.2f}"
            )

        print("\nBu bloktan çıkan sayıları doğrudan LaTeX tablosuna koyabilirsin.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACLD-UAV Python implementation (demo)")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "jamming", "runtime", "sweep", "paper", "baselines"],
        help="Çalışma modu: demo / jamming / runtime / sweep / paper / baselines",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        _demo()
    elif args.mode == "jamming":
        _demo_jamming()
    elif args.mode == "runtime":
        bench_runtime()
    elif args.mode == "sweep":
        simple_param_sweep()
    elif args.mode == "paper":
        run_paper_like_experiments()
    elif args.mode == "baselines":
        run_baseline_table_experiment()
