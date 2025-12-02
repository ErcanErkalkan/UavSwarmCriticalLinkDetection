#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baselines.py

ACLD vs. klasik graph baz çizgileri (TARJAN@t, TARJAN@(t+Δt),
Edge Betweenness, Min-cut / k-edge connectivity) için:

- Partition Early Warning (PEW)
- False Positive oranı
- Edge-level F1
- Detection latency (ms)

hesaplayan deney sürücüsü.

Bu dosya:
  - 2 temsilci senaryo tanımlar (S1: n=150, S2: n=500, medium density).
  - Her senaryo için N episode koşar.
  - Sonuçları CSV ve LaTeX tablo olarak çıktılar.
"""

import time
import random
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------
# 1. Senaryo konfigürasyonu
# ---------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    name: str
    n_uav: int
    density: str
    delta_t: float
    horizon_s: float
    num_episodes: int
    eta_eb: int            # Edge betweenness için seçilecek top-η
    seed_base: int = 1337
    # İstersen jamming, Rician vb. parametreleri de ekleyebilirsin
    jammer_dbm: float = 0.0


# ---------------------------------------------------------------------
# 2. Yardımcı graph fonksiyonları
# ---------------------------------------------------------------------


def is_connected_from_adj(A: np.ndarray) -> bool:
    """
    Basit BFS ile bağlantılılık testi.
    A: (n x n) 0/1 adjacency matrix (simetrik).
    """
    n = A.shape[0]
    if n == 0:
        return True

    visited = [False] * n
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        neighbors = np.where(A[u] > 0)[0]
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    return all(visited)


def adjacency_to_graph(A: np.ndarray) -> nx.Graph:
    """
    Numpy adjacency -> networkx.Graph
    """
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] != 0]
    G.add_edges_from(edges)
    return G


def label_critical_edges(A_hat_next: np.ndarray,
                         max_samples: int = 200) -> Set[Tuple[int, int]]:
    """
    Ground-truth critical edges @ t+Δt:
    - Önce grafın kendisinin bağlantılı olduğundan emin ol.
    - Sonra kenarları (i,j) tek tek kaldırıp bağlantılılığı test et.
    - Bağlantılılık bozuluyorsa (i,j) 'critical' olarak etiketle.
    - Büyük n için max_samples ile sınırlıyoruz (rastgele sampling).

    Dönüş: {(i,j), ...} şeklinde set, i<j.
    """
    n = A_hat_next.shape[0]
    base_connected = is_connected_from_adj(A_hat_next)
    if not base_connected:
        # Zaten parçalıysa, bu adımı ground-truth kritik olarak kullanmıyoruz.
        return set()

    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if A_hat_next[i, j] == 1]

    if len(edges) > max_samples:
        edges = random.sample(edges, max_samples)

    critical: Set[Tuple[int, int]] = set()
    for (i, j) in edges:
        A_tmp = A_hat_next.copy()
        A_tmp[i, j] = A_tmp[j, i] = 0
        if not is_connected_from_adj(A_tmp):
            critical.add((i, j))

    return critical


# ---------------------------------------------------------------------
# 3. Baz çizgi algoritmaları wrapper'ları
# ---------------------------------------------------------------------
# Burada ACLD dışındakiler pure-graph tabanlı.
# ---------------------------------------------------------------------


def run_tarjan_bridges(A: np.ndarray) -> Set[Tuple[int, int]]:
    """
    TARJAN@... -> köprü (bridge) kenarlarını döndürür.
    networkx.graphs.bridges fonksiyonu Tarjan temelli.
    """
    G = adjacency_to_graph(A)
    bridges = list(nx.bridges(G))
    # Kenarları (i<j) normal formuna getir
    return {tuple(sorted(e)) for e in bridges}


def run_edge_betweenness(A: np.ndarray, eta: int) -> Set[Tuple[int, int]]:
    """
    Edge betweenness centrality (Brandes).
    Büyük n için pahalı olabilir, gerekirse approximate versiyona geçebilirsin.
    """
    G = adjacency_to_graph(A)
    if G.number_of_edges() == 0:
        return set()

    # Tüm kenarlar için betweenness skoru
    scores = nx.edge_betweenness_centrality(G)
    # (edge, score) çiftlerini skora göre sırala
    sorted_edges = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [e for (e, _s) in sorted_edges[:eta]]
    return {tuple(sorted(e)) for e in selected}


def run_min_cut_based(A: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Global min-cut / k-edge connectivity tabanlı riskli kenarlar.
    Basit yaklaşım: tüm graf için min edge-cut bul ve o kesimdeki kenarları
    'risky' kabul et.
    """
    G = adjacency_to_graph(A)

    if G.number_of_edges() == 0 or not nx.is_connected(G):
        return set()

    # Global min edge-cut:
    cut_edges = nx.minimum_edge_cut(G)
    return {tuple(sorted(e)) for e in cut_edges}


# ---------------------------------------------------------------------
# 4. ACLD wrapper – şu an self-contained “akıllı placeholder”
# ---------------------------------------------------------------------


def acld_pipeline(state,
                  A_t: np.ndarray,
                  A_hat_next: np.ndarray) -> Set[Tuple[int, int]]:
    """
    ACLD için self-contained bir placeholder.

    Gerçek makale deneyinde:
      - Bu fonksiyonun içini kendi ACLD pipeline'ınla değiştirmelisin.
      - state, A_t, A_hat_next'ten gelen özellikleri kullanarak
        'gelecekte partition riski taşıyan' kenarları döndürmelisin.

    Buradaki placeholder mantığı:
      - A_hat_next grafını kullanır (yani geleceğe bakar).
      - Köprü kenarları (Tarjan) + en yüksek edge-betweenness'e sahip
        top-η kenarları 'risky' olarak işaretler.
    """
    G = adjacency_to_graph(A_hat_next)
    if G.number_of_edges() == 0:
        return set()

    # 1) Bridges
    bridges = list(nx.bridges(G))
    risky: Set[Tuple[int, int]] = {tuple(sorted(e)) for e in bridges}

    # 2) Edge betweenness (top η)
    scores = nx.edge_betweenness_centrality(G, normalized=True)
    m = len(scores)
    eta = max(1, int(0.05 * m))  # kenarların %5'i kadarını seç
    sorted_edges = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_edges = [e for (e, _s) in sorted_edges[:eta]]

    risky.update({tuple(sorted(e)) for e in top_edges})
    return risky


# ---------------------------------------------------------------------
# 5. Metrik tutucu sınıf
# ---------------------------------------------------------------------


class MethodStats:
    """
    Her method için metrikleri toplayan küçük bir sınıf.
    """

    def __init__(self, name: str):
        self.name = name
        # PEW: erken uyarı başarı oranı
        self.pew_hits = 0
        self.pew_total = 0

        # Edge-level F1 için:
        self.tp = 0
        self.fp = 0
        self.fn = 0

        # False positive rate -> FP / (TP + FP)
        # Detection latency (ms)
        self.latencies_ms: List[float] = []

        # Partition öncesi “warning” başlangıç zamanını tutmak için:
        self.warning_active = False
        self.warning_first_t: Optional[float] = None

    def update_edge_counts(self,
                           risky: Set[Tuple[int, int]],
                           critical: Set[Tuple[int, int]]):
        """
        Her zaman adımında edge-level TP/FP/FN güncelle.
        """
        inter = risky & critical
        only_risky = risky - critical
        only_critical = critical - risky

        self.tp += len(inter)
        self.fp += len(only_risky)
        self.fn += len(only_critical)

    def begin_warning_if_needed(self,
                                t: float,
                                risky: Set[Tuple[int, int]]):
        """
        Her adımda çağrılır.
        Eğer method bu adımda ilk defa risky non-empty flagliyorsa,
        warning durumuna geçer ve başlangıç zamanını kaydeder.
        """
        if risky and not self.warning_active:
            self.warning_active = True
            self.warning_first_t = t

    def register_partition_event(self,
                                 t: float,
                                 critical_now: Set[Tuple[int, int]],
                                 risky_now: Set[Tuple[int, int]]):
        """
        Partition olayı t anında (veya t->t+Δt aralığında) gerçekleştiğinde
        PEW ve latency açısından bu methodun durumunu günceller.

        Burada:
          - Eğer method hiç warning vermediyse -> PEW miss.
          - Eğer warning verdiyse:
              * risky_now ∩ critical_now boş değilse:
                  PEW hit + latency hesapla.
              * aksi halde -> PEW miss.
        """
        self.pew_total += 1

        # Hiç warning vermemişse direkt miss.
        if not self.warning_active or self.warning_first_t is None:
            return

        if risky_now & critical_now:
            self.pew_hits += 1
            latency = t - self.warning_first_t
            self.latencies_ms.append(latency * 1000.0)

        # Bu partition event kapandı, warning state reset
        self.warning_active = False
        self.warning_first_t = None

    def summary(self) -> Dict[str, float]:
        """
        Episode sonu / senaryo sonu özet metrikler.
        """
        pew = self.pew_hits / self.pew_total if self.pew_total > 0 else 0.0

        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        mean_lat = float(np.mean(self.latencies_ms)) if self.latencies_ms else np.nan

        return dict(
            method=self.name,
            PEW=pew,
            F1=f1,
            FP_rate=fp_rate,
            mean_latency_ms=mean_lat,
        )


# ---------------------------------------------------------------------
# 6. UAVSwarm stub – self-contained DummySwarm
# ---------------------------------------------------------------------


class DummySwarm:
    """
    Gerçek kodunda bu sınıf yok; bunun yerine:
      from uav_swarm import UAVSwarm
    vb. ile kendi simülatörünü import edeceksin.

    Burada self-contained bir dummy swarm var.
    """

    def __init__(self, config: ScenarioConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.n = config.n_uav
        self.t = 0.0

    @property
    def state(self):
        # Gerçek koddaki state yerine basit bir dict dönüyoruz.
        return dict(t=self.t)

    def build_adjacencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A_t ve A_hat_next üretir.

        Şimdilik tamamen random bir örnek üretiyoruz:
          - A_t: current graph
          - A_hat_next: bir sonraki zaman adımı için benzer yoğunlukta graph

        Gerçekte sen:
          - şu anki pozisyonlardan A_t,
          - velocity/threshold’lardan A_hat_next
        üretmelisin.
        """
        n = self.n
        A_t = np.zeros((n, n), dtype=int)
        A_hat = np.zeros((n, n), dtype=int)

        # Yoğunluk parametresi 'medium' ise p ~ 0.05 gibi alalım.
        if self.config.density == "low":
            p_now = 0.02
            p_next = 0.02
        elif self.config.density == "high":
            p_now = 0.10
            p_next = 0.10
        else:
            p_now = 0.05
            p_next = 0.05

        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < p_now:
                    A_t[i, j] = A_t[j, i] = 1
                if self.rng.random() < p_next:
                    A_hat[i, j] = A_hat[j, i] = 1

        return A_t, A_hat

    def step(self):
        """
        Simülasyonu bir adım ilerlet.
        """
        self.t += self.config.delta_t


def make_swarm(config: ScenarioConfig, rng: np.random.Generator):
    """
    Gerçek kodunda DummySwarm yerine kendi UAVSwarm sınıfını döndür.

    Örn:
        from my_swarm_module import UAVSwarm
        return UAVSwarm(config=config, rng=rng)
    """
    return DummySwarm(config=config, rng=rng)


# ---------------------------------------------------------------------
# 7. Bir episode çalıştırma
# ---------------------------------------------------------------------


def run_episode(config: ScenarioConfig, episode_idx: int) -> Dict[str, MethodStats]:
    """
    Tek bir episode koşar; her method için MethodStats objesini döndürür.
    """
    seed = config.seed_base + episode_idx
    rng = np.random.default_rng(seed)
    random.seed(seed)

    swarm = make_swarm(config, rng)

    # Method listesi
    method_names = [
        "ACLD",
        "TARJAN_t",
        "TARJAN_tplus",
        "EB_APPX",
        "CUT_KEC",
    ]

    stats = {name: MethodStats(name=name) for name in method_names}

    num_steps = int(config.horizon_s / config.delta_t)

    # Partition event tespitini basitçe: A_hat_next bağlantılı mı?
    # bir partition epizodu bittiğinde resetleniyor.
    for k in range(num_steps):
        t = k * config.delta_t

        # 1) Swarm state & adjacency'ler
        state = swarm.state
        A_t, A_hat_next = swarm.build_adjacencies()

        # 2) Ground-truth critical edges @ t+Δt
        critical_edges = label_critical_edges(A_hat_next)

        # 3) Her method için risky setleri hesapla
        risky_sets: Dict[str, Set[Tuple[int, int]]] = {}

        risky_sets["ACLD"] = acld_pipeline(state, A_t, A_hat_next)
        risky_sets["TARJAN_t"] = run_tarjan_bridges(A_t)
        risky_sets["TARJAN_tplus"] = run_tarjan_bridges(A_hat_next)
        risky_sets["EB_APPX"] = run_edge_betweenness(A_t, eta=config.eta_eb)
        risky_sets["CUT_KEC"] = run_min_cut_based(A_t)

        # 4) Her method için edge-level TP/FP/FN güncelle
        for name in method_names:
            stats[name].update_edge_counts(risky_sets[name], critical_edges)

        # 5) Partition olacak mı? (t->t+Δt aralığında)
        partition_soon = not is_connected_from_adj(A_hat_next)

        # 6) Warning state & latency için method'ları güncelle
        for name in method_names:
            stats[name].begin_warning_if_needed(t, risky_sets[name])

        if partition_soon:
            # Bu adımı partition olayı olarak kabul ediyoruz;
            # detection latency: ilk warning zamanı ile bu an arasındaki fark.
            for name in method_names:
                stats[name].register_partition_event(
                    t=t,
                    critical_now=critical_edges,
                    risky_now=risky_sets[name],
                )

        # 7) Simülasyonu bir adım ilerlet
        swarm.step()

    return stats


# ---------------------------------------------------------------------
# 8. Senaryo bazında çoklu episode koşma ve özet tablo üretme
# ---------------------------------------------------------------------


def run_scenario(config: ScenarioConfig) -> pd.DataFrame:
    """
    Belirli bir senaryo için config.num_episodes kadar episode koşar.
    Tüm method'ların ortalama metriklerini DataFrame olarak döndürür.
    """
    print(f"[INFO] Running scenario {config.name} "
          f"n={config.n_uav}, episodes={config.num_episodes}")

    method_names = [
        "ACLD",
        "TARJAN_t",
        "TARJAN_tplus",
        "EB_APPX",
        "CUT_KEC",
    ]

    # Tüm episode'ların stats'larını birleştirmek için:
    agg_stats: Dict[str, MethodStats] = {
        name: MethodStats(name=name) for name in method_names
    }

    # Detection latency’leri birleştirebilmek için ekstra liste
    latencies_all: Dict[str, List[float]] = {name: [] for name in method_names}

    t0 = time.time()

    for e in range(config.num_episodes):
        ep_stats = run_episode(config, episode_idx=e)

        for name in method_names:
            s_ep = ep_stats[name]
            s_agg = agg_stats[name]

            # scalar sayıları topla
            s_agg.pew_hits += s_ep.pew_hits
            s_agg.pew_total += s_ep.pew_total
            s_agg.tp += s_ep.tp
            s_agg.fp += s_ep.fp
            s_agg.fn += s_ep.fn

            # latency listelerini birleştir
            latencies_all[name].extend(s_ep.latencies_ms)

    # latency'leri geri yaz
    for name in method_names:
        agg_stats[name].latencies_ms = latencies_all[name]

    elapsed = time.time() - t0
    print(f"[INFO] Scenario {config.name} finished in {elapsed:.2f} s.")

    # DataFrame'e dök
    rows = []
    for name in method_names:
        rows.append(agg_stats[name].summary())

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# 9. Ana fonksiyon – S1 & S2 senaryolarını çalıştır ve sonuçları kaydet
# ---------------------------------------------------------------------


def main():
    # Örnek olarak iki senaryo:
    S1 = ScenarioConfig(
        name="S1_n150_medium",
        n_uav=150,
        density="medium",
        delta_t=0.01,
        horizon_s=20.0,      # quick test; makalede 200 s vb. yaparsın
        num_episodes=20,     # makalede 100-200 gibi arttırabilirsin
        eta_eb=50,
        seed_base=1337,
    )

    S2 = ScenarioConfig(
        name="S2_n500_medium",
        n_uav=500,
        density="medium",
        delta_t=0.01,
        horizon_s=20.0,
        num_episodes=20,
        eta_eb=100,
        seed_base=2025,
    )

    scenarios = [S1, S2]

    all_results = []
    for cfg in scenarios:
        df = run_scenario(cfg)
        df["scenario"] = cfg.name

        # CSV olarak kaydet
        csv_path = f"results_baselines_{cfg.name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV: {csv_path}")

        # LaTeX tablo çıktısı:
        print(f"\n[LaTeX table for {cfg.name}]\n")
        print(df.to_latex(index=False,
                          float_format="%.3f",
                          columns=["method", "PEW", "F1", "FP_rate", "mean_latency_ms"]))

        all_results.append(df)

    # İster birleşik tek bir CSV de yazabilirsin:
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv("results_baselines_all.csv", index=False)
    print("[INFO] Saved combined CSV: results_baselines_all.csv")


if __name__ == "__main__":
    main()
