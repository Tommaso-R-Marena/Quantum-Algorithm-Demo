"""
baselines.py
Classical baseline algorithms for HP lattice protein folding.

Implements five baselines of increasing sophistication:
  1. ExactSolver       — DFS with lower-bound pruning (branch-and-bound)
  2. GreedyLocalSearch  — hill-climber from random initial conformations
  3. SimulatedAnnealing — SA with pull-move neighbourhood
  4. GeneticAlgorithm   — GA with turn-sequence crossover
  5. ReplicaExchangeMC  — parallel-tempering Monte Carlo (REMC)

All baselines use the same energy evaluation from protein.py and the
same lattice geometry from lattice.py, ensuring fair comparison with
quantum algorithms.

References:
  [1] Dill et al., Protein Sci. 4, 561 (1995) — HP model
  [2] Lesh et al., RECOMB 2003 — pull moves
  [3] Unger & Moult, J. Mol. Biol. 231, 75 (1993) — GA for folding
  [4] Hukushima & Nemoto, J. Phys. Soc. Jpn. 65, 1604 (1996) — REMC
"""

from __future__ import annotations

import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from copy import deepcopy

from ..core.lattice import CubicLattice, CoordinateFrame
from ..core.protein import Protein


# ═══════════════════════════════════════════════════════════════════════════
# 1. Exact Solver (DFS with branch-and-bound pruning)
# ═══════════════════════════════════════════════════════════════════════════

class ExactSolver:
    """
    DFS brute-force search over all self-avoiding walks on the 3D cubic lattice.

    Uses branch-and-bound pruning: at each partial configuration, compute
    a lower bound on the achievable energy. If this bound is ≥ current best,
    prune the branch.

    Lower bound: current_contacts + max_future_contacts ≤ best_so_far.
    max_future_contacts is bounded by the number of remaining H-H pairs
    that could possibly form contacts.

    For N ≤ 12 this runs in seconds; for N ≤ 15 in minutes.
    """

    def __init__(self, protein: Protein):
        self.protein = protein
        self.best_energy = float("inf")
        self.best_coords: List[np.ndarray] = []
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def solve(self) -> Tuple[float, List[np.ndarray]]:
        """
        Find the globally optimal conformation.

        Returns
        -------
        best_energy : float
        best_coords : list of np.ndarray
        """
        initial_coords = [
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64),
        ]
        occupied = {(0, 0, 0), (1, 0, 0)}
        self._dfs(2, initial_coords, occupied, last_dir=0)
        return self.best_energy, self.best_coords

    def _dfs(
        self,
        bead_idx: int,
        current_coords: List[np.ndarray],
        occupied: set,
        last_dir: int,
    ):
        """Recursive DFS with pruning."""
        # Base case: all beads placed
        if bead_idx == self.protein.n:
            self.nodes_explored += 1
            e = self.protein.evaluate_energy(current_coords, collision_penalty=0.0)
            if e < self.best_energy:
                self.best_energy = e
                self.best_coords = [c.copy() for c in current_coords]
            return

        # --- Branch-and-bound pruning ---
        # Compute partial energy (contacts from beads already placed)
        partial_e = self._partial_energy(current_coords)

        # Upper bound on future contacts (negative energies)
        remaining_h = sum(
            1 for k in range(bead_idx, self.protein.n)
            if self.protein.sequence[k] == "H"
        )
        placed_h = sum(
            1 for k in range(bead_idx)
            if self.protein.sequence[k] == "H"
        )
        # Maximum new contacts: each remaining H can contact at most
        # 4 neighbours (5 minus the bonded one), and each placed H
        max_new_contacts = remaining_h * placed_h + remaining_h * (remaining_h - 1) // 2
        # Each contact contributes -1 in HP model
        best_possible = partial_e - max_new_contacts

        if best_possible >= self.best_energy:
            self.nodes_pruned += 1
            return

        last_pos = current_coords[-1]

        # Try all 6 directions (excluding immediate reverse)
        non_reverse = CubicLattice.get_non_reverse_directions(last_dir)

        for d in non_reverse:
            vec = CubicLattice.get_vector_from_int(d)
            new_pos = last_pos + vec
            new_key = tuple(new_pos)

            if new_key not in occupied:
                current_coords.append(new_pos)
                occupied.add(new_key)
                self._dfs(bead_idx + 1, current_coords, occupied, d)
                current_coords.pop()
                occupied.remove(new_key)

    def _partial_energy(self, coords: List[np.ndarray]) -> float:
        """Compute energy for partially placed beads (no collision penalty)."""
        energy = 0.0
        n = len(coords)
        for i in range(n):
            for j in range(i + 2, n):
                d2 = int(np.sum((coords[i] - coords[j]) ** 2))
                energy += self.protein.get_interaction_energy(i, j, d2)
        return energy


# ═══════════════════════════════════════════════════════════════════════════
# 2. Greedy Local Search
# ═══════════════════════════════════════════════════════════════════════════

class GreedyLocalSearch:
    """
    Stochastic hill-climbing from random initial conformations.

    Neighbourhood: single-bead end moves and corner moves.
    Repeats n_restarts times and returns the best result.
    """

    def __init__(
        self,
        protein: Protein,
        n_restarts: int = 100,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        self.protein = protein
        self.n_restarts = n_restarts
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def solve(self) -> Tuple[float, List[np.ndarray], Dict]:
        """
        Returns (best_energy, best_coords, info_dict).
        """
        best_e = float("inf")
        best_c = []
        energies_over_restarts = []

        for _ in range(self.n_restarts):
            coords = self._random_saw()
            if coords is None:
                continue
            e = self.protein.evaluate_energy(coords, collision_penalty=0.0)

            for _ in range(self.max_steps):
                neighbour = self._random_move(coords)
                if neighbour is None:
                    break
                ne = self.protein.evaluate_energy(neighbour, collision_penalty=0.0)
                if ne <= e:
                    coords = neighbour
                    e = ne

            energies_over_restarts.append(e)
            if e < best_e:
                best_e = e
                best_c = [c.copy() for c in coords]

        return best_e, best_c, {
            "n_restarts": self.n_restarts,
            "energies": energies_over_restarts,
        }

    def _random_saw(self) -> Optional[List[np.ndarray]]:
        """Generate a random self-avoiding walk."""
        coords = [
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64),
        ]
        occupied = {(0, 0, 0), (1, 0, 0)}

        for _ in range(self.protein.n - 2):
            last = coords[-1]
            dirs = list(range(6))
            self.rng.shuffle(dirs)
            placed = False
            for d in dirs:
                vec = CubicLattice.get_vector_from_int(d)
                new_pos = last + vec
                key = tuple(new_pos)
                if key not in occupied:
                    coords.append(new_pos.copy())
                    occupied.add(key)
                    placed = True
                    break
            if not placed:
                return None  # trapped
        return coords

    def _random_move(self, coords: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """Try a random end move or corner move."""
        n = len(coords)
        # End moves: move bead 0 or bead N-1 to an adjacent empty site
        # Corner moves: move bead i (1 ≤ i ≤ N-2) if it has a corner
        occupied = {tuple(c) for c in coords}
        move_type = self.rng.integers(0, 3)

        if move_type == 0 and n >= 2:
            # Move first bead
            return self._try_end_move(coords, 0, occupied)
        elif move_type == 1 and n >= 2:
            # Move last bead
            return self._try_end_move(coords, n - 1, occupied)
        else:
            # Corner move on a random interior bead
            if n < 3:
                return None
            i = self.rng.integers(1, n - 1)
            return self._try_corner_move(coords, i, occupied)

    def _try_end_move(
        self, coords: List[np.ndarray], idx: int, occupied: set
    ) -> Optional[List[np.ndarray]]:
        """Try to move an end bead to a vacant neighbour of its bonded partner."""
        n = len(coords)
        if idx == 0:
            anchor = coords[1]
        else:
            anchor = coords[n - 2]

        for d in range(6):
            vec = CubicLattice.get_vector_from_int(d)
            new_pos = anchor + vec
            key = tuple(new_pos)
            if key not in occupied or key == tuple(coords[idx]):
                new_coords = [c.copy() for c in coords]
                new_coords[idx] = new_pos.copy()
                if CubicLattice.is_self_avoiding(new_coords):
                    return new_coords
        return None

    def _try_corner_move(
        self, coords: List[np.ndarray], idx: int, occupied: set
    ) -> Optional[List[np.ndarray]]:
        """Try to move bead idx to a corner position."""
        prev_pos = coords[idx - 1]
        next_pos = coords[idx + 1]

        # A corner move is valid if prev and next are diagonal neighbours
        diff = next_pos - prev_pos
        d2 = int(np.sum(diff * diff))
        if d2 != 2:
            return None  # not a corner

        # The two possible corner positions
        positions = []
        for d in range(6):
            vec = CubicLattice.get_vector_from_int(d)
            candidate = prev_pos + vec
            if (
                int(np.sum((candidate - next_pos) ** 2)) == 1
                and not np.array_equal(candidate, coords[idx])
            ):
                key = tuple(candidate)
                if key not in occupied:
                    positions.append(candidate)

        if not positions:
            return None

        new_pos = positions[self.rng.integers(0, len(positions))]
        new_coords = [c.copy() for c in coords]
        new_coords[idx] = new_pos.copy()
        return new_coords


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simulated Annealing (SA) with pull-move neighbourhood
# ═══════════════════════════════════════════════════════════════════════════

class SimulatedAnnealing:
    """
    Simulated Annealing for HP lattice protein folding.

    Uses a combination of end-moves, corner-moves, and crankshaft-moves
    as the neighbourhood. Temperature schedule is exponential decay.

    Parameters
    ----------
    protein : Protein
    t_start : float
        Initial temperature.
    t_end : float
        Final temperature.
    n_steps : int
        Total number of MC steps.
    n_restarts : int
        Number of independent SA runs.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        protein: Protein,
        t_start: float = 5.0,
        t_end: float = 0.01,
        n_steps: int = 10000,
        n_restarts: int = 10,
        seed: Optional[int] = None,
    ):
        self.protein = protein
        self.t_start = t_start
        self.t_end = t_end
        self.n_steps = n_steps
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self._greedy = GreedyLocalSearch(protein, n_restarts=1, max_steps=0, seed=seed)

    def solve(self) -> Tuple[float, List[np.ndarray], Dict]:
        """Run SA and return (best_energy, best_coords, info_dict)."""
        global_best_e = float("inf")
        global_best_c = []
        all_trajectories = []

        for restart in range(self.n_restarts):
            coords = self._greedy._random_saw()
            if coords is None:
                continue

            e = self.protein.evaluate_energy(coords, collision_penalty=0.0)
            best_e = e
            best_c = [c.copy() for c in coords]
            trajectory = [e]

            cooling_rate = (self.t_end / self.t_start) ** (1.0 / self.n_steps)

            T = self.t_start
            for step in range(self.n_steps):
                neighbour = self._greedy._random_move(coords)
                if neighbour is None:
                    T *= cooling_rate
                    continue

                ne = self.protein.evaluate_energy(neighbour, collision_penalty=0.0)
                delta = ne - e

                if delta <= 0 or self.rng.random() < np.exp(-delta / max(T, 1e-15)):
                    coords = neighbour
                    e = ne

                if e < best_e:
                    best_e = e
                    best_c = [c.copy() for c in coords]

                trajectory.append(e)
                T *= cooling_rate

            all_trajectories.append(trajectory)
            if best_e < global_best_e:
                global_best_e = best_e
                global_best_c = best_c

        return global_best_e, global_best_c, {
            "trajectories": all_trajectories,
            "n_restarts": self.n_restarts,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Genetic Algorithm (GA)
# ═══════════════════════════════════════════════════════════════════════════

class GeneticAlgorithm:
    """
    Genetic Algorithm for lattice protein folding on turn sequences.

    Individuals are represented as turn sequences (integers 0–4 for
    5 non-reverse absolute directions). Crossover operates on the
    turn sequence; mutation flips random turns. Fitness is the
    negative energy (maximise fitness = minimise energy).

    Parameters
    ----------
    protein : Protein
    pop_size : int
        Population size.
    n_generations : int
        Number of generations.
    mutation_rate : float
        Per-gene mutation probability.
    elite_frac : float
        Fraction of population preserved as elites.
    seed : int, optional
    """

    def __init__(
        self,
        protein: Protein,
        pop_size: int = 100,
        n_generations: int = 200,
        mutation_rate: float = 0.1,
        elite_frac: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.protein = protein
        self.n_genes = protein.n - 2  # number of variable links
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.rng = np.random.default_rng(seed)

    def solve(self) -> Tuple[float, List[np.ndarray], Dict]:
        """Run GA and return (best_energy, best_coords, info_dict)."""
        # Initialise population
        population = self._init_population()
        fitness = np.array([self._evaluate(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_e = fitness[best_idx]
        best_ind = population[best_idx].copy()
        best_gen_e = [best_e]

        n_elite = max(1, int(self.elite_frac * self.pop_size))

        for gen in range(self.n_generations):
            # Selection (tournament)
            new_pop = []

            # Elitism
            elite_indices = np.argsort(fitness)[:n_elite]
            for ei in elite_indices:
                new_pop.append(population[ei].copy())

            # Fill rest via crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop[:self.pop_size]
            fitness = np.array([self._evaluate(ind) for ind in population])

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_e:
                best_e = fitness[gen_best_idx]
                best_ind = population[gen_best_idx].copy()

            best_gen_e.append(best_e)

        # Convert best individual to coordinates
        best_coords = self._decode(best_ind)

        return best_e, best_coords, {
            "generation_best": best_gen_e,
            "n_generations": self.n_generations,
        }

    def _init_population(self) -> List[np.ndarray]:
        """Random population of turn sequences."""
        pop = []
        for _ in range(self.pop_size):
            # Each gene is a direction 0-4 (5 non-reverse directions)
            ind = self.rng.integers(0, 5, size=self.n_genes)
            pop.append(ind)
        return pop

    def _evaluate(self, individual: np.ndarray) -> float:
        """Evaluate energy of a turn-sequence individual."""
        coords = self._decode(individual)
        if not CubicLattice.is_self_avoiding(coords):
            return 1000.0  # penalty for invalid
        return self.protein.evaluate_energy(coords, collision_penalty=0.0)

    def _decode(self, individual: np.ndarray) -> List[np.ndarray]:
        """Convert turn-sequence individual to coordinates."""
        # Map 0–4 to non-reverse absolute directions from last direction
        coords = [
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64),
        ]
        last_dir = 0  # +x

        for gene in individual:
            non_rev = CubicLattice.get_non_reverse_directions(last_dir)
            d = non_rev[min(int(gene), len(non_rev) - 1)]
            vec = CubicLattice.get_vector_from_int(d)
            coords.append(coords[-1] + vec)
            last_dir = d

        return coords

    def _tournament_select(
        self, population: List[np.ndarray], fitness: np.ndarray, k: int = 3
    ) -> np.ndarray:
        """Tournament selection with tournament size k."""
        indices = self.rng.integers(0, len(population), size=k)
        best = indices[np.argmin(fitness[indices])]
        return population[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        point = self.rng.integers(1, self.n_genes)
        child = np.concatenate([p1[:point], p2[point:]])
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Per-gene mutation."""
        for i in range(len(individual)):
            if self.rng.random() < self.mutation_rate:
                individual[i] = self.rng.integers(0, 5)
        return individual


# ═══════════════════════════════════════════════════════════════════════════
# 5. Replica Exchange Monte Carlo (REMC)
# ═══════════════════════════════════════════════════════════════════════════

class ReplicaExchangeMC:
    """
    Replica Exchange Monte Carlo (Parallel Tempering) for protein folding.

    Runs multiple SA replicas at different temperatures and periodically
    attempts to swap adjacent replicas (Metropolis-Hastings acceptance).
    This helps escape local minima by allowing high-temperature replicas
    to explore broadly while low-temperature replicas refine.

    Parameters
    ----------
    protein : Protein
    n_replicas : int
        Number of temperature replicas.
    t_min : float
        Lowest temperature.
    t_max : float
        Highest temperature.
    n_steps : int
        MC steps per replica between swap attempts.
    n_exchanges : int
        Number of exchange rounds.
    seed : int, optional
    """

    def __init__(
        self,
        protein: Protein,
        n_replicas: int = 8,
        t_min: float = 0.1,
        t_max: float = 10.0,
        n_steps: int = 500,
        n_exchanges: int = 100,
        seed: Optional[int] = None,
    ):
        self.protein = protein
        self.n_replicas = n_replicas
        self.n_steps = n_steps
        self.n_exchanges = n_exchanges
        self.rng = np.random.default_rng(seed)

        # Geometric temperature ladder
        self.temperatures = np.geomspace(t_min, t_max, n_replicas)

        self._greedy = GreedyLocalSearch(protein, n_restarts=1, max_steps=0, seed=seed)

    def solve(self) -> Tuple[float, List[np.ndarray], Dict]:
        """Run REMC and return (best_energy, best_coords, info_dict)."""
        # Initialise replicas
        replicas = []
        energies = []
        for _ in range(self.n_replicas):
            coords = self._greedy._random_saw()
            attempts = 0
            while coords is None and attempts < 100:
                coords = self._greedy._random_saw()
                attempts += 1
            if coords is None:
                coords = [
                    np.array([0, 0, 0], dtype=np.int64),
                    np.array([1, 0, 0], dtype=np.int64),
                ]
                for k in range(self.protein.n - 2):
                    coords.append(coords[-1] + np.array([1, 0, 0], dtype=np.int64) * (k + 2))
            replicas.append(coords)
            energies.append(
                self.protein.evaluate_energy(coords, collision_penalty=0.0)
            )

        best_e = min(energies)
        best_c = [c.copy() for c in replicas[np.argmin(energies)]]
        swap_count = 0
        history = [best_e]

        for exchange in range(self.n_exchanges):
            # MC steps within each replica
            for r in range(self.n_replicas):
                T = self.temperatures[r]
                for _ in range(self.n_steps):
                    neighbour = self._greedy._random_move(replicas[r])
                    if neighbour is None:
                        continue
                    ne = self.protein.evaluate_energy(
                        neighbour, collision_penalty=0.0
                    )
                    delta = ne - energies[r]
                    if delta <= 0 or self.rng.random() < np.exp(
                        -delta / max(T, 1e-15)
                    ):
                        replicas[r] = neighbour
                        energies[r] = ne

                    if energies[r] < best_e:
                        best_e = energies[r]
                        best_c = [c.copy() for c in replicas[r]]

            # Attempt swaps between adjacent replicas
            for r in range(self.n_replicas - 1):
                beta_lo = 1.0 / max(self.temperatures[r], 1e-15)
                beta_hi = 1.0 / max(self.temperatures[r + 1], 1e-15)
                delta_beta = beta_lo - beta_hi
                delta_e = energies[r] - energies[r + 1]

                if delta_beta * delta_e >= 0 or self.rng.random() < np.exp(
                    delta_beta * delta_e
                ):
                    replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                    energies[r], energies[r + 1] = energies[r + 1], energies[r]
                    swap_count += 1

            history.append(best_e)

        return best_e, best_c, {
            "swap_count": swap_count,
            "history": history,
            "n_exchanges": self.n_exchanges,
        }
