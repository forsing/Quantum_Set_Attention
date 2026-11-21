# Quantum_Set_Attention



from __future__ import annotations

import csv
import hashlib
import itertools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, RYGate, StatePreparation
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4724_k93.csv"),
    Path("/data/loto7_4724_k93_loto_2984.csv"),
    Path("/data/loto7_4724_k93_loto_plus_1740.csv"),
)



N = 39
K = 7
HEADS = 2
WINDOW = 32

REGULARIZATIONS = (1.0, 0.1, 0.01)
TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

GRADIENT_TOLERANCE = 2e-6
BFGS_ITERATIONS = 650
CG_ITERATIONS = 850
MAX_CONSECUTIVE = 4

DIAGNOSTICS = False


# ------------------------------------------------------------
# MODEL
#
# Ovo je moja adaptacija ideje visеglave paznje.
#
# Q/K:
#   Svako prethodno kolo je skup, a ne niz sedam pozicija.
#   Ucene amplitude predstavljaju prisutne brojeve.
#   Ucene faze razlikuju query od key reprezentacije.
#   Dva vremenska para sin/cos kodiraju polozaj kola.
#   Query i key stanja staju u sest qubita.
#
#   Skor paznje je naucena temperatura pomnozena
#   kvadratom apsolutnog preklapanja Q/K amplituda.
#
# V:
#   Svako dostupno kolo C nosi normalizovano kvantno stanje
#   nad CELOVITIM kandidatima S sa tacno K elemenata:
#
#   |V_h(C)> =
#       sum_S sqrt(exp(sum_{j in S} v_j) * kernel_h(|S & C|) / Z_h(C))
#       |S>
#
#   kernel nastaje iz naucenih kontrolisanih RY rotacija.
#   v_j su parametri amplitudnog filtera, ne linearni readout.
#
#   Broj zajednickih elemenata r=0..K nije predstavljanje
#   sedmorke preko razmaka.
#
# Izlaz:
#   Paznja mesa normalizovana V stanja kao matricu gustine.
#   Bornova verovatnoca cele sedmorke je:
#
#   P(S | istorija) =
#       sum_h pi_h sum_m attention_hm * P_h(S | C_m)
#
#   Mesanje, softmax i optimizacija su klasicni.
#   Q/K amplitude, RY amplitude i amplitudni filter
#   predstavljaju kvantni deo lokalne simulacije.
#
# Normalizacija:
#   Tacni elementarni simetricni polinomi posebno za brojeve
#   unutar i izvan istorijskog skupa C.
#
# NEXT:
#   Tacna pretraga maksimuma sa dokazivom gornjom granicom.
#   Nema pohlepnog izbora konacnih sedam brojeva.
#
# Logicki izlaz |S> ima kanonsko kodiranje u sedam
# sestokubitnih registara. Veliki statevector se ne alocira.
#
# Prozor je samo ulaz jednog uzorka.
# Zavrsno ucenje obuhvata sve istorijske prelaze.
# ------------------------------------------------------------


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_draws(path):
    if not path.is_file():
        raise FileNotFoundError(f"CSV nije pronadjen: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample, delimiters=",;\t "
            )
        except csv.Error as exc:
            raise ValueError(
                "Nije prepoznat razdelnik CSV-a."
            ) from exc

        for line_number, row in enumerate(
            csv.reader(handle, dialect), start=1
        ):
            fields = [field.strip() for field in row if field.strip()]

            if not fields:
                continue

            if len(fields) != K:
                raise ValueError(
                    f"Red {line_number}: ocekujem {K} brojeva."
                )

            try:
                values = [int(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"Red {line_number}: neispravan ceo broj."
                ) from exc

            if (
                len(set(values)) != K
                or min(values) < 1
                or max(values) > N
            ):
                raise ValueError(
                    f"Red {line_number}: neispravna kombinacija."
                )

            rows.append(sorted(value - 1 for value in values))

    draws = np.asarray(rows, dtype=np.int64)

    if len(draws) < 100:
        raise ValueError("Potrebno je najmanje 100 redova.")

    return draws


def binary_draws(draws, n):
    result = np.zeros((len(draws), n), dtype=np.float64)
    result[np.arange(len(draws))[:, None], draws] = 1.0
    return result


class ShellTree:
    """
    Normalizovano RY stablo nad brojem zajednickih elemenata.
    Kombinatorni brojevi sluze normalizaciji i pocetnoj tacki.
    Parametri stabla se uce iz NLL-a cele kombinacije.
    """

    def __init__(self, n, k):
        require(n >= 2 * k, "Ova konstrukcija zahteva N >= 2K.")

        self.n = n
        self.k = k
        self.size = k + 1
        self.bits = max(1, (self.size - 1).bit_length())

        self.multiplicity = np.array(
            [
                math.comb(k, r) * math.comb(n - k, k - r)
                for r in range(k + 1)
            ],
            dtype=np.float64,
        )

        self.nodes = []
        self.fixed = []
        base = []

        for bit in range(self.bits):
            for prefix in range(1 << bit):
                mask = (1 << bit) - 1
                members = [
                    r for r in range(self.size)
                    if (r & mask) == prefix
                ]

                if not members:
                    continue

                zero = [r for r in members if ((r >> bit) & 1) == 0]
                one = [r for r in members if ((r >> bit) & 1) == 1]

                if zero and one:
                    mass0 = self.multiplicity[zero].sum()
                    mass1 = self.multiplicity[one].sum()

                    base.append(
                        (mass1 - mass0)
                        / (2.0 * math.sqrt(mass0 * mass1))
                    )
                    self.nodes.append((bit, prefix))

                elif one:
                    self.fixed.append((bit, prefix))

        self.base = np.asarray(base, dtype=np.float64)
        self.sign = np.zeros((self.size, len(base)))

        for node, (bit, prefix) in enumerate(self.nodes):
            mask = (1 << bit) - 1

            for r in range(self.size):
                if (r & mask) == prefix:
                    self.sign[r, node] = (
                        1.0 if ((r >> bit) & 1) else -1.0
                    )

        self.visit = np.abs(self.sign)

    def evaluate(self, delta):
        z = self.base + delta
        radius = np.hypot(1.0, z)

        log_mass = (
            self.sign @ np.arcsinh(z)
            - self.visit @ np.log(2.0 * radius)
        )

        derivative = (
            self.sign / radius[None, :]
            - self.visit * (z / radius / radius)[None, :]
        )

        mass = np.exp(log_mass)

        require(
            np.all(mass > 0.0)
            and abs(mass.sum() - 1.0) < 1e-10,
            "RY raspodela nije normalizovana.",
        )

        return log_mass, derivative


class Layout:
    def __init__(self, n, k, heads):
        self.n = n
        self.k = k
        self.heads = heads
        self.tree = ShellTree(n, k)

        position = 0

        self.embedding = slice(position, position + n)
        position += n

        self.value = slice(position, position + n)
        position += n

        self.head = []

        for h in range(heads):
            item = {}

            item["phase"] = slice(position, position + n)
            position += n

            item["mix"] = position
            position += 1

            item["frequency"] = slice(position, position + 2)
            position += 2

            item["gain"] = position
            position += 1

            size = len(self.tree.base)
            item["shell"] = slice(position, position + size)
            position += size

            fractions = np.array(
                [(h + 1) / 8.0, (h + 1) / 32.0]
            )

            require(
                np.all((fractions > 0) & (fractions < 1)),
                "Neispravan broj glava.",
            )

            item["frequency_base"] = np.log(
                fractions / (1.0 - fractions)
            )
            item["phase_base"] = 0.35 * (h + 1)
            self.head.append(item)

        self.head_mix = slice(position, position + heads)
        position += heads
        self.size = position

    def initial(self):
        result = np.zeros(self.size, dtype=np.float64)

        for h, item in enumerate(self.head):
            count = item["shell"].stop - item["shell"].start

            result[item["shell"]] = (
                0.08 * np.sin(
                    np.arange(count, dtype=np.float64)
                    + 0.7 + 1.3 * h
                )
            )

        return result


class Dataset:
    def __init__(self, history, queries, window, observed=True):
        self.queries = np.asarray(queries, dtype=np.int64)
        require(len(self.queries) > 0, "Prazan skup uzoraka.")

        last_query = int(self.queries.max())
        require(
            self.queries.min() >= 0 and last_query < len(history),
            "Neispravan indeks istorije.",
        )

        # Centri V stanja nikada ne ukljucuju buduci ishod.
        self.centers = np.ascontiguousarray(
            history[:last_query + 1], dtype=np.float64
        )

        self.lags = np.arange(window, dtype=np.int64)
        raw_memory = self.queries[:, None] - self.lags[None, :]
        self.valid = raw_memory >= 0
        self.memory = np.maximum(raw_memory, 0)

        self.intersections = np.ascontiguousarray(
            self.centers[self.queries, None, :]
            * self.centers[self.memory],
            dtype=np.float64,
        )

        self.targets = None
        self.overlap = None

        if observed:
            require(
                last_query + 1 < len(history),
                "Nedostaje ciljni ishod.",
            )

            self.targets = np.ascontiguousarray(
                history[self.queries + 1], dtype=np.float64
            )

            self.overlap = np.einsum(
                "tn,twn->tw",
                self.targets,
                self.centers[self.memory],
            ).astype(np.int64)

    @property
    def size(self):
        return len(self.queries)


def value_prefixes(centers, weights, k):
    count, n = centers.shape

    inside = np.zeros((count, n + 1, k + 1))
    outside = np.zeros_like(inside)

    inside[:, 0, 0] = 1.0
    outside[:, 0, 0] = 1.0

    for j in range(n):
        inside[:, j + 1] = inside[:, j]
        outside[:, j + 1] = outside[:, j]

        wi = centers[:, j] * weights[j]
        wo = (1.0 - centers[:, j]) * weights[j]

        inside[:, j + 1, 1:] += (
            wi[:, None] * inside[:, j, :-1]
        )
        outside[:, j + 1, 1:] += (
            wo[:, None] * outside[:, j, :-1]
        )

    return inside, outside


def value_forward(parameters, data, layout):
    value = parameters[layout.value]
    shifted = value - value.max()
    weights = np.exp(shifted)

    inside, outside = value_prefixes(
        data.centers, weights, layout.k
    )

    ei = inside[:, -1, :]
    eo = outside[:, -1, :]

    volumes = ei * eo[:, ::-1]

    log_kernel = []
    shell_derivatives = []

    for item in layout.head:
        log_mass, derivative = layout.tree.evaluate(
            parameters[item["shell"]]
        )

        log_kernel.append(
            log_mass - np.log(layout.tree.multiplicity)
        )
        shell_derivatives.append(derivative)

    log_kernel = np.asarray(log_kernel)
    kernel = np.exp(log_kernel)

    normalization = kernel @ volumes.T

    require(
        np.all(normalization > 0.0)
        and np.all(np.isfinite(normalization)),
        "Neispravna normalizacija V stanja.",
    )

    shell_probability = (
        kernel[:, None, :] * volumes[None, :, :]
        / normalization[:, :, None]
    )

    return {
        "shifted": shifted,
        "weights": weights,
        "inside": inside,
        "outside": outside,
        "volumes": volumes,
        "log_kernel": log_kernel,
        "kernel": kernel,
        "normalization": normalization,
        "shell_probability": shell_probability,
        "derivative": shell_derivatives,
    }


def attention_forward(parameters, data, layout):
    embedding = parameters[layout.embedding]
    weights = np.exp(embedding - embedding.max())

    totals = data.centers @ weights
    require(np.all(totals > 0), "Neispravna amplitudna reprezentacija.")

    probabilities = (
        data.centers * weights[None, :] / totals[:, None]
    )

    inverse_norm = 1.0 / np.sqrt(
        totals[data.queries, None] * totals[data.memory]
    )

    log_attention = []
    caches = []

    lag = data.lags.astype(np.float64)[None, :]

    for item in layout.head:
        phase = (
            item["phase_base"] + parameters[item["phase"]]
        )
        phase_factor = np.exp(1j * phase)

        semantic = np.einsum(
            "twn,n->tw",
            data.intersections,
            weights * phase_factor,
            optimize=True,
        ) * inverse_norm

        mix = float(
            expit(math.log(3.0) + parameters[item["mix"]])
        )
        d_mix = mix * (1.0 - mix)

        frequency_fraction = expit(
            item["frequency_base"]
            + parameters[item["frequency"]]
        )
        frequency = math.pi * frequency_fraction
        d_frequency = (
            math.pi * frequency_fraction * (1.0 - frequency_fraction)
        )

        position = 0.5 * (
            np.cos(frequency[0] * lag)
            + np.cos(frequency[1] * lag)
        )

        raw_gain = (
            math.log(math.expm1(4.0))
            + parameters[item["gain"]]
        )
        gain = float(np.logaddexp(0.0, raw_gain))
        d_gain = float(expit(raw_gain))

        amplitude = mix * semantic + (1.0 - mix) * position
        fidelity = np.abs(amplitude) ** 2

        scores = gain * fidelity
        scores = np.where(data.valid, scores, -np.inf)
        log_alpha = scores - logsumexp(
            scores, axis=1, keepdims=True
        )

        log_attention.append(log_alpha)
        caches.append(
            {
                "phase": phase_factor,
                "semantic": semantic,
                "position": position,
                "amplitude": amplitude,
                "fidelity": fidelity,
                "mix": mix,
                "d_mix": d_mix,
                "frequency": frequency,
                "d_frequency": d_frequency,
                "gain": gain,
                "d_gain": d_gain,
            }
        )

    return (
        np.asarray(log_attention),
        caches,
        probabilities,
        inverse_norm,
        weights,
    )


def objective_parts(parameters, data, layout, gradient=True):
    require(data.targets is not None, "Nedostaju ciljevi za ucenje.")

    value = value_forward(parameters, data, layout)

    (
        log_attention,
        attention_cache,
        embedding_probability,
        inverse_norm,
        embedding_weights,
    ) = attention_forward(parameters, data, layout)

    head_logits = parameters[layout.head_mix]
    log_head = head_logits - logsumexp(head_logits)
    head_probability = np.exp(log_head)

    target_log_weight = data.targets @ value["shifted"]

    component = np.empty_like(log_attention)

    for h in range(layout.heads):
        component[h] = (
            log_head[h]
            + log_attention[h]
            + value["log_kernel"][h, data.overlap]
            - np.log(value["normalization"][h, data.memory])
            + target_log_weight[:, None]
        )

    log_probability = logsumexp(component, axis=(0, 2))

    require(
        np.all(np.isfinite(log_probability)),
        "Nekonacna verovatnoca ciljne sedmorke.",
    )

    losses = -log_probability

    if not gradient:
        return losses, None

    responsibility = np.exp(
        component - log_probability[None, :, None]
    )

    result = np.zeros(layout.size, dtype=np.float64)
    count = data.size

    center_responsibility = np.zeros(
        (layout.heads, len(data.centers)), dtype=np.float64
    )

    for h in range(layout.heads):
        center_responsibility[h] = np.bincount(
            data.memory.ravel(),
            weights=responsibility[h].ravel(),
            minlength=len(data.centers),
        ) / count

    result[layout.head_mix] = (
        head_probability
        - responsibility.sum(axis=(1, 2)) / count
    )

    # Gradijenti RY parametara V stanja.
    for h, item in enumerate(layout.head):
        expected = np.einsum(
            "c,cr->r",
            center_responsibility[h],
            value["shell_probability"][h],
        )

        observed = np.bincount(
            data.overlap.ravel(),
            weights=responsibility[h].ravel(),
            minlength=layout.k + 1,
        ) / count

        result[item["shell"]] = (
            expected - observed
        ) @ value["derivative"][h]

    # Gradijent amplitudnog filtera.
    value_gradient = -data.targets.mean(axis=0)

    ei = value["inside"][:, -1, :]
    eo = value["outside"][:, -1, :]

    for h in range(layout.heads):
        factors = (
            center_responsibility[h]
            / value["normalization"][h]
        )

        adj_inside = (
            factors[:, None]
            * value["kernel"][h][None, :]
            * eo[:, ::-1]
        )

        adj_outside = (
            factors[:, None]
            * value["kernel"][h, ::-1][None, :]
            * ei[:, ::-1]
        )

        for j in range(layout.n - 1, -1, -1):
            di = np.sum(
                adj_inside[:, 1:] * value["inside"][:, j, :-1],
                axis=1,
            )
            do = np.sum(
                adj_outside[:, 1:] * value["outside"][:, j, :-1],
                axis=1,
            )

            membership = data.centers[:, j]
            weight = value["weights"][j]

            value_gradient[j] += weight * np.sum(
                membership * di + (1.0 - membership) * do
            )

            wi = membership * weight
            wo = (1.0 - membership) * weight

            adj_inside[:, :-1] += wi[:, None] * adj_inside[:, 1:]
            adj_outside[:, :-1] += wo[:, None] * adj_outside[:, 1:]

    result[layout.value] = value_gradient

    # Gradijenti paznje.
    alpha = np.exp(log_attention)
    head_responsibility = responsibility.sum(axis=2)

    score_gradient = (
        head_responsibility[:, :, None] * alpha - responsibility
    ) / count

    embedding_gradient = np.zeros(layout.n)
    lag = data.lags.astype(np.float64)[None, :]

    for h, item in enumerate(layout.head):
        cache = attention_cache[h]
        g = score_gradient[h]

        amplitude = cache["amplitude"]
        semantic = cache["semantic"]
        mix = cache["mix"]
        gain = cache["gain"]

        complex_adjoint = (
            2.0 * gain * mix * g * np.conjugate(amplitude)
        )

        direct = (
            embedding_weights
            * cache["phase"]
            * np.einsum(
                "tw,twn->n",
                complex_adjoint * inverse_norm,
                data.intersections,
                optimize=True,
            )
        )

        normalization_adjoint = np.real(
            complex_adjoint * semantic
        )

        query_weights = normalization_adjoint.sum(axis=1)

        key_weights = np.bincount(
            data.memory.ravel(),
            weights=normalization_adjoint.ravel(),
            minlength=len(data.centers),
        )

        embedding_gradient += (
            np.real(direct)
            - 0.5 * (
                query_weights @ embedding_probability[data.queries]
                + key_weights @ embedding_probability
            )
        )

        result[item["phase"]] = -np.imag(direct)

        result[item["mix"]] = cache["d_mix"] * np.sum(
            2.0 * gain * g * np.real(
                np.conjugate(amplitude)
                * (semantic - cache["position"])
            )
        )

        frequency_gradient = np.zeros(2)

        for b in range(2):
            d_position = (
                -0.5 * lag * np.sin(cache["frequency"][b] * lag)
            )

            frequency_gradient[b] = (
                cache["d_frequency"][b]
                * np.sum(
                    2.0 * gain * (1.0 - mix)
                    * g * np.real(amplitude) * d_position
                )
            )

        result[item["frequency"]] = frequency_gradient

        result[item["gain"]] = (
            cache["d_gain"] * np.sum(g * cache["fidelity"])
        )

    result[layout.embedding] = embedding_gradient

    require(
        np.all(np.isfinite(result)),
        "Nekonacan analiticki gradijent.",
    )

    return losses, result


@dataclass
class Model:
    layout: Layout
    parameters: np.ndarray
    regularization: float


def fit_model(data, layout, regularization, warm=None):
    start = (
        layout.initial()
        if warm is None else warm.parameters.copy()
    )

    anchor = start.copy()
    anchor_losses, _ = objective_parts(
        anchor, data, layout, gradient=False
    )

    scale = np.full(layout.size, 3.0)

    for item in layout.head:
        scale[item["frequency"]] = 0.7
        scale[item["mix"]] = 2.0
        scale[item["gain"]] = 2.0
        scale[item["shell"]] = 1.0

    best_value = math.inf
    best_norm = math.inf
    best_parameters = start.copy()

    iteration = 0
    last_report = time.monotonic()

    def objective(coordinates):
        nonlocal best_value, best_norm, best_parameters

        parameters = coordinates * scale
        losses, derivative = objective_parts(
            parameters, data, layout, gradient=True
        )

        value = float(
            np.mean(losses - anchor_losses)
            + 0.5 * regularization
            * np.mean((parameters - anchor) * (parameters + anchor))
        )

        derivative += (
            regularization * parameters / layout.size
        )

        norm = float(np.max(np.abs(derivative)))

        if (
            value < best_value
            or (
                abs(value - best_value) < 1e-13
                and norm < best_norm
            )
        ):
            best_value = value
            best_norm = norm
            best_parameters = parameters.copy()

        return value, derivative * scale

    def callback(_):
        nonlocal iteration, last_report

        iteration += 1
        now = time.monotonic()

        if now - last_report >= 60.0:
            print(f"Ucenje: iteracija {iteration}.", flush=True)
            last_report = now

    coordinates = start / scale

    for method, limit in (
        ("BFGS", BFGS_ITERATIONS),
        ("CG", CG_ITERATIONS),
    ):
        optimized = minimize(
            objective,
            coordinates,
            jac=True,
            method=method,
            callback=callback,
            options={
                "maxiter": limit,
                "gtol": GRADIENT_TOLERANCE * 0.15,
                "disp": False,
            },
        )

        objective(optimized.x)

        if best_norm <= GRADIENT_TOLERANCE:
            return Model(
                layout,
                best_parameters.copy(),
                regularization,
            )

        coordinates = best_parameters / scale

        if DIAGNOSTICS:
            print(
                f"{method}: gradijent={best_norm:.6g}; "
                f"iteracije={optimized.nit}."
            )

    raise RuntimeError(
        f"Optimizacija nije konvergirala; gradijent={best_norm:.6g}."
    )


def evaluate_model(model, data):
    losses, _ = objective_parts(
        model.parameters, data, model.layout, gradient=False
    )
    return float(losses.mean())


# ------------------------------------------------------------
# TACAN NEXT DEKODER
# ------------------------------------------------------------


@njit(cache=False)
def popcount(value):
    count = 0

    while value != np.uint64(0):
        value &= value - np.uint64(1)
        count += 1

    return count


@njit(cache=False)
def forbidden_run(mask, maximum):
    intersection = mask

    for shift in range(1, maximum + 1):
        intersection &= mask >> np.uint64(shift)

    return intersection != np.uint64(0)


@njit(cache=False)
def kernel_score(mask, log_weight, centers, coefficients):
    total = 0.0

    for m in range(len(centers)):
        overlap = popcount(mask & centers[m])
        total += coefficients[m, overlap]

    if total <= 0.0:
        return -np.inf

    return log_weight + math.log(total)


@njit(cache=False)
def search_maximum(
    depth,
    start,
    mask,
    log_weight,
    order,
    ordered_weights,
    weight_prefix,
    centers,
    coefficients,
    range_maximum,
    suffix_inside,
    k,
    maximum_run,
    best_score,
    best_mask,
    leaves,
):
    n = len(order)
    remaining = k - depth

    if remaining == 0:
        leaves[0] += 1

        score = kernel_score(
            mask, log_weight, centers, coefficients
        )

        if (
            score > best_score[0]
            or (
                score == best_score[0]
                and mask < best_mask[0]
            )
        ):
            best_score[0] = score
            best_mask[0] = mask

        return

    if n - start < remaining:
        return

    # Granica se racuna u gornjim nivoima stabla.
    # Dublje se kandidati racunaju direktno.
    if depth <= 2:
        kernel_upper = 0.0
        available = n - start

        for m in range(len(centers)):
            current = popcount(mask & centers[m])
            inside = suffix_inside[m, start]
            outside = available - inside

            low = current + max(0, remaining - outside)
            high = current + min(remaining, inside)

            kernel_upper += range_maximum[m, low, high]

        if kernel_upper <= 0.0:
            return

        weight_upper = (
            log_weight
            + weight_prefix[start + remaining]
            - weight_prefix[start]
        )

        upper = weight_upper + math.log(kernel_upper)
        tolerance = 1e-11 * (1.0 + abs(upper))

        if upper < best_score[0] - tolerance:
            return

    for position in range(start, n - remaining + 1):
        bit = np.uint64(1) << np.uint64(order[position])
        candidate = mask | bit

        if forbidden_run(candidate, maximum_run):
            continue

        search_maximum(
            depth + 1,
            position + 1,
            candidate,
            log_weight + ordered_weights[position],
            order,
            ordered_weights,
            weight_prefix,
            centers,
            coefficients,
            range_maximum,
            suffix_inside,
            k,
            maximum_run,
            best_score,
            best_mask,
            leaves,
        )


@dataclass
class Prediction:
    log_weights: np.ndarray
    centers: np.ndarray
    coefficients: np.ndarray
    k: int

    def score(self, mask):
        log_weight = sum(
            self.log_weights[j]
            for j in range(len(self.log_weights))
            if (int(mask) >> j) & 1
        )

        return float(
            kernel_score(
                np.uint64(mask),
                float(log_weight),
                self.centers,
                self.coefficients,
            )
        )


def prepare_prediction(model, data):
    require(data.size == 1, "NEXT zahteva jedan trenutni kontekst.")

    layout = model.layout
    value = value_forward(model.parameters, data, layout)
    log_attention, _, _, _, _ = attention_forward(
        model.parameters, data, layout
    )

    head_logits = model.parameters[layout.head_mix]
    head_probability = np.exp(
        head_logits - logsumexp(head_logits)
    )

    valid = data.valid[0]
    indices = data.memory[0, valid]
    coefficients = np.zeros((len(indices), layout.k + 1))

    for h in range(layout.heads):
        weight = (
            head_probability[h]
            * np.exp(log_attention[h, 0, valid])
            / value["normalization"][h, indices]
        )

        coefficients += weight[:, None] * value["kernel"][h]

    normalization = np.sum(
        coefficients * value["volumes"][indices]
    )

    require(
        abs(normalization - 1.0) < 1e-9,
        "Zajednicka NEXT raspodela nije normalizovana.",
    )

    merged = {}

    for index, coefficient in zip(indices, coefficients):
        numbers = np.flatnonzero(data.centers[index])
        mask = sum(1 << int(j) for j in numbers)

        if mask not in merged:
            merged[mask] = np.zeros(layout.k + 1)

        merged[mask] += coefficient

    masks = sorted(merged)

    return Prediction(
        value["shifted"].copy(),
        np.asarray(masks, dtype=np.uint64),
        np.ascontiguousarray([merged[mask] for mask in masks]),
        layout.k,
    )


def decode(prediction, maximum_run=MAX_CONSECUTIVE):
    n = len(prediction.log_weights)
    k = prediction.k

    # Ovo je samo redosled pretrage, ne konacan izbor.
    order = np.lexsort(
        (np.arange(n), -prediction.log_weights)
    ).astype(np.int64)

    ordered_weights = prediction.log_weights[order]
    weight_prefix = np.concatenate(
        ([0.0], np.cumsum(ordered_weights))
    )

    center_count = len(prediction.centers)

    suffix_inside = np.zeros(
        (center_count, n + 1), dtype=np.int64
    )
    range_maximum = np.zeros(
        (center_count, k + 1, k + 1), dtype=np.float64
    )

    for m, center in enumerate(prediction.centers):
        center = int(center)

        for position in range(n - 1, -1, -1):
            suffix_inside[m, position] = (
                suffix_inside[m, position + 1]
                + ((center >> int(order[position])) & 1)
            )

        for low in range(k + 1):
            largest = 0.0

            for high in range(low, k + 1):
                largest = max(
                    largest, prediction.coefficients[m, high]
                )
                range_maximum[m, low, high] = largest

    best_score = np.array([-math.inf], dtype=np.float64)
    best_mask = np.array([0], dtype=np.uint64)
    leaves = np.array([0], dtype=np.int64)

    # Pocetne dozvoljene tacke sluze samo kao donje granice.
    candidates = [int(mask) for mask in prediction.centers]

    greedy_mask = 0
    selected = 0

    for number in order:
        candidate = greedy_mask | (1 << int(number))

        if not forbidden_run(np.uint64(candidate), maximum_run):
            greedy_mask = candidate
            selected += 1

            if selected == k:
                candidates.append(greedy_mask)
                break

    for mask in candidates:
        if (
            mask.bit_count() == k
            and not forbidden_run(np.uint64(mask), maximum_run)
        ):
            score = prediction.score(mask)

            if score > best_score[0]:
                best_score[0] = score
                best_mask[0] = np.uint64(mask)

    search_maximum(
        0,
        0,
        np.uint64(0),
        0.0,
        order,
        ordered_weights,
        weight_prefix,
        prediction.centers,
        prediction.coefficients,
        range_maximum,
        suffix_inside,
        k,
        maximum_run,
        best_score,
        best_mask,
        leaves,
    )

    mask = int(best_mask[0])

    require(
        math.isfinite(best_score[0])
        and mask.bit_count() == k
        and not forbidden_run(np.uint64(mask), maximum_run),
        "NEXT dekoder nije dobio ispravan maksimum.",
    )

    require(
        abs(prediction.score(mask) - best_score[0]) < 1e-9,
        "Skor NEXT dekodera nije saglasan sa modelom.",
    )

    numbers = tuple(
        j + 1 for j in range(n) if (mask >> j) & 1
    )

    return numbers, float(best_score[0]), int(leaves[0])


# ------------------------------------------------------------
# QISKIT I MATEMATICKE PROVERE
# ------------------------------------------------------------


def controlled_operation(circuit, target, bit, prefix, gate):
    controls = list(range(bit))

    for control in controls:
        if ((prefix >> control) & 1) == 0:
            circuit.x(control)

    if controls:
        circuit.append(gate.control(len(controls)), controls + [target])
    else:
        circuit.append(gate, [target])

    for control in controls:
        if ((prefix >> control) & 1) == 0:
            circuit.x(control)


def check_shell_qiskit(layout, parameters):
    for item in layout.head:
        delta = parameters[item["shell"]]
        log_mass, _ = layout.tree.evaluate(delta)

        circuit = QuantumCircuit(layout.tree.bits)

        for node, (bit, prefix) in enumerate(layout.tree.nodes):
            z = layout.tree.base[node] + delta[node]
            angle = math.pi / 2.0 + math.atan(z)

            controlled_operation(
                circuit, bit, bit, prefix, RYGate(angle)
            )

        for bit, prefix in layout.tree.fixed:
            controlled_operation(
                circuit, bit, bit, prefix, RYGate(math.pi)
            )

        state = Statevector.from_instruction(circuit).data
        probability = np.abs(state) ** 2

        require(
            np.max(
                np.abs(
                    probability[:layout.k + 1] - np.exp(log_mass)
                )
            ) < 1e-10
            and probability[layout.k + 1:].sum() < 1e-10,
            "Qiskit provera RY stabla nije prosla.",
        )


def check_attention_qiskit(parameters, data, layout):
    _, caches, probabilities, _, _ = attention_forward(
        parameters, data, layout
    )

    row = data.size - 1
    memory_position = min(1, len(data.lags) - 1)

    query_time = int(data.queries[row])
    key_time = int(data.memory[row, memory_position])

    for h, item in enumerate(layout.head):
        cache = caches[h]
        mix = cache["mix"]

        def state_at(index):
            vector = np.zeros(64, dtype=np.complex128)
            vector[:layout.n] = (
                math.sqrt(mix) * np.sqrt(probabilities[index])
            )

            for b in range(2):
                angle = cache["frequency"][b] * index
                amplitude = math.sqrt((1.0 - mix) / 2.0)

                vector[layout.n + 2 * b] = (
                    amplitude * math.cos(angle)
                )
                vector[layout.n + 2 * b + 1] = (
                    amplitude * math.sin(angle)
                )

            return vector

        query_circuit = QuantumCircuit(6)
        query_circuit.append(
            StatePreparation(state_at(query_time)), range(6)
        )

        diagonal = np.ones(64, dtype=np.complex128)
        diagonal[:layout.n] = np.conjugate(cache["phase"])

        query_circuit.append(DiagonalGate(diagonal), range(6))

        key_circuit = QuantumCircuit(6)
        key_circuit.append(
            StatePreparation(state_at(key_time)), range(6)
        )

        query_state = Statevector.from_instruction(query_circuit)
        key_state = Statevector.from_instruction(key_circuit)

        actual = abs(query_state.inner(key_state)) ** 2
        expected = cache["fidelity"][row, memory_position]

        require(
            abs(actual - expected) < 1e-9,
            "Qiskit provera Q/K paznje nije prosla.",
        )


def check_value_qiskit(parameters, data, layout):
    value = value_forward(parameters, data, layout)

    center_index = 0
    center = data.centers[center_index]
    combinations = list(
        itertools.combinations(range(layout.n), layout.k)
    )

    require(
        len(combinations) <= 64,
        "Kontrolni Qiskit primer je prevelik.",
    )

    # Mali prostor kombinacija kodiran je indeksima 0..63.
    # To je izometrijsko prekodiranje istih amplituda.
    for h in range(layout.heads):
        initial = np.zeros(64, dtype=np.complex128)
        expected = np.zeros(64, dtype=np.float64)

        for index, combo in enumerate(combinations):
            overlap = int(center[list(combo)].sum())
            kernel = value["kernel"][h, overlap]
            log_weight = value["shifted"][list(combo)].sum()

            initial[index] = math.sqrt(kernel)
            expected[index] = (
                math.exp(log_weight) * kernel
                / value["normalization"][h, center_index]
            )

        require(
            abs(np.vdot(initial, initial).real - 1.0) < 1e-10,
            "Pocetno V stanje nije normalizovano.",
        )

        circuit = QuantumCircuit(7)
        circuit.append(StatePreparation(initial), range(6))

        for index, combo in enumerate(combinations):
            log_weight = value["shifted"][list(combo)].sum()
            filter_amplitude = math.exp(0.5 * log_weight)
            angle = 2.0 * math.asin(filter_amplitude)

            for bit in range(6):
                if ((index >> bit) & 1) == 0:
                    circuit.x(bit)

            circuit.append(
                RYGate(angle).control(6), list(range(7))
            )

            for bit in range(6):
                if ((index >> bit) & 1) == 0:
                    circuit.x(bit)

        state = Statevector.from_instruction(circuit).data
        selected = state[64:]
        success = float(np.vdot(selected, selected).real)

        require(
            abs(
                success - value["normalization"][h, center_index]
            ) < 1e-9,
            "Qiskit normalizacija amplitudnog filtera nije prosla.",
        )

        actual = np.abs(selected) ** 2 / success

        require(
            np.max(np.abs(actual - expected)) < 1e-9,
            "Qiskit V raspodela nije saglasna sa modelom.",
        )


def internal_checks():
    n, k = 6, 2
    combinations = list(itertools.combinations(range(n), k))

    draws = np.asarray(
        [
            combinations[(7 * t + t // 3) % len(combinations)]
            for t in range(14)
        ],
        dtype=np.int64,
    )
    history = binary_draws(draws, n)

    layout = Layout(n, k, HEADS)
    parameters = layout.initial()
    parameters += 0.035 * np.cos(
        0.41 * np.arange(layout.size) + 0.2
    )

    data = Dataset(
        history, np.arange(len(history) - 1), window=4
    )

    _, analytic = objective_parts(
        parameters, data, layout, gradient=True
    )
    numerical = np.zeros_like(parameters)

    for index in range(len(parameters)):
        step = 2e-5
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += step
        minus[index] -= step

        fp, _ = objective_parts(
            plus, data, layout, gradient=False
        )
        fm, _ = objective_parts(
            minus, data, layout, gradient=False
        )

        numerical[index] = (
            fp.mean() - fm.mean()
        ) / (2.0 * step)

    require(
        np.max(np.abs(analytic - numerical)) < 3e-6,
        "Analiticki gradijent nije prosao proveru.",
    )

    check_shell_qiskit(layout, parameters)
    check_attention_qiskit(parameters, data, layout)
    check_value_qiskit(parameters, data, layout)

    model = Model(layout, parameters, 0.1)

    next_data = Dataset(
        history, [len(history) - 1], window=4, observed=False
    )
    prediction = prepare_prediction(model, next_data)

    total = 0.0
    brute_best = -math.inf

    for combo in combinations:
        mask = sum(1 << j for j in combo)
        score = prediction.score(mask)
        total += math.exp(score)
        brute_best = max(brute_best, score)

    require(
        abs(total - 1.0) < 1e-10,
        "Raspodela cele kombinacije nije normalizovana.",
    )

    _, decoded_score, _ = decode(prediction)

    require(
        abs(decoded_score - brute_best) < 1e-10,
        "Dekoder nije pronasao globalni maksimum.",
    )

    # Provera da buduci redovi ne uticu na raniji uzorak.
    queries = np.arange(5)
    original = Dataset(history, queries, window=4)

    altered_history = history.copy()
    altered_history[7:] = np.roll(
        altered_history[7:], 1, axis=1
    )
    altered = Dataset(altered_history, queries, window=4)

    first, _ = objective_parts(
        parameters, original, layout, gradient=False
    )
    second, _ = objective_parts(
        parameters, altered, layout, gradient=False
    )

    require(
        np.array_equal(first, second),
        "Otkriven pristup buducim podacima.",
    )

    # Provera ekvivarijantnosti na promenu oznaka brojeva.
    permutation = np.array([4, 1, 5, 0, 3, 2])
    permuted_parameters = parameters.copy()

    permuted_parameters[layout.embedding] = (
        parameters[layout.embedding][permutation]
    )
    permuted_parameters[layout.value] = (
        parameters[layout.value][permutation]
    )

    for item in layout.head:
        permuted_parameters[item["phase"]] = (
            parameters[item["phase"]][permutation]
        )

    permuted_data = Dataset(
        history[:, permutation],
        np.arange(len(history) - 1),
        window=4,
    )

    original_loss, _ = objective_parts(
        parameters, data, layout, gradient=False
    )
    permuted_loss, _ = objective_parts(
        permuted_parameters, permuted_data, layout, gradient=False
    )

    require(
        np.max(np.abs(original_loss - permuted_loss)) < 1e-10,
        "Model uvodi zavisnost od numerickog susedstva oznaka.",
    )

    # Posebna provera ogranicenja uzastopnog niza.
    test_n, test_k = 8, 5
    test_weights = -0.2 * np.arange(test_n, dtype=np.float64)
    center_sets = [
        tuple(range(5)),
        tuple(range(3, 8)),
    ]

    center_binary = binary_draws(
        np.asarray(center_sets, dtype=np.int64), test_n
    )

    inside, outside = value_prefixes(
        center_binary, np.exp(test_weights), test_k
    )
    volumes = inside[:, -1] * outside[:, -1, ::-1]

    test_coefficients = np.ones((2, test_k + 1))

    for row in range(2):
        test_coefficients[row] /= 2.0 * volumes[row].sum()

    controlled = Prediction(
        test_weights,
        np.asarray(
            [sum(1 << j for j in c) for c in center_sets],
            dtype=np.uint64,
        ),
        test_coefficients,
        test_k,
    )

    _, controlled_score, _ = decode(controlled, maximum_run=4)
    brute = -math.inf

    for combo in itertools.combinations(range(test_n), test_k):
        mask = sum(1 << j for j in combo)

        if not forbidden_run(np.uint64(mask), 4):
            brute = max(brute, controlled.score(mask))

    require(
        abs(controlled_score - brute) < 1e-10,
        "Ograniceni dekoder nije prosao proveru.",
    )


# ------------------------------------------------------------
# HRONOLOSKO UCENJE I TRI CSV-A
# ------------------------------------------------------------


def process_csv(path):
    draws = read_draws(path)
    history = binary_draws(draws, N)

    transitions = len(history) - 1
    train_end = int(TRAIN_FRACTION * transitions)
    validation_end = int(VALIDATION_END * transitions)

    require(
        0 < train_end < validation_end < transitions,
        "Neispravna hronoloska podela.",
    )

    layout = Layout(N, K, HEADS)

    training = Dataset(
        history[:train_end + 1],
        np.arange(train_end),
        WINDOW,
    )
    validation = Dataset(
        history[:validation_end + 1],
        np.arange(train_end, validation_end),
        WINDOW,
    )

    best_model = None
    best_validation = math.inf
    warm = None

    for regularization in REGULARIZATIONS:
        print(
            f"Ucenje paznje: reg={regularization:g}.",
            flush=True,
        )

        try:
            candidate = fit_model(
                training, layout, regularization, warm=warm
            )
            score = evaluate_model(candidate, validation)
            warm = candidate

        except (RuntimeError, ValueError, FloatingPointError) as exc:
            print(
                f"PROVERA: reg={regularization:g} odbacen: {exc}"
            )
            continue

        if score < best_validation:
            best_validation = score
            best_model = candidate

    if best_model is None:
        raise RuntimeError(
            "Nijedan validacioni kandidat nije prosao provere."
        )

    print(
        f"Validacija NLL: {best_validation:.6f} | "
        f"reg: {best_model.regularization:g}"
    )

    del training, validation

    print("Ucenje modela za test.", flush=True)

    test_training = Dataset(
        history[:validation_end + 1],
        np.arange(validation_end),
        WINDOW,
    )

    test_model = fit_model(
        test_training,
        layout,
        best_model.regularization,
        warm=best_model,
    )

    del test_training

    test_data = Dataset(
        history,
        np.arange(validation_end, transitions),
        WINDOW,
    )

    print(f"Test NLL: {evaluate_model(test_model, test_data):.6f}")

    del test_data

    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    full_training = Dataset(
        history, np.arange(transitions), WINDOW
    )

    final_model = fit_model(
        full_training,
        layout,
        best_model.regularization,
        warm=test_model,
    )

    del full_training

    next_data = Dataset(
        history,
        [len(history) - 1],
        WINDOW,
        observed=False,
    )

    prediction = prepare_prediction(final_model, next_data)

    print("Tacna pretraga NEXT maksimuma.", flush=True)
    numbers, score, evaluated = decode(prediction)

    require(
        len(numbers) == K
        and len(set(numbers)) == K
        and min(numbers) >= 1
        and max(numbers) <= N,
        "Neispravna NEXT kombinacija.",
    )
    require(score <= 1e-9, "NEXT verovatnoca je veca od jedan.")

    print(f"NEXT: {list(numbers)}")

    if DIAGNOSTICS:
        print(f"Potpuno ocenjeni kandidati: {evaluated}.")

    return {
        "name": path.name,
        "next": numbers,
        "prediction": prediction,
        "fingerprint": hashlib.sha256(
            np.ascontiguousarray(draws).tobytes()
        ).hexdigest(),
    }


def diagnostic_difference(first, second):
    largest = 0.0

    for step in (1, 2, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19):
        for start in range(N):
            numbers = (start + step * np.arange(K)) % N
            mask = sum(1 << int(j) for j in numbers)

            largest = max(
                largest,
                abs(first.score(mask) - second.score(mask)),
            )

    return largest


def main():
    print(
        "Quantum Set Attention V1: ucene amplitude, "
        "vremensko kodiranje i visеglava paznja."
    )
    print(
        "Q/K preklapanja i V amplitude: lokalna kvantna simulacija."
    )
    print(
        "Softmax, mesanje stanja i optimizacija: klasicni."
    )
    print(
        f"Kontekst: {WINDOW} kola | "
        "tacan NLL cele sedmorke."
    )
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()

    print("Interne provere: prosle.")

    results = []
    failures = 0

    for path in FILES:
        print()
        print(f"CSV: {path.name}", flush=True)

        try:
            current = process_csv(path)

            for previous in results:
                if current["next"] != previous["next"]:
                    continue

                same_input = (
                    current["fingerprint"] == previous["fingerprint"]
                )

                difference = diagnostic_difference(
                    current["prediction"],
                    previous["prediction"],
                )

                print(
                    f"PROVERA: isti NEXT za "
                    f"{previous['name']} i {current['name']}."
                )

                input_status = "isti" if same_input else "razliciti"

                print(
                    f"CSV sadrzaji: {input_status}; "
                    "maksimalna razlika logP na provernim kombinacijama: "
                    f"{difference:.6g}."
                )

            results.append(current)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}")

            if DIAGNOSTICS:
                import traceback
                traceback.print_exc()

    if failures:
        print()
        print(
            "Obrada nije kompletna: neki CSV-ovi nisu prosli provere."
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")

    except Exception as exc:
        print()
        print(f"STOP: {exc}")

        if DIAGNOSTICS:
            import traceback
            traceback.print_exc()



"""
Quantum Set Attention V1: ucene amplitude, vremensko kodiranje i viseglava paznja.
Q/K preklapanja i V amplitude: lokalna kvantna simulacija.
Softmax, mesanje stanja i optimizacija: klasicni.
Kontekst: 32 kola | tacan NLL cele sedmorke.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4724_k93.csv
Ucenje paznje: reg=1.
Ucenje paznje: reg=0.1.
Ucenje paznje: reg=0.01.
Ucenje: iteracija 482.
Ucenje: iteracija 808.
Validacija NLL: 16.546337 | reg: 1
Ucenje modela za test.
Test NLL: 16.543651
Zavrsno ucenje: svi istorijski prelazi.
Tacna pretraga NEXT maksimuma.
NEXT: [8, x, 11, y, 23, z, 34]

CSV: loto7_4724_k93_loto_2984.csv
Ucenje paznje: reg=1.
Ucenje paznje: reg=0.1.
Ucenje paznje: reg=0.01.
Ucenje: iteracija 784.
Validacija NLL: 16.560782 | reg: 1
Ucenje modela za test.
Test NLL: 16.552515
Zavrsno ucenje: svi istorijski prelazi.
Tacna pretraga NEXT maksimuma.
NEXT: [8, x, 21, y, 23, z, 38]

CSV: loto7_4724_k93_loto_plus_1740.csv
Ucenje paznje: reg=1.
Ucenje paznje: reg=0.1.
Ucenje paznje: reg=0.01.
Validacija NLL: 16.574794 | reg: 1
Ucenje modela za test.
Test NLL: 16.557370
Zavrsno ucenje: svi istorijski prelazi.
Tacna pretraga NEXT maksimuma.
NEXT: [2, x, 8, y, 26, z, 34]
"""



"""
Pažnja zavisi od trenutnog konteksta: model uči kojim prethodnim kolima treba dati veći značaj. 
To je osnovni mehanizam transformera. 

- Učeno predstavljanje brojeva i kola. 
    Sličnost se uči iz podataka; numeričko susedstvo brojeva ne mora značiti sličnost.
- Kodiranje vremenskog položaja. 
    Model razlikuje koliko je neko kolo udaljeno od trenutka predikcije. Redosled sedam brojeva unutar istog kola tretiram kao nebitan.
- Višeglava pažnja. 
    Različite glave mogu učiti različite odnose između istorijskih kola, bez unapred zadatih uloga.
- Vremensko maskiranje. 
    Pri učenju svakog prelaza dostupna je samo njegova prošlost.

Za kvantnu razradu istražio sam učene amplitudne reprezentacije i njihova preklapanja kao osnovu pažnje. 
Pažnja mora stvarno upravljati učenjem zajedničke raspodele sedmorke. 
Njene težine same nisu verovatnoće NEXT kombinacija.

Doprinos je prilagodljiv izbor relevantne prošlosti; poboljšanje treba potvrditi hronološkom validacijom.

Učena pažnji između istorijskih konteksta, sa vremenskim kodiranjem i amplitudnim izlazom za celu sedmorku.

V1 za ideju transformera: učene reprezentacije, vremensko kodiranje i višeglava pažnja za NEXT predikciju

Razrađena pažnja nad prethodnim kolima, vremensko kodiranje i normalizovan amplitudni model cele sedmorke. 
Ugrađene provere gradijenata, Qiskit amplituda i izbora maksimuma. 

Ovo je amplitudna adaptacija višeglave pažnje za skupove brojeva.
"""
