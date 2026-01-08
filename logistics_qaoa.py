"""
# QAOA for Logistics-Style Optimization (Exploratory)

This experiment explores how a simple logistics decision problem can be mapped
to a quantum optimization formulation.

## Problem
Given a small set of candidate shipment options, choose exactly one option
that minimizes a weighted objective (cost vs delay).

## Method
1. Normalize objectives
2. Formulate a QUBO with a one-hot constraint
3. Map QUBO → Ising Hamiltonian
4. Solve using QAOA (PennyLane)
5. Compare against classical baselines

## Notes
- Data access is illustrative; Snowflake is used as a placeholder for structured data.
- The focus is on **modeling and quantum formulation**, not infrastructure.
- QAOA depth and problem size are intentionally small for clarity.

## Purpose
This code is intended as a learning experiment to understand the full pipeline
from problem definition to quantum execution.
"""


from fastapi import FastAPI, HTTPException
import os
import pandas as pd
import snowflake.connector
import pennylane as qml
from pennylane import numpy as np

app = FastAPI(title="Logistics Quantum Optimizer (Demo)")

# -----------------------------
# Snowflake connection helper
# -----------------------------
# NOTE: Snowflake is used here as a stand-in for structured enterprise data.
# The focus of this experiment is the quantum optimization layer.

def get_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        role=os.environ["SNOWFLAKE_ROLE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
    )

# -----------------------------
# Normalization
# -----------------------------
def minmax(series: pd.Series) -> np.ndarray:
    x = series.astype(float).values
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# -----------------------------
# Build QUBO for choose-1
# Objective: sum w_i x_i
# Constraint: (sum x_i - 1)^2
# Q_ii = w_i - A
# Q_ij = 2A (i<j)
# -----------------------------
def build_qubo_choose_one(weights: np.ndarray, A: float) -> np.ndarray:
    n = len(weights)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = float(weights[i]) - A
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * A
    return Q

# -----------------------------
# QUBO -> Ising Hamiltonian for PennyLane
# x_i in {0,1}, map to z_i in {+1,-1} with x_i = (1 - z_i)/2
# Returns constant, h (linear Z), J (ZZ couplings)
# -----------------------------
def qubo_to_ising(Q: np.ndarray):
    n = Q.shape[0]
    h = np.zeros(n)
    J = {}

    constant = 0.0

    # diagonal terms
    for i in range(n):
        # Q_ii * x_i = Q_ii * (1 - z_i)/2
        constant += Q[i, i] / 2.0
        h[i] += -Q[i, i] / 2.0

    # off-diagonal terms (upper triangle)
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j]
            if abs(q) < 1e-12:
                continue
            # q * x_i x_j = q * (1 - z_i)/2 * (1 - z_j)/2
            # = q/4 * (1 - z_i - z_j + z_i z_j)
            constant += q / 4.0
            h[i] += -q / 4.0
            h[j] += -q / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + q / 4.0

    return constant, h, J

def build_hamiltonian(constant, h, J):
    coeffs = []
    ops = []

    # constant term (PennyLane supports Identity)
    coeffs.append(constant)
    ops.append(qml.Identity(0))

    for i, hi in enumerate(h):
        if abs(hi) < 1e-12:
            continue
        coeffs.append(hi)
        ops.append(qml.PauliZ(i))

    for (i, j), Jij in J.items():
        if abs(Jij) < 1e-12:
            continue
        coeffs.append(Jij)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)

# -----------------------------
# QAOA solver (demo-friendly)
# -----------------------------
def qaoa_solve(Q: np.ndarray, p: int = 1, steps: int = 60, lr: float = 0.2, seed: int = 7):
    n = Q.shape[0]
    np.random.seed(seed)

    const, h, J = qubo_to_ising(Q)
    H = build_hamiltonian(const, h, J)

    dev = qml.device("default.qubit", wires=n)

    def cost_layer(gamma):
        qml.qaoa.cost_layer(gamma, H)

    def mixer_layer(beta):
        for i in range(n):
            qml.RX(2.0 * beta, wires=i)

    @qml.qnode(dev)
    def circuit(params):
        # init
        for i in range(n):
            qml.Hadamard(wires=i)

        # p layers
        for layer in range(p):
            cost_layer(params[layer, 0])
            mixer_layer(params[layer, 1])

        return qml.expval(H)

    params = np.random.uniform(0, np.pi, (p, 2), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    for _ in range(steps):
        params = opt.step(circuit, params)

    # Sample bitstrings to get a feasible one-hot solution
    @qml.qnode(dev)
    def sample_probs(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        for layer in range(p):
            cost_layer(params[layer, 0])
            mixer_layer(params[layer, 1])
        return qml.probs(wires=range(n))

    probs = sample_probs(params)

    # enumerate best few bitstrings by prob (n is small: K<=8)
    candidates = np.argsort(probs)[::-1][: min(50, 2**n)]

    best_x = None
    best_energy = float("inf")

    for idx in candidates:
        bits = np.array(list(np.binary_repr(idx, width=n))).astype(int)
        # bits are 0/1; map to x directly
        if bits.sum() != 1:
            continue
        # energy for QUBO: x^T Q x
        x = bits.astype(float)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = bits

    return best_x, best_energy

# -----------------------------
# API endpoint
# -----------------------------
@app.get("/optimize/shipment/{shipment_id}")
def optimize_shipment(shipment_id: str, lambda_cost: float = 0.5, k: int = 6):
    if not (0.0 <= lambda_cost <= 1.0):
        raise HTTPException(status_code=400, detail="lambda_cost must be between 0 and 1")

    conn = get_conn()
    try:
        df = pd.read_sql(
            f"""
            select
              carrier,
              service_level_code,
              pred_cost,
              pred_delay,
              hist_count
            from LOGISTICS.SHIPMENTS_MARTS."shipment_option_scores"
            where shipment_id = %s
            order by hist_count desc
            limit %s
            """,
            conn,
            params=(shipment_id, k),
        )
    finally:
        conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="No candidate options found for this shipment_id (check lane/options)")

    # normalize objectives
    cost_n = minmax(df["PRED_COST"])
    delay_n = minmax(df["PRED_DELAY"])

    weights = lambda_cost * cost_n + (1.0 - lambda_cost) * delay_n

    # penalty: strong enough to enforce one-hot
    A = 10.0 * float(weights.max() + 1e-6)

    Q = build_qubo_choose_one(weights, A=A)

    # quantum solve (fallback to classical best if needed)
    try:
        x_bits, energy = qaoa_solve(Q, p=1, steps=60, lr=0.2)
        if x_bits is None:
            raise RuntimeError("No feasible one-hot solution found from samples")
        chosen_idx = int(np.argmax(x_bits))
        method = "qaoa"
    except Exception:
        chosen_idx = int(np.argmin(weights))
        energy = float(weights[chosen_idx])
        method = "fallback_argmin"

    row = df.iloc[chosen_idx]

    return {
        "shipment_id": shipment_id,
        "lambda_cost": lambda_cost,
        "k": int(len(df)),
        "method": method,
        "recommended": {
            "carrier": row["CARRIER"],
            "service_level_code": row["SERVICE_LEVEL_CODE"],
        },
        "scores": {
            "pred_cost": float(row["PRED_COST"]),
            "pred_delay": float(row["PRED_DELAY"]),
            "weighted_objective": float(weights[chosen_idx]),
            "qubo_energy": float(energy),
        },
        "candidates_preview": df.to_dict(orient="records"),
    }


# Run it -  uvicorn app:app --reload --port 8000