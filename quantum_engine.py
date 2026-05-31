import math
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.circuit.library import TwoLocal
import streamlit as st

# Maximum qubits we allow before refusing to run (statevector sim is 2^n)
MAX_SAFE_QUBITS = 18

def estimate_qubits(n_assets: int, max_weight: int) -> int:
    """Estimate how many binary qubits QUBO encoding will use."""
    bits_per_var = math.ceil(math.log2(max_weight + 1)) if max_weight > 0 else 1
    return n_assets * bits_per_var

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

@st.cache_data(show_spinner=False)
def build_weighted_qp(
    mean_returns,
    cov_matrix,
    risk_aversion=0.5,
    max_weight=2,
    transaction_cost_penalty=0.0,
    initial_allocation=None,
    sector_weights_penalty=0.0,
    sector_map=None,
    target_sectors=None
):
    n = len(mean_returns)
    qp = QuadraticProgram()

    # Define variables constraint limiting quantum memory load
    for i in range(n):
        qp.integer_var(name=f"x{i}", lowerbound=0, upperbound=max_weight)

    # Linear terms represent returns (we minimize negative returns = maximize returns)
    # We will adjust linear and quadratic coefficients based on constraints below.
    linear = {f"x{i}": -mean_returns.iloc[i] for i in range(n)}
    quadratic = {}

    # Quadratic terms represent variance/covariance risk
    for i in range(n):
        for j in range(n):
            quadratic[(f"x{i}", f"x{j}")] = risk_aversion * cov_matrix.iloc[i, j]

    # 1. Transaction Cost Penalty: lambda_tc * (x_i - x_init_i)^2
    # We expand: lambda_tc * (x_i^2 - 2 * x_init_i * x_i + x_init_i^2)
    # Ignoring the constant term, we add lambda_tc to x_i^2 (quadratic diagonal)
    # and subtract 2 * lambda_tc * x_init_i from x_i (linear term).
    if transaction_cost_penalty > 0:
        for i in range(n):
            ticker = mean_returns.index[i]
            # Get initial allocation fraction; fallback to uniform 1/n
            init_frac = 1.0 / n
            if initial_allocation is not None and ticker in initial_allocation:
                init_frac = initial_allocation[ticker]
            
            # Scale fractional allocation to integer target
            x_init_val = init_frac * max_weight
            
            # Modify quadratic term x_i^2 (diagonal)
            key = (f"x{i}", f"x{i}")
            quadratic[key] = quadratic.get(key, 0.0) + transaction_cost_penalty
            
            # Modify linear term
            linear[f"x{i}"] -= 2.0 * transaction_cost_penalty * x_init_val

    # 2. Sector Concentration Penalty: lambda_sec * (sum_{i in Sector} x_i)^2
    # We expand: lambda_sec * sum_{i, j in Sector} x_i * x_j
    if sector_weights_penalty > 0 and sector_map and target_sectors:
        for sec in target_sectors:
            sec_indices = [
                idx for idx, ticker in enumerate(mean_returns.index)
                if sector_map.get(ticker) == sec
            ]
            # Add penalty to all pairwise terms in the same sector
            for i in sec_indices:
                for j in sec_indices:
                    key = (f"x{i}", f"x{j}")
                    quadratic[key] = quadratic.get(key, 0.0) + sector_weights_penalty

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp

def solve_qaoa(qp, use_ibm=False, n_assets=4, max_weight=2):
    # --- Qubit complexity guard ---
    est_qubits = estimate_qubits(n_assets, max_weight)
    if est_qubits > MAX_SAFE_QUBITS:
        raise RuntimeError(
            f"This problem requires ~{est_qubits} qubits, which exceeds the safe "
            f"limit of {MAX_SAFE_QUBITS} for local simulation. "
            f"Please reduce the Share Multiplier or number of tickers."
        )

    # Scale QAOA effort down as problem grows to stay responsive
    if est_qubits <= 8:
        maxiter, reps = 30, 1
    elif est_qubits <= 12:
        maxiter, reps = 15, 1
    else:
        maxiter, reps = 8, 1

    if use_ibm and IBM_AVAILABLE:
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False)
        sampler = IBMSampler(backend=backend)
        st.info(f"Running QAOA on real IBM Quantum Hardware: {backend.name}", icon="🚀")
    else:
        sampler = StatevectorSampler()
        st.info(f"Running QAOA on Local Statevector Simulator (~{est_qubits} qubits)", icon="💻")

    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    
    opt_circ = None
    if hasattr(result.min_eigen_solver_result, 'optimal_circuit') and result.min_eigen_solver_result.optimal_circuit:
        opt_circ = result.min_eigen_solver_result.optimal_circuit
    elif hasattr(result.min_eigen_solver_result, 'optimal_point'):
        try:
            # Reconstruct circuit
            op, _ = qp.to_ising()
            ansatz = qaoa.ansatz
            if ansatz is None:
                from qiskit.circuit.library import QAOAAnsatz
                ansatz = QAOAAnsatz(op, reps=reps)
            opt_circ = ansatz.bind_parameters(result.min_eigen_solver_result.optimal_point)
        except Exception:
            pass
            
    return result, opt_circ

def solve_vqe(qp, use_ibm=False, n_assets=4, max_weight=2):
    est_qubits = estimate_qubits(n_assets, max_weight)
    if est_qubits > MAX_SAFE_QUBITS:
        raise RuntimeError(
            f"This problem requires ~{est_qubits} qubits, which exceeds the safe "
            f"limit of {MAX_SAFE_QUBITS} for local simulation."
        )

    if est_qubits <= 8:
        maxiter = 30
    else:
        maxiter = 15

    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=1)
    
    if use_ibm and IBM_AVAILABLE:
        from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False)
        estimator = IBMEstimator(backend=backend)
        st.info(f"Running VQE on real IBM Quantum Hardware: {backend.name}", icon="🚀")
    else:
        estimator = StatevectorEstimator()
        st.info(f"Running VQE on Local Statevector Simulator (~{est_qubits} qubits)", icon="💻")

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter))
    optimizer = MinimumEigenOptimizer(vqe)
    result = optimizer.solve(qp)
    
    opt_circ = None
    if hasattr(result.min_eigen_solver_result, 'optimal_circuit') and result.min_eigen_solver_result.optimal_circuit:
        opt_circ = result.min_eigen_solver_result.optimal_circuit
    elif hasattr(result.min_eigen_solver_result, 'optimal_point'):
        try:
            op, _ = qp.to_ising()
            a = vqe.ansatz
            a.num_qubits = op.num_qubits
            opt_circ = a.bind_parameters(result.min_eigen_solver_result.optimal_point)
        except Exception:
            pass
            
    return result, opt_circ

def interpret_weights(result, tickers):
    weights = {}
    total = sum(result.x)

    for i, val in enumerate(result.x):
        weights[tickers[i]] = (val / total) if total > 0 else 0

    return weights
