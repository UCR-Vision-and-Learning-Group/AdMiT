"""
Kernel Mean Embedding (KME) implementation for AdMiT.
Includes KME approximation via weighted fixed synthetic points,
coefficient estimation for module matching, and MMD calculation.
"""

import numpy as np
import logging
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
# CVXOPT is an optional dependency, handle import error if not installed
try:
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False # Suppress solver output
    CVXOPT_AVAILABLE = True
except ImportError:
    logging.warning("CVXOPT not found. Coefficient estimation with constraints might fallback to unconstrained. Install with 'pip install cvxopt'.")
    CVXOPT_AVAILABLE = False

from . import config
from . import utils # Assuming utils contains logging setup and save/load functions

class ApproximateKME:
    """
    Represents an approximation of the Kernel Mean Embedding (KME)
    using a weighted combination of kernel functions centered at fixed
    synthetic data points 'z' with weights 'w'.

    The synthetic points 'z' are initialized (e.g., via K-Means) and then fixed.
    The weights 'w' are optimized to minimize the MMD between the empirical
    KME of the data and the KME represented by the weighted synthetic points.
    """
    def __init__(self, k=config.KME_SYNTHETIC_SIZE, gamma=config.KME_GAMMA,
                 init_method='kmeans', random_seed=config.SEED,
                 constraint_type=config.KME_CONSTRAINT_TYPE, # For weight constraints
                 reg=1e-8): # Regularization for QP solver
        """
        Initializes the ApproximateKME calculator.
        Args:
            k (int): Number of synthetic points (M in paper).
            gamma (float): Parameter for the RBF kernel exp(-gamma * ||x-y||^2).
            init_method (str): Method to initialize synthetic points 'z' ('kmeans', 'random').
            random_seed (int): Seed for initialization if random or KMeans.
            constraint_type (int): Constraint on weights w {0: None, 1: Non-negative, 2: Simplex}.
            reg (float): Small regularization added to QP matrix for stability.
        """
        self.z = None # Synthetic points (k, n_features) - Fixed after initialization
        self.weights = None # Optimized weights (k,)

        self.k = k
        self.gamma = gamma
        self.init_method = init_method
        self.random_seed = random_seed
        self.constraint_type = constraint_type
        self.reg = reg

        if constraint_type != 0 and not CVXOPT_AVAILABLE:
            logging.warning(f"CVXOPT is required for constraint_type {constraint_type}, but not installed. Falling back to constraint_type 0.")
            self.constraint_type = 0

        logging.debug(f"ApproximateKME initialized with k={k}, gamma={gamma}, init='{init_method}', constraint={constraint_type}")

    def fit(self, X):
        """
        Fits the KME approximation to the input data X.
        Initializes synthetic points 'z' and solves for the optimal weights 'w'.
        Args:
            X (numpy.ndarray): Input data (n_samples, n_features).
        """
        if X is None or X.shape[0] == 0:
            logging.error("Cannot fit ApproximateKME: Input data X is empty.")
            return

        n_samples, n_features = X.shape
        if self.k > n_samples:
            logging.warning(f"Synthetic set size k ({self.k}) is larger than number of samples ({n_samples}). Setting k = {n_samples}.")
            self.k = n_samples
        if self.k <= 0:
             logging.error(f"Synthetic set size k must be positive, got {self.k}.")
             return

        # 1. Initialize synthetic points 'z' and keep them fixed
        self._initialize_z(X)
        if self.z is None:
            logging.error("Failed to initialize synthetic points 'z'.")
            return

        # 2. Compute kernel matrices involving z and X
        K_zz = rbf_kernel(self.z, self.z, gamma=self.gamma)
        K_xz = rbf_kernel(X, self.z, gamma=self.gamma) # Shape (n_samples, k)

        # 3. Solve the QP for weights 'w'
        # Objective: min_w || \hat{\mu}(X) - \sum w_m k(z_m, \cdot) ||^2
        # Equivalent QP: min_w w^T K_zz w - 2 * (\frac{1}{N} K_{xz})^T w
        # Standard QP form: min (1/2) x'Px + q'x s.t. Gx <= h, Ax = b
        # Here: x=w, P = 2 * K_zz, q = -2 * (\frac{1}{N} \sum_n k(x_n, z_m))_{m=1}^k = -2/N * K_xz^T @ ones(N)

        P = 2 * (K_zz + self.reg * np.eye(self.k)) # Add regularization for stability
        # q term simplifies to -2 * mean(K_xz) column-wise
        q = -2 * np.mean(K_xz, axis=0) # Shape (k,)

        P_mat = matrix(P.astype(np.double))
        q_mat = matrix(q.astype(np.double))

        A, b, G, h = None, None, None, None
        if self.constraint_type == 1: # Non-negative: w >= 0 -> -w <= 0
            G = matrix(-np.eye(self.k).astype(np.double))
            h = matrix(np.zeros(self.k).astype(np.double))
        elif self.constraint_type == 2: # Simplex: w >= 0, sum(w) = 1
            G = matrix(-np.eye(self.k).astype(np.double))
            h = matrix(np.zeros(self.k).astype(np.double))
            A = matrix(np.ones((1, self.k)).astype(np.double))
            b = matrix(np.array([1.0]).astype(np.double))

        try:
            if self.constraint_type == 0: # Unconstrained QP
                # Solve Pw = -q => w = P^{-1} (-q)
                weights_sol = np.linalg.solve(P, -q)
                self.weights = weights_sol
                logging.debug(f"Computed unconstrained weights for ApproximateKME.")
            elif CVXOPT_AVAILABLE: # Constrained QP via CVXOPT
                sol = solvers.qp(P_mat, q_mat, G, h, A, b)
                if sol['status'] == 'optimal' or sol['status'] == 'optimal_inaccurate': # Accept slightly inaccurate solutions
                    self.weights = np.array(sol['x']).squeeze()
                    if sol['status'] == 'optimal_inaccurate':
                        logging.warning("CVXOPT found optimal_inaccurate solution for KME weights.")
                    # Clamp small negatives if non-negativity constraint was used
                    if self.constraint_type > 0:
                        self.weights[self.weights < 0] = 0
                    logging.debug(f"Computed constrained weights (type {self.constraint_type}) for ApproximateKME via CVXOPT.")
                else:
                    logging.error(f"CVXOPT solver failed for KME weights (status: {sol['status']}). Falling back to pseudo-inverse (unconstrained).")
                    weights_sol = np.linalg.pinv(P) @ (-q)
                    self.weights = weights_sol
            else: # Fallback if CVXOPT not available but constraints requested
                 logging.warning("CVXOPT unavailable for constraints, using pseudo-inverse (unconstrained) for KME weights.")
                 weights_sol = np.linalg.pinv(P) @ (-q)
                 self.weights = weights_sol

        except np.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error during KME weight calculation: {e}. Using pseudo-inverse.")
            weights_sol = np.linalg.pinv(P) @ (-q)
            self.weights = weights_sol
        except ValueError as e: # Handles CVXOPT rank issues etc.
             logging.error(f"ValueError during CVXOPT KME weight calculation: {e}. Using pseudo-inverse.")
             weights_sol = np.linalg.pinv(P) @ (-q)
             self.weights = weights_sol
        except Exception as e:
             logging.error(f"Unexpected error during KME weight calculation: {e}. Using pseudo-inverse.")
             weights_sol = np.linalg.pinv(P) @ (-q) # Fallback just in case
             self.weights = weights_sol


        # Optional: Normalize weights if simplex constraint wasn't used but desired post-hoc
        # if self.constraint_type != 2 and np.sum(self.weights) > 1e-6:
        #     self.weights /= np.sum(self.weights)

        logging.debug(f"Finished fitting ApproximateKME. z shape: {self.z.shape}, weights shape: {self.weights.shape if self.weights is not None else 'None'}")


    def _initialize_z(self, X):
        """Initializes fixed synthetic points z."""
        logging.debug(f"Initializing {self.k} synthetic points using '{self.init_method}' method.")
        n_samples, n_features = X.shape
        if self.init_method == 'kmeans':
            try:
                if self.k <= 0: raise ValueError(f"k must be > 0 for KMeans, got {self.k}")
                kmeans = KMeans(n_clusters=self.k, random_state=self.random_seed, n_init='auto')
                # Fit on a subset for efficiency if n_samples is very large?
                # kmeans.fit(X[np.random.choice(n_samples, size=min(n_samples, 5000), replace=False)])
                kmeans.fit(X)
                self.z = kmeans.cluster_centers_
                logging.debug(f"Initialized z using KMeans with {self.k} clusters.")
            except Exception as e:
                logging.error(f"KMeans initialization failed: {e}. Falling back to random initialization.")
                self.init_method = 'random' # Force fallback

        if self.init_method == 'random': # Also serves as fallback
             try:
                 rng = np.random.default_rng(self.random_seed)
                 indices = rng.choice(n_samples, size=self.k, replace=False)
                 self.z = X[indices, :]
                 logging.debug(f"Initialized z using random sampling of {self.k} points from data.")
             except Exception as e:
                 logging.error(f"Random initialization failed: {e}")
                 self.z = None

        if self.z is None:
             logging.error("Synthetic points 'z' could not be initialized.")

    def eval(self, X_eval):
        """
        Evaluates the approximated KME representation for new input data X_eval.
        Computes sum_m w_m * k(z_m, x) for each x in X_eval.
        Args:
            X_eval (numpy.ndarray): Input data (n_eval_samples, n_features).
        Returns:
            numpy.ndarray: Evaluated values (n_eval_samples,). Returns zeros on failure.
        """
        if self.z is None or self.weights is None:
            logging.error("Cannot evaluate: KME is not fitted (z or weights missing).")
            return np.zeros(X_eval.shape[0])

        K_eval_z = rbf_kernel(X_eval, self.z, gamma=self.gamma) # Shape (n_eval_samples, k)
        v = K_eval_z @ self.weights # Shape (n_eval_samples,)
        return v

    # Provide access to weights similar to previous 'beta'
    @property
    def beta(self):
        return self.weights

# --- KME Computation Helper ---

def compute_kme_representation(X):
    """
    Computes the ApproximateKME representation for data X.
    Args:
        X (numpy.ndarray): Input data (n_samples, n_features).
    Returns:
        ApproximateKME: The fitted KME representation object, or None if failed.
    """
    if X is None or X.shape[0] == 0:
        logging.error("Cannot compute KME representation: Input data X is empty.")
        return None
    logging.info(f"Computing KME representation for data shape: {X.shape}")

    # Use parameters from config
    kme_repr = ApproximateKME(
        k=config.KME_SYNTHETIC_SIZE,
        gamma=config.KME_GAMMA,
        random_seed=config.SEED,
        constraint_type=config.KME_CONSTRAINT_TYPE
        # init_method can be added to config if needed, defaults to kmeans
    )
    kme_repr.fit(X)

    # Check if fitting was successful
    if kme_repr.z is None or kme_repr.weights is None:
         logging.error("Failed to fit KME representation.")
         return None
    return kme_repr


# --- Coefficient Estimation (Matching) ---

def coefficient_estimation(target_kme_repr, source_kme_reprs):
    """
    Estimates mixture weights w_j for source KMEs to approximate the target KME.
    Solves min || target_kme - sum(w_j * source_kme_j) ||^2 w.r.t w_j,
    where the KMEs are approximated using the ApproximateKME class.

    Args:
        target_kme_repr (ApproximateKME): Fitted KME representation for target data.
        source_kme_reprs (list[ApproximateKME]): List of fitted KME representations for source domains.
    Returns:
        numpy.ndarray: Estimated coefficients w_j (n_sources,). Returns None on failure.
    """
    if target_kme_repr is None or not source_kme_reprs:
        logging.error("Invalid input for coefficient estimation: target or source KME is missing.")
        return None
    if target_kme_repr.z is None or target_kme_repr.weights is None:
        logging.error("Target KME representation is not fitted.")
        return None

    num_sources = len(source_kme_reprs)
    gamma = target_kme_repr.gamma # Assume all use the same gamma

    # Filter out invalid source KMEs
    valid_source_kme_reprs = []
    valid_indices = []
    for i, src_kme in enumerate(source_kme_reprs):
        if src_kme is None or src_kme.z is None or src_kme.weights is None:
            logging.warning(f"Source KME at index {i} is invalid or not fitted. Skipping.")
        elif src_kme.gamma != gamma:
            logging.warning(f"Source KME gamma {src_kme.gamma} differs from target gamma {gamma}. Skipping source {i}.")
        else:
            valid_source_kme_reprs.append(src_kme)
            valid_indices.append(i)

    if not valid_source_kme_reprs:
         logging.error("No valid source KME representations found.")
         return None

    num_valid_sources = len(valid_source_kme_reprs)

    # Calculate H matrix: H[i, j] = <source_i, source_j>_H = w_i^T K_{z_i z_j} w_j
    H = np.zeros((num_valid_sources, num_valid_sources))
    for i in range(num_valid_sources):
        for j in range(i, num_valid_sources):
            Zi, wi = valid_source_kme_reprs[i].z, valid_source_kme_reprs[i].weights
            Zj, wj = valid_source_kme_reprs[j].z, valid_source_kme_reprs[j].weights
            K_ij = rbf_kernel(Zi, Zj, gamma=gamma)
            H[i, j] = wi.T @ K_ij @ wj
            if i != j:
                H[j, i] = H[i, j] # Symmetric

    # Calculate C vector: C[i] = <source_i, target>_H = w_i^T K_{z_i z_target} w_target
    C = np.zeros(num_valid_sources)
    target_z, target_w = target_kme_repr.z, target_kme_repr.weights
    for i in range(num_valid_sources):
        Zi, wi = valid_source_kme_reprs[i].z, valid_source_kme_reprs[i].weights
        K_i_target = rbf_kernel(Zi, target_z, gamma=gamma)
        C[i] = wi.T @ K_i_target @ target_w

    # Add regularization to H for stability (increase slightly from KME reg)
    H += 1e-7 * np.eye(num_valid_sources)

    # Solve the QP: min_coeffs (1/2) coeffs' * (2H) * coeffs + (-2C)' * coeffs
    # Subject to constraints based on config.KME_EQ_CONSTRAINT, config.KME_NEQ_CONSTRAINT

    solver_type = config.KME_SOLVER
    use_constraints = config.KME_EQ_CONSTRAINT or config.KME_NEQ_CONSTRAINT

    if use_constraints and not CVXOPT_AVAILABLE:
        logging.warning("CVXOPT not available for constrained coefficient estimation. Using unconstrained 'inv' solver.")
        solver_type = 'inv'
    if solver_type == 'inv' and use_constraints:
        logging.warning("Solver 'inv' does not support constraints. Ignoring constraints.")
        use_constraints = False # Ensure constraints aren't attempted with 'inv'

    coeffs = None
    if solver_type == 'cvxopt' and CVXOPT_AVAILABLE:
        P = matrix(2 * H.astype(np.double))
        q = matrix(-2 * C.astype(np.double))
        A, b, G, h = None, None, None, None

        if config.KME_NEQ_CONSTRAINT: # w >= 0 -> -w <= 0
            G = matrix(-np.eye(num_valid_sources).astype(np.double))
            h = matrix(np.zeros(num_valid_sources).astype(np.double))
        if config.KME_EQ_CONSTRAINT: # sum(w) = 1
            A = matrix(np.ones((1, num_valid_sources)).astype(np.double))
            b = matrix(np.array([1.0]).astype(np.double))

        try:
            sol = solvers.qp(P, q, G, h, A, b)
            if sol['status'] == 'optimal' or sol['status'] == 'optimal_inaccurate':
                coeffs = np.array(sol['x']).squeeze()
                if sol['status'] == 'optimal_inaccurate':
                    logging.warning("CVXOPT found optimal_inaccurate solution for coefficients.")
                # Post-processing is important, especially if inaccurate
                if config.KME_NEQ_CONSTRAINT:
                    coeffs[coeffs < 0] = 0 # Clamp small negatives
                if config.KME_EQ_CONSTRAINT: # Ensure sum = 1 if simplex was target
                    coeff_sum = np.sum(coeffs)
                    if coeff_sum > 1e-6: coeffs /= coeff_sum
                    else: coeffs = np.ones_like(coeffs) / len(coeffs) # Fallback if sum is zero
                elif np.sum(coeffs) < 0: # Handle case where unconstrained leads to all negatives
                     coeffs[:] = 1.0 / len(coeffs) # Fallback to uniform

                logging.info(f"Estimated coefficients (cvxopt): {coeffs}")
            else:
                logging.error(f"CVXOPT solver failed for coefficients (status: {sol['status']}). Falling back to pseudo-inverse.")
                solver_type = 'inv' # Force fallback

        except ValueError as e: # Catches issues like Rank(A) < p or Rank([P; A; G]) < n
            logging.error(f"CVXOPT ValueError during coefficient estimation: {e}. Falling back to pseudo-inverse.")
            solver_type = 'inv'
        except Exception as e: # Catch other potential CVXOPT errors
            logging.error(f"Unexpected error during CVXOPT coefficient estimation: {e}. Falling back to pseudo-inverse.")
            solver_type = 'inv'

    # Fallback or direct use of pseudo-inverse (unconstrained or if cvxopt failed)
    if coeffs is None: # If solver_type was 'inv' or cvxopt failed
        try:
            coeffs = np.linalg.pinv(H) @ C
            # Apply constraints post-hoc if specified in config, even if not using CVXOPT
            if config.KME_NEQ_CONSTRAINT:
                coeffs[coeffs < 0] = 0
            if config.KME_EQ_CONSTRAINT:
                coeff_sum = np.sum(coeffs)
                if coeff_sum > 1e-6:
                    coeffs /= coeff_sum
                else: # Handle case where all coeffs are zero or negative
                    coeffs = np.ones_like(coeffs) / len(coeffs)
            elif np.sum(coeffs) < 0: # Handle case where unconstrained leads to all negatives
                 coeffs[:] = 1.0 / len(coeffs) # Fallback to uniform


            logging.info(f"Estimated coefficients (inv/fallback): {coeffs}")
        except np.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error during inverse coefficient estimation: {e}. Failed to estimate coefficients.")
            return None

    # Ensure coefficients are returned in the original source indexing order
    final_coeffs = np.zeros(num_sources)
    final_coeffs[valid_indices] = coeffs
    return final_coeffs


# --- MMD Calculation ---

def mmd_distance(kme1, kme2):
    """
    Calculates the Maximum Mean Discrepancy (MMD^2) between two KME approximations.
    Args:
        kme1 (ApproximateKME): First KME representation.
        kme2 (ApproximateKME): Second KME representation.
    Returns:
        float: The squared MMD value. Returns infinity on error.
    """
    if kme1 is None or kme2 is None or kme1.z is None or kme2.z is None or kme1.weights is None or kme2.weights is None:
        logging.error("Cannot compute MMD: One or both KME representations are invalid or not fitted.")
        return float('inf')
    if not np.isclose(kme1.gamma, kme2.gamma):
         logging.error(f"Cannot compute MMD: Gamma values differ ({kme1.gamma} vs {kme2.gamma}).")
         return float('inf')

    gamma = kme1.gamma
    Z1, w1 = kme1.z, kme1.weights
    Z2, w2 = kme2.z, kme2.weights

    K11 = rbf_kernel(Z1, Z1, gamma=gamma)
    K22 = rbf_kernel(Z2, Z2, gamma=gamma)
    K12 = rbf_kernel(Z1, Z2, gamma=gamma)

    term1 = w1.T @ K11 @ w1
    term2 = w2.T @ K22 @ w2
    term3 = w1.T @ K12 @ w2

    mmd2 = term1 + term2 - 2 * term3
    # MMD^2 should be non-negative, clamp small negative values due to precision
    mmd2_non_negative = max(0.0, mmd2)
    if mmd2 < -1e-6: # Log if clamping was significant
        logging.warning(f"MMD^2 calculation resulted in negative value {mmd2:.4g}, clamped to 0.")

    return mmd2_non_negative


# --- Example Usage (Self-Test) ---
if __name__ == '__main__':
    print("--- Testing KME Module (ApproximateKME) ---")
    utils.setup_logging() # Ensure logging is active
    # Create dummy data for two "domains"
    np.random.seed(config.SEED)
    X1 = np.random.rand(200, 10) * 0.5 # 200 samples, 10 features
    X2 = np.random.rand(250, 10) * 0.5 + 0.5 # 250 samples, 10 features (shifted)
    # Target is mix of first 50 samples of each domain
    X_target = np.vstack((X1[:50], X2[:50]))
    np.random.shuffle(X_target)

    # Configure KME settings for test (override config if needed)
    config.KME_SYNTHETIC_SIZE = 20 # Example: Use 20 synthetic points
    config.KME_GAMMA = 1.0 # Example: Use broader kernel gamma=1.0
    config.KME_CONSTRAINT_TYPE = 2 # Example: Simplex constraints on KME weights 'w'
    config.KME_EQ_CONSTRAINT = True # Example: Simplex constraints on coefficients
    config.KME_NEQ_CONSTRAINT = True

    print("Computing KME representations...")
    kme1 = compute_kme_representation(X1)
    kme2 = compute_kme_representation(X2)
    kme_target = compute_kme_representation(X_target)

    if kme1 and kme2 and kme_target:
        print("KME representations computed.")
        source_kmes = [kme1, kme2]

        print("\nEstimating coefficients...")
        coeffs = coefficient_estimation(kme_target, source_kmes)

        if coeffs is not None:
            print(f"Estimated coefficients: {coeffs}")
            # Expected: coeffs should be roughly [0.5, 0.5] given how X_target was created

            print("\nCalculating MMD distances...")
            mmd_1_target = mmd_distance(kme1, kme_target)
            mmd_2_target = mmd_distance(kme2, kme_target)
            mmd_1_2 = mmd_distance(kme1, kme2)
            print(f"MMD(Source1, Target): {mmd_1_target:.4f}")
            print(f"MMD(Source2, Target): {mmd_2_target:.4f}")
            print(f"MMD(Source1, Source2): {mmd_1_2:.4f}")
            # Expected: MMD(Source1, Target) and MMD(Source2, Target) should be smaller than MMD(Source1, Source2)

            # Check KME evaluation
            print("\nEvaluating KME on sample data...")
            eval_points = np.vstack([X1[0:2], X2[0:2], X_target[0:2]])
            kme1_eval = kme1.eval(eval_points)
            kme2_eval = kme2.eval(eval_points)
            kme_target_eval = kme_target.eval(eval_points)
            print(f"KME1 eval: {kme1_eval.round(3)}")
            print(f"KME2 eval: {kme2_eval.round(3)}")
            print(f"KMETarget eval: {kme_target_eval.round(3)}")

            logging.info("KME module test completed successfully.")
        else:
            logging.error("Coefficient estimation failed.")
    else:
        logging.error("Failed to compute KME representations.")