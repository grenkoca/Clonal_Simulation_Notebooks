"""
Beta Regression with Direct (α, β) Parameterization

Models CpG methylation using beta regression where both shape parameters
α and β are modeled directly via log-link functions:

    log(α_i) = X_alpha @ gamma_alpha
    log(β_i) = X_beta  @ gamma_beta

This is a reparameterization of the classical mean/precision beta regression:
    mu  = alpha / (alpha + beta)      (mean)
    phi = alpha + beta                 (precision)

The beta log-likelihood is the same:
    log p(y | α, β) = log Γ(α+β) - log Γ(α) - log Γ(β)
                      + (α-1) log y + (β-1) log(1-y)

Moment computations use scipy.special.digamma (psi) and polygamma for
entropy and log-space variance.
"""

import numpy as np
from scipy.special import loggamma, digamma, polygamma, betaln, gammaln
from scipy.optimize import minimize
from scipy.stats import beta as scipy_beta


class BetaRegressionAlphaBeta:
    """Beta regression using direct (α, β) parameterization.

    Parameters are stored as:
        gamma_alpha : ndarray, shape (n_alpha_features,)
        gamma_beta  : ndarray, shape (n_beta_features,)
        converged   : bool
        loglik      : float  (maximized log-likelihood)
    """

    def __init__(self):
        self.gamma_alpha = None
        self.gamma_beta = None
        self.converged = False
        self.loglik = None

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def _beta_loglik(self, params, y, X_alpha, X_beta):
        n_alpha = X_alpha.shape[1]
        g_alpha = params[:n_alpha]
        g_beta = params[n_alpha:]

        log_alpha = np.clip(X_alpha @ g_alpha, -10, 10)
        log_beta = np.clip(X_beta @ g_beta, -10, 10)
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)

        phi = alpha + beta

        # Beta log-likelihood per observation
        ll = (
            loggamma(phi)
            - loggamma(alpha)
            - loggamma(beta)
            + (alpha - 1) * np.log(y)
            + (beta - 1) * np.log(1 - y)
        )
        return -np.sum(ll)  # negative LL for minimization

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, y, X_alpha, X_beta, max_iter=200, ftol=1e-6):
        """Fit the model via L-BFGS-B.

        Parameters
        ----------
        y : array-like, shape (n,)
            Methylation proportions in (0, 1).
        X_alpha : array-like, shape (n, p_alpha)
            Design matrix for the α sub-model.
        X_beta : array-like, shape (n, p_beta)
            Design matrix for the β sub-model.
        max_iter : int
        ftol : float
        """
        y = np.clip(np.asarray(y, dtype=float), 1e-4, 1 - 1e-4)
        X_alpha = np.asarray(X_alpha, dtype=float)
        X_beta = np.asarray(X_beta, dtype=float)

        # Initialization: match moment estimates
        mu_hat = np.mean(y)
        var_hat = np.var(y)
        phi_hat = max(1.0, (mu_hat * (1 - mu_hat) / (var_hat + 1e-6)) - 1)
        alpha_hat = max(0.5, mu_hat * phi_hat)
        beta_hat = max(0.5, (1 - mu_hat) * phi_hat)

        g_alpha_init = np.zeros(X_alpha.shape[1])
        g_alpha_init[0] = np.log(alpha_hat)

        g_beta_init = np.zeros(X_beta.shape[1])
        g_beta_init[0] = np.log(beta_hat)

        init_params = np.concatenate([g_alpha_init, g_beta_init])

        try:
            result = minimize(
                self._beta_loglik,
                init_params,
                args=(y, X_alpha, X_beta),
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": ftol},
            )
            n_alpha = X_alpha.shape[1]
            self.gamma_alpha = result.x[:n_alpha]
            self.gamma_beta = result.x[n_alpha:]
            self.converged = result.success
            self.loglik = -result.fun
        except Exception as e :
            self.converged = False

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_alpha(self, X_alpha):
        """Return α = exp(X_alpha @ gamma_alpha)."""
        if self.gamma_alpha is None:
            raise ValueError("Model not fitted yet")
        return np.exp(X_alpha @ self.gamma_alpha)

    def predict_beta(self, X_beta):
        """Return β = exp(X_beta @ gamma_beta)."""
        if self.gamma_beta is None:
            raise ValueError("Model not fitted yet")
        return np.exp(X_beta @ self.gamma_beta)

    def predict_shapes(self, X_alpha, X_beta):
        """Return (alpha, beta) tuple of arrays."""
        return self.predict_alpha(X_alpha), self.predict_beta(X_beta)

    def predict_mean(self, X_alpha, X_beta):
        """Return E[Y] = α / (α + β)."""
        alpha, beta = self.predict_shapes(X_alpha, X_beta)
        return alpha / (alpha + beta)

    def predict_precision(self, X_alpha, X_beta):
        """Return φ = α + β."""
        alpha, beta = self.predict_shapes(X_alpha, X_beta)
        return alpha + beta

    def predict_variance(self, X_alpha, X_beta):
        """Return Var[Y] = α·β / ((α+β)²·(α+β+1))."""
        alpha, beta = self.predict_shapes(X_alpha, X_beta)
        phi = alpha + beta
        return (alpha * beta) / (phi**2 * (phi + 1))

    def predict_moments(self, X_alpha, X_beta):
        """Return a dict of per-observation moment arrays.

        Keys
        ----
        alpha, beta, phi : shape parameters and precision
        mean             : E[Y] = α/(α+β)
        median           : no closed form; NaN array returned
        mode             : (α-1)/(α+β-2) if α>1 and β>1, else NaN
        variance         : α·β / (φ²·(φ+1))
        std              : sqrt(variance)
        skewness         : 2(β-α)·sqrt(φ+1) / (sqrt(α·β)·(φ+2))
        excess_kurtosis  : 6·[(α-β)²·(φ+1) - α·β·(φ+2)] /
                             [α·β·(φ+2)·(φ+3)]
        geometric_mean   : exp(E[log Y]) = exp(ψ(α) - ψ(φ))
        harmonic_mean    : 1/E[1/Y] = (φ-1)/(α-1) if α>1, else NaN
        log_variance     : Var[log(Y)] = ψ₁(α) - ψ₁(φ)   (polygamma(1,·))
        log_variance_complement : Var[log(1-Y)] = ψ₁(β) - ψ₁(φ)
        log_covariance   : Cov[log(Y), log(1-Y)] = -ψ₁(φ)
        entropy          : log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (φ-2)ψ(φ)
        """
        alpha, beta = self.predict_shapes(X_alpha, X_beta)
        phi = alpha + beta
        n = len(alpha)

        mean = alpha / phi
        variance = (alpha * beta) / (phi**2 * (phi + 1))
        std = np.sqrt(variance)

        # Mode: (α-1)/(φ-2) when α>1 and β>1
        mode = np.where(
            (alpha > 1) & (beta > 1),
            (alpha - 1) / (phi - 2),
            np.nan,
        )

        # Skewness
        skewness = (
            2 * (beta - alpha) * np.sqrt(phi + 1) / (np.sqrt(alpha * beta) * (phi + 2))
        )

        # Excess kurtosis
        excess_kurtosis = (
            6
            * ((alpha - beta) ** 2 * (phi + 1) - alpha * beta * (phi + 2))
            / (alpha * beta * (phi + 2) * (phi + 3))
        )

        # Geometric mean: exp(ψ(α) - ψ(φ))
        geometric_mean = np.exp(digamma(alpha) - digamma(phi))

        # Harmonic mean: (φ-1)/(α-1) when α>1
        harmonic_mean = np.where(alpha > 1, (phi - 1) / (alpha - 1), np.nan)

        # Log-space variance: ψ₁(α) - ψ₁(φ)
        psi1_alpha = polygamma(1, alpha)
        psi1_beta = polygamma(1, beta)
        psi1_phi = polygamma(1, phi)
        log_variance = psi1_alpha - psi1_phi
        log_variance_complement = psi1_beta - psi1_phi
        log_covariance = -psi1_phi

        # Entropy: log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (φ-2)ψ(φ)
        log_beta_fn = betaln(alpha, beta)
        entropy = (
            log_beta_fn
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (phi - 2) * digamma(phi)
        )

        # Median has no closed form
        median = np.full(n, np.nan)

        return {
            "alpha": alpha,
            "beta": beta,
            "phi": phi,
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "geometric_mean": geometric_mean,
            "harmonic_mean": harmonic_mean,
            "log_variance": log_variance,
            "log_variance_complement": log_variance_complement,
            "log_covariance": log_covariance,
            "entropy": entropy,
        }


class BetaRegressionChangePoint:
    """Beta regression with a per-CpG piecewise linear (change-point) model.

    Jointly optimizes (gamma_alpha, gamma_beta, tau) in a single L-BFGS-B pass.
    Each CpG estimates regression coefficients and a structural break age tau.

    The design matrix for each sub-model is built dynamically on every
    log-likelihood evaluation:
        [intercept, t_linear, hinge=(t_linear-tau).clip(0), sex]

    Parameters are stored as:
        gamma_alpha : ndarray, shape (4,)  [intercept, t_linear, hinge, sex]
        gamma_beta  : ndarray, shape (4,)  [intercept, t_linear, hinge, sex]
        tau         : float
        converged   : bool
        loglik      : float
    """

    def __init__(self):
        self.gamma_alpha = None
        self.gamma_beta = None
        self.tau = None
        self.converged = False
        self.loglik = None

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def _beta_loglik_cp(self, params, y, t_linear, sex):
        """Negative log-likelihood for change-point beta regression."""
        g_alpha = params[:4]
        g_beta = params[4:8]
        tau = params[8]

        hinge = np.maximum(0.0, t_linear - tau)
        # Design: [intercept, t_linear, hinge, sex]
        X = np.column_stack([np.ones(len(t_linear)), t_linear, hinge, sex])

        log_alpha = np.clip(X @ g_alpha, -10, 10)
        log_beta = np.clip(X @ g_beta, -10, 10)
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)

        phi = alpha + beta

        ll = (
            loggamma(phi)
            - loggamma(alpha)
            - loggamma(beta)
            + (alpha - 1) * np.log(y)
            + (beta - 1) * np.log(1 - y)
        )
        return -np.sum(ll)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, y, t_linear, sex, tau_min=0.4, max_iter=200, ftol=1e-6):
        """Fit the change-point model via L-BFGS-B.

        Parameters
        ----------
        y : array-like, shape (n,)
            Methylation proportions in (0, 1).
        t_linear : array-like, shape (n,)
            Pre-computed normalized age vector (not a design matrix).
            The hinge column max(0, t_linear - tau) is built internally.
        sex : array-like, shape (n,)
            Binary sex covariate (1=female, 0=male, 0.5 for missing).
        tau_min : float
            Lower bound for tau. Default 0.4.
        max_iter : int
        ftol : float
        """
        y = np.clip(np.asarray(y, dtype=float), 1e-4, 1 - 1e-4)
        t_linear = np.asarray(t_linear, dtype=float)
        sex = np.asarray(sex, dtype=float)

        # Initialization: match moment estimates for intercepts
        mu_hat = np.mean(y)
        var_hat = np.var(y)
        phi_hat = max(1.0, (mu_hat * (1 - mu_hat) / (var_hat + 1e-6)) - 1)
        alpha_hat = max(0.5, mu_hat * phi_hat)
        beta_hat = max(0.5, (1 - mu_hat) * phi_hat)

        g_alpha_init = np.zeros(4)
        g_alpha_init[0] = np.log(alpha_hat)

        g_beta_init = np.zeros(4)
        g_beta_init[0] = np.log(beta_hat)

        # tau initialization: 0.75, or 85th percentile if < 10% samples have t_linear > 0.75
        frac_above = np.mean(t_linear > 0.75)
        if frac_above < 0.10:
            tau_init = np.percentile(t_linear, 85)
        else:
            tau_init = 0.75
        tau_init = float(np.clip(tau_init, tau_min + 0.01, 0.94))

        init_params = np.concatenate([g_alpha_init, g_beta_init, [tau_init]])

        # Bounds: (-inf, inf) for gamma coefficients, (tau_min, 0.95) for tau
        bounds = [(None, None)] * 8 + [(tau_min, 0.95)]

        try:
            result = minimize(
                self._beta_loglik_cp,
                init_params,
                args=(y, t_linear, sex),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": ftol},
            )
            self.gamma_alpha = result.x[:4]
            self.gamma_beta = result.x[4:8]
            self.tau = float(result.x[8])
            self.converged = result.success
            self.loglik = -result.fun
        except Exception:
            self.converged = False

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _build_design(self, t_linear, sex, tau=None):
        """Build design matrix [intercept, t_linear, hinge, sex]."""
        if tau is None:
            tau = self.tau
        hinge = np.maximum(0.0, t_linear - tau)
        return np.column_stack([np.ones(len(t_linear)), t_linear, hinge, sex])

    def predict_shapes(self, t_linear, sex, tau=None):
        """Return (alpha, beta) arrays.

        Parameters
        ----------
        t_linear : array-like
            Normalized age values.
        sex : array-like
            Sex covariate.
        tau : float, optional
            Change-point. Uses self.tau if None.
        """
        if self.gamma_alpha is None:
            raise ValueError("Model not fitted yet")
        t_linear = np.asarray(t_linear, dtype=float)
        sex = np.asarray(sex, dtype=float)
        X = self._build_design(t_linear, sex, tau)
        alpha = np.exp(np.clip(X @ self.gamma_alpha, -10, 10))
        beta = np.exp(np.clip(X @ self.gamma_beta, -10, 10))
        return alpha, beta

    def predict_mean(self, t_linear, sex, tau=None):
        """Return E[Y] = alpha / (alpha + beta)."""
        alpha, beta = self.predict_shapes(t_linear, sex, tau)
        return alpha / (alpha + beta)

    def predict_variance(self, t_linear, sex, tau=None):
        """Return Var[Y] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        alpha, beta = self.predict_shapes(t_linear, sex, tau)
        phi = alpha + beta
        return (alpha * beta) / (phi**2 * (phi + 1))

    def predict_precision(self, t_linear, sex, tau=None):
        """Return phi = alpha + beta."""
        alpha, beta = self.predict_shapes(t_linear, sex, tau)
        return alpha + beta

    def predict_moments(self, t_linear, sex):
        """Return a dict of moment arrays evaluated at three canonical design points.

        The three canonical design points are:
            [0]: t_min  = youngest observed t_linear
            [1]: tau    = estimated change-point
            [2]: t=1.0  = oldest possible age

        Each key in the returned dict maps to a length-3 array.

        Keys
        ----
        alpha, beta, phi, mean, median, mode, variance, std, skewness,
        excess_kurtosis, geometric_mean, harmonic_mean, log_variance,
        log_variance_complement, log_covariance, entropy
        """
        if self.gamma_alpha is None:
            raise ValueError("Model not fitted yet")
        t_linear = np.asarray(t_linear, dtype=float)

        t_min = float(t_linear.min())
        t_designs = np.array([t_min, self.tau, 1.0])
        # Use sex=0.5 (neutral) for canonical design points
        sex_designs = np.full(3, 0.5)

        alpha, beta = self.predict_shapes(t_designs, sex_designs)
        phi = alpha + beta

        mean = alpha / phi
        variance = (alpha * beta) / (phi**2 * (phi + 1))
        std = np.sqrt(variance)

        mode = np.where(
            (alpha > 1) & (beta > 1),
            (alpha - 1) / (phi - 2),
            np.nan,
        )

        skewness = (
            2 * (beta - alpha) * np.sqrt(phi + 1) / (np.sqrt(alpha * beta) * (phi + 2))
        )

        excess_kurtosis = (
            6
            * ((alpha - beta) ** 2 * (phi + 1) - alpha * beta * (phi + 2))
            / (alpha * beta * (phi + 2) * (phi + 3))
        )

        geometric_mean = np.exp(digamma(alpha) - digamma(phi))
        harmonic_mean = np.where(alpha > 1, (phi - 1) / (alpha - 1), np.nan)

        psi1_alpha = polygamma(1, alpha)
        psi1_beta = polygamma(1, beta)
        psi1_phi = polygamma(1, phi)
        log_variance = psi1_alpha - psi1_phi
        log_variance_complement = psi1_beta - psi1_phi
        log_covariance = -psi1_phi

        log_beta_fn = betaln(alpha, beta)
        entropy = (
            log_beta_fn
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (phi - 2) * digamma(phi)
        )

        # Use scipy.stats.beta.median for accurate median computation
        median = scipy_beta.median(alpha, beta)

        return {
            "alpha": alpha,
            "beta": beta,
            "phi": phi,
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "geometric_mean": geometric_mean,
            "harmonic_mean": harmonic_mean,
            "log_variance": log_variance,
            "log_variance_complement": log_variance_complement,
            "log_covariance": log_covariance,
            "entropy": entropy,
        }


def _squeeze(y, n):
    """Smithson & Verkuilen (2006) squeeze transform: y*(n-1)+0.5)/n."""
    return (y * (n - 1) + 0.5) / n


class BetaRegressionModeConcentration:
    """Beta regression using mode (ω) / concentration (c) parameterization.

    Parameters are stored as:
        gamma_omega : ndarray, shape (n_omega_features,)
        gamma_conc  : ndarray, shape (n_conc_features,)
        se_omega    : ndarray, shape (n_omega_features,)  (standard errors)
        se_conc     : ndarray, shape (n_conc_features,)   (standard errors)
        converged   : bool
        loglik      : float  (maximized log-likelihood)
        n_samples   : int
    """

    def __init__(self):
        self.gamma_omega = None
        self.gamma_conc = None
        self.se_omega = None
        self.se_conc = None
        self.converged = False
        self.loglik = None
        self.n_samples = -1

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    @staticmethod
    def _params_to_shapes(params, X_omega, X_conc):
        """Map parameter vector → (omega, c, alpha, beta, logit_omega, log_c).

        Returns all intermediates needed for both NLL and gradient.
        """
        n_omega = X_omega.shape[1]
        g_omega = params[:n_omega]
        g_conc = params[n_omega:]

        logit_omega = np.clip(X_omega @ g_omega, -20, 20)
        log_c = np.clip(X_conc @ g_conc, -20, 20)

        omega = BetaRegressionModeConcentration._sigmoid(logit_omega)
        c = np.exp(log_c)

        alpha = 1.0 + omega * c
        beta = 1.0 + (1.0 - omega) * c

        return omega, c, alpha, beta, logit_omega, log_c

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    def _nll(self, params, y, X_omega, X_conc):
        """Negative log-likelihood for minimization."""
        _, _, alpha, beta, _, _ = self._params_to_shapes(params, X_omega, X_conc)
        phi = alpha + beta

        ll = (
            loggamma(phi)
            - loggamma(alpha)
            - loggamma(beta)
            + (alpha - 1.0) * np.log(y)
            + (beta - 1.0) * np.log(1.0 - y)
        )
        return -np.sum(ll)

    # ------------------------------------------------------------------
    # Analytic gradient
    # ------------------------------------------------------------------

    def _nll_grad(self, params, y, X_omega, X_conc):
        """Analytic gradient of negative log-likelihood.

        Chain rule:
            d_NLL/d_gamma_omega = X_omega.T @ (d_NLL/d_logit_omega)
            d_NLL/d_gamma_conc  = X_conc.T  @ (d_NLL/d_log_c)

        where the per-observation derivatives chain through:
            (alpha, beta) → (omega, c) → (logit_omega, log_c)
        """
        omega, c, alpha, beta, _, _ = self._params_to_shapes(params, X_omega, X_conc)
        phi = alpha + beta

        # --- d_NLL / d_alpha, d_NLL / d_beta (standard beta deriv) ---
        psi_phi = digamma(phi)
        psi_alpha = digamma(alpha)
        psi_beta = digamma(beta)
        log_y = np.log(y)
        log_1my = np.log(1.0 - y)

        # These are d(+LL)/d(alpha) and d(+LL)/d(beta), so negate for NLL
        dll_dalpha = psi_phi - psi_alpha + log_y  # d(LL)/d(alpha)
        dll_dbeta = psi_phi - psi_beta + log_1my  # d(LL)/d(beta)

        # --- d(alpha, beta) / d(omega, c) ---
        # alpha = 1 + omega * c  =>  d_alpha/d_omega = c,  d_alpha/d_c = omega
        # beta  = 1 + (1-omega)*c => d_beta/d_omega = -c,  d_beta/d_c = 1-omega
        dll_domega = dll_dalpha * c + dll_dbeta * (-c)
        dll_dc = dll_dalpha * omega + dll_dbeta * (1.0 - omega)

        # --- d(omega, c) / d(logit_omega, log_c) ---
        # omega = sigmoid(logit_omega)  =>  d_omega/d_logit = omega*(1-omega)
        # c = exp(log_c)                =>  d_c/d_log_c = c
        dll_dlogit = dll_domega * omega * (1.0 - omega)
        dll_dlogc = dll_dc * c

        # --- Chain to parameter vectors (negate for NLL) ---
        grad_omega = -X_omega.T @ dll_dlogit
        grad_conc = -X_conc.T @ dll_dlogc

        return np.concatenate([grad_omega, grad_conc])

    # ------------------------------------------------------------------
    # Combined NLL + gradient (efficient: single forward pass)
    # ------------------------------------------------------------------

    def _nll_and_grad(self, params, y, X_omega, X_conc):
        """Return (NLL, gradient) in a single forward pass."""
        omega, c, alpha, beta, _, _ = self._params_to_shapes(params, X_omega, X_conc)
        phi = alpha + beta

        # --- NLL ---
        log_y = np.log(y)
        log_1my = np.log(1.0 - y)

        ll = (
            loggamma(phi)
            - loggamma(alpha)
            - loggamma(beta)
            + (alpha - 1.0) * log_y
            + (beta - 1.0) * log_1my
        )
        nll = -np.sum(ll)

        # --- Gradient ---
        psi_phi = digamma(phi)
        psi_alpha = digamma(alpha)
        psi_beta = digamma(beta)

        dll_dalpha = psi_phi - psi_alpha + log_y
        dll_dbeta = psi_phi - psi_beta + log_1my

        dll_domega = (dll_dalpha - dll_dbeta) * c
        dll_dc = dll_dalpha * omega + dll_dbeta * (1.0 - omega)

        dll_dlogit = dll_domega * omega * (1.0 - omega)
        dll_dlogc = dll_dc * c

        grad_omega = -X_omega.T @ dll_dlogit
        grad_conc = -X_conc.T @ dll_dlogc
        grad = np.concatenate([grad_omega, grad_conc])

        return nll, grad

    # ------------------------------------------------------------------
    # Standard errors via numerical Hessian
    # ------------------------------------------------------------------

    @staticmethod
    def _numerical_hessian(f, x, eps=1e-5):
        """Central-difference Hessian of scalar function f at x."""
        n = len(x)
        H = np.zeros((n, n))
        f0 = f(x)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
                H[j, i] = H[i, j]
        return H

    def _compute_standard_errors(self, params, y, X_omega, X_conc):
        """Compute SEs from the observed Fisher information (Hessian of NLL)."""

        def nll_only(p):
            return self._nll(p, y, X_omega, X_conc)

        try:
            H = self._numerical_hessian(nll_only, params)
            cov = np.linalg.inv(H)
            diag = np.diag(cov)
            # Negative diagonal entries indicate a saddle / non-positive-definite
            # Hessian — flag with NaN rather than imaginary SEs
            se = np.where(diag > 0, np.sqrt(diag), np.nan)
        except np.linalg.LinAlgError:
            se = np.full(len(params), np.nan)
            warnings.warn("Hessian singular — standard errors unavailable.")

        n_omega = X_omega.shape[1]
        self.se_omega = se[:n_omega]
        self.se_conc = se[n_omega:]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, y, X_omega, X_conc, max_iter=200, ftol=1e-6, compute_se=True):
        """Fit the model via L-BFGS-B with analytic gradients.

        Parameters
        ----------
        y : array-like, shape (n,)
            Methylation proportions in (0, 1).
        X_omega : array-like, shape (n, p_omega)
            Design matrix for the mode sub-model (logit link).
        X_conc : array-like, shape (n, p_conc)
            Design matrix for the concentration sub-model (log link).
        max_iter : int
        ftol : float
        compute_se : bool
            If True, compute standard errors from the numerical Hessian.
        """
        y = np.asarray(y, dtype=float)
        self.n_samples = y.shape[0]
        y = _squeeze(np.clip(y, 0.0, 1.0), self.n_samples)

        X_omega = np.asarray(X_omega, dtype=float)
        X_conc = np.asarray(X_conc, dtype=float)

        # Initialization: match moment estimates
        mu_hat = np.mean(y)
        var_hat = np.var(y)
        phi_hat = max(1.0, (mu_hat * (1 - mu_hat) / (var_hat + 1e-6)) - 1)
        alpha_hat = max(1.5, mu_hat * phi_hat)
        beta_hat = max(1.5, (1 - mu_hat) * phi_hat)

        c_hat = alpha_hat + beta_hat - 2.0
        omega_hat = np.clip((alpha_hat - 1.0) / c_hat, 0.01, 0.99)

        g_omega_init = np.zeros(X_omega.shape[1])
        g_omega_init[0] = np.log(omega_hat / (1.0 - omega_hat))

        g_conc_init = np.zeros(X_conc.shape[1])
        g_conc_init[0] = np.log(max(c_hat, 0.1))

        init_params = np.concatenate([g_omega_init, g_conc_init])

        try:
            result = minimize(
                self._nll_and_grad,
                init_params,
                args=(y, X_omega, X_conc),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iter, "ftol": ftol},
            )
            n_omega = X_omega.shape[1]
            self.gamma_omega = result.x[:n_omega]
            self.gamma_conc = result.x[n_omega:]
            self.converged = result.success
            self.loglik = -result.fun

            if compute_se:
                self._compute_standard_errors(result.x, y, X_omega, X_conc)

        except Exception:
            self.converged = False

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_omega(self, X_omega):
        """Return ω = sigmoid(X_omega @ gamma_omega)."""
        if self.gamma_omega is None:
            raise ValueError("Model not fitted yet")
        logit = np.clip(X_omega @ self.gamma_omega, -20, 20)
        return self._sigmoid(logit)

    def predict_concentration(self, X_conc):
        """Return c = exp(X_conc @ gamma_conc)."""
        if self.gamma_conc is None:
            raise ValueError("Model not fitted yet")
        log_c = np.clip(X_conc @ self.gamma_conc, -20, 20)
        return np.exp(log_c)

    def predict_shapes(self, X_omega, X_conc):
        """Return (alpha, beta) tuple of arrays."""
        omega = self.predict_omega(X_omega)
        c = self.predict_concentration(X_conc)
        alpha = 1.0 + omega * c
        beta = 1.0 + (1.0 - omega) * c
        return alpha, beta

    def predict_mean(self, X_omega, X_conc):
        """Return E[Y] = α / (α + β)."""
        alpha, beta = self.predict_shapes(X_omega, X_conc)
        return alpha / (alpha + beta)

    def predict_precision(self, X_omega, X_conc):
        """Return φ = α + β."""
        alpha, beta = self.predict_shapes(X_omega, X_conc)
        return alpha + beta

    def predict_variance(self, X_omega, X_conc):
        """Return Var[Y] = α·β / ((α+β)²·(α+β+1))."""
        alpha, beta = self.predict_shapes(X_omega, X_conc)
        phi = alpha + beta
        return (alpha * beta) / (phi**2 * (phi + 1))

    def predict_moments(self, X_omega, X_conc):
        """Return a dict of per-observation moment arrays.

        Keys
        ----
        alpha, beta, phi : shape parameters and precision
        omega            : mode = (α-1)/(α+β-2)  (always valid here)
        concentration    : c = α + β - 2
        mean             : E[Y] = α/(α+β)
        median           : scipy.stats.beta.median(α, β)
        variance         : α·β / (φ²·(φ+1))
        std              : sqrt(variance)
        skewness         : 2(β-α)·sqrt(φ+1) / (sqrt(α·β)·(φ+2))
        excess_kurtosis  : 6·[(α-β)²·(φ+1) - α·β·(φ+2)] /
                             [α·β·(φ+2)·(φ+3)]
        geometric_mean   : exp(ψ(α) - ψ(φ))
        harmonic_mean    : (φ-1)/(α-1)  (always valid since α > 1)
        log_variance     : Var[log(Y)] = ψ₁(α) - ψ₁(φ)
        log_variance_complement : Var[log(1-Y)] = ψ₁(β) - ψ₁(φ)
        log_covariance   : Cov[log(Y), log(1-Y)] = -ψ₁(φ)
        entropy          : log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (φ-2)ψ(φ)
        """
        alpha, beta = self.predict_shapes(X_omega, X_conc)
        phi = alpha + beta

        mean = alpha / phi
        variance = (alpha * beta) / (phi**2 * (phi + 1))
        std = np.sqrt(variance)

        # Mode is always valid for this parameterization (α > 1, β > 1)
        omega = (alpha - 1.0) / (phi - 2.0)
        concentration = phi - 2.0

        skewness = (
            2 * (beta - alpha) * np.sqrt(phi + 1) / (np.sqrt(alpha * beta) * (phi + 2))
        )

        excess_kurtosis = (
            6
            * ((alpha - beta) ** 2 * (phi + 1) - alpha * beta * (phi + 2))
            / (alpha * beta * (phi + 2) * (phi + 3))
        )

        geometric_mean = np.exp(digamma(alpha) - digamma(phi))

        # Harmonic mean always valid since α > 1 by construction
        harmonic_mean = (phi - 1) / (alpha - 1)

        psi1_alpha = polygamma(1, alpha)
        psi1_beta = polygamma(1, beta)
        psi1_phi = polygamma(1, phi)
        log_variance = psi1_alpha - psi1_phi
        log_variance_complement = psi1_beta - psi1_phi
        log_covariance = -psi1_phi

        log_beta_fn = betaln(alpha, beta)
        entropy = (
            log_beta_fn
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (phi - 2) * digamma(phi)
        )

        median = scipy_beta.median(alpha, beta)

        return {
            "alpha": alpha,
            "beta": beta,
            "phi": phi,
            "omega": omega,
            "concentration": concentration,
            "mean": mean,
            "median": median,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "geometric_mean": geometric_mean,
            "harmonic_mean": harmonic_mean,
            "log_variance": log_variance,
            "log_variance_complement": log_variance_complement,
            "log_covariance": log_covariance,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        """Print fitted coefficients and standard errors."""
        print("Model Converged:", self.converged)
        print(f"Log-Likelihood:  {self.loglik:.4f}" if self.loglik else "")
        print(f"N samples:       {self.n_samples}")
        print()
        print("Mode coefficients (logit space):")
        if self.gamma_omega is not None:
            for i, (g, se) in enumerate(
                zip(
                    self.gamma_omega,
                    self.se_omega
                    if self.se_omega is not None
                    else [np.nan] * len(self.gamma_omega),
                )
            ):
                print(f"  gamma_omega[{i}] = {g:+.6f}  (SE = {se:.6f})")
        print()
        print("Concentration coefficients (log space):")
        if self.gamma_conc is not None:
            for i, (g, se) in enumerate(
                zip(
                    self.gamma_conc,
                    self.se_conc
                    if self.se_conc is not None
                    else [np.nan] * len(self.gamma_conc),
                )
            ):
                print(f"  gamma_conc[{i}]  = {g:+.6f}  (SE = {se:.6f})")
