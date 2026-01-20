import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, List, Tuple
import seaborn as sns

class SobolEstimator:
    """
    Classe pour l'estimation des indices de Sobol selon la méthode Pick-and-Freeze
    Implémente l'estimateur T^u_{N,Cl} de l'article
    """
    
    def __init__(self, model: Callable, d: int):
        """
        Parameters:
        -----------
        model : fonction à analyser Y = f(X1, ..., Xd)
        d : nombre de variables d'entrée
        """
        self.model = model
        self.d = d
    
    def generate_samples(self, N: int, distribution='uniform'):
        """
        Génère les échantillons X et X' pour la méthode Pick-and-Freeze
        
        Parameters:
        -----------
        N : taille de l'échantillon
        distribution : 'uniform' ou 'gaussian'
        
        Returns:
        --------
        X, X_prime : matrices (N, d)
        """
        if distribution == 'uniform':
            X = np.random.uniform(-np.pi, np.pi, (N, self.d))
            X_prime = np.random.uniform(-np.pi, np.pi, (N, self.d))
        elif distribution == 'gaussian':
            X = np.random.randn(N, self.d)
            X_prime = np.random.randn(N, self.d)
        else:
            raise ValueError("Distribution doit être 'uniform' ou 'gaussian'")
        
        return X, X_prime
    
    def compute_Y_u(self, X: np.ndarray, X_prime: np.ndarray, u: List[int]) -> np.ndarray:
        """
        Calcule Y^u = f(X^u) où X^u_i = X_i si i in u, X'_i sinon
        
        Parameters:
        -----------
        X, X_prime : matrices (N, d)
        u : liste des indices à fixer
        
        Returns:
        --------
        Y_u : vecteur (N,)
        """
        N = X.shape[0]
        X_u = X_prime.copy()
        X_u[:, u] = X[:, u]
        
        Y_u = np.array([self.model(X_u[i]) for i in range(N)])
        return Y_u
    
    def estimate_T_Cl(self, N: int, u_list: List[List[int]], 
                      distribution='uniform') -> Tuple[np.ndarray, dict]:
        """
        Estime les indices de Sobol fermés avec l'estimateur T^u_{N,Cl}
        
        Parameters:
        -----------
        N : taille de l'échantillon
        u_list : liste des sous-ensembles u à estimer, ex: [[0], [1], [0,1]]
        distribution : type de distribution des entrées
        
        Returns:
        --------
        T_Cl : vecteur des indices estimés
        intermediates : dictionnaire avec Y, Y^u_j, Z^u, M^u
        """
        # Génération des échantillons
        X, X_prime = self.generate_samples(N, distribution)
        
        # Calcul de Y et des Y^{u_j}
        Y = np.array([self.model(X[i]) for i in range(N)])
        
        k = len(u_list)
        Y_u_list = []
        for u in u_list:
            Y_u = self.compute_Y_u(X, X_prime, u)
            Y_u_list.append(Y_u)
        
        # Calcul de Z^u et M^u
        Z_u = (Y + sum(Y_u_list)) / (k + 1)
        M_u = (Y**2 + sum([Y_u**2 for Y_u in Y_u_list])) / (k + 1)
        
        # Calcul du dénominateur commun
        denominator = np.mean(M_u) - np.mean(Z_u)**2
        
        # Calcul des indices
        T_Cl = np.zeros(k)
        for j in range(k):
            numerator = np.mean(Y * Y_u_list[j]) - (np.mean(Y + Y_u_list[j]) / 2)**2
            T_Cl[j] = numerator / denominator
        
        intermediates = {
            'Y': Y,
            'Y_u_list': Y_u_list,
            'Z_u': Z_u,
            'M_u': M_u,
            'denominator': denominator
        }
        
        return T_Cl, intermediates


# ==================== FONCTIONS TEST ====================

def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> float:
    """
    Fonction d'Ishigami: Y = sin(X1) + a*sin²(X2) + b*X3^4*sin(X1)
    
    Indices théoriques (avec a=7, b=0.1):
    S1 = 0.3139, S2 = 0.4424, S3 = 0
    """
    return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

def ishigami_indices_theory(a: float = 7.0, b: float = 0.1) -> dict:
    """Calcul analytique des indices de Sobol pour Ishigami"""
    V1 = 0.5 * (1 + b * np.pi**4 / 5)**2
    V2 = a**2 / 8
    V3 = 0
    V12 = 0  # Pas d'interaction pure entre X1 et X2
    V13 = b**2 * np.pi**8 / 18
    V23 = 0
    
    V_total = V1 + V2 + V3 + V12 + V13 + V23
    
    return {
        'S1': V1 / V_total,
        'S2': V2 / V_total,
        'S3': V3 / V_total,
        'S12': (V1 + V2 + V12) / V_total,
        'S13': (V1 + V3 + V13) / V_total,
        'V_total': V_total
    }

def sobol_g(x: np.ndarray, a: np.ndarray) -> float:
    """
    Fonction Sobol-G: Y = ∏ g_k(X_k) avec g_k(X_k) = |4X_k - 2| + a_k / (1 + a_k)
    X_k ~ Uniform[0, 1]
    
    Indice théorique: S_i = 1/(3(1+a_i)²) / ∏(1 + 1/(3(1+a_j)²))
    """
    d = len(x)
    result = 1.0
    for k in range(d):
        g_k = (np.abs(4*x[k] - 2) + a[k]) / (1 + a[k])
        result *= g_k
    return result

def sobol_g_indices_theory(a: np.ndarray) -> dict:
    """Calcul analytique des indices de Sobol pour la fonction Sobol-G"""
    d = len(a)
    V_i = 1 / (3 * (1 + a)**2)
    V_total = np.prod(1 + V_i) - 1
    S_i = V_i / V_total
    return {'S': S_i, 'V_total': V_total}

def linear_model(x: np.ndarray, coeffs: np.ndarray) -> float:
    """
    Modèle linéaire: Y = ∑ c_i * X_i
    Pour X_i ~ N(0,1): S_i = c_i² / ∑c_j²
    """
    return np.dot(coeffs, x)


# ==================== TESTS ET VISUALISATIONS ====================

def test_convergence_ishigami():
    """Test 1: Vérification de la convergence pour la fonction d'Ishigami"""
    print("=" * 60)
    print("TEST 1: Convergence pour la fonction d'Ishigami")
    print("=" * 60)
    
    estimator = SobolEstimator(ishigami, d=3)
    theory = ishigami_indices_theory()
    
    # Différentes tailles d'échantillon
    N_values = [100, 500, 1000, 5000, 10000]
    n_replications = 50
    
    results = {i: [] for i in range(3)}
    
    for N in N_values:
        print(f"\nN = {N}")
        estimates = []
        for rep in range(n_replications):
            T_Cl, _ = estimator.estimate_T_Cl(N, [[0], [1], [2]], 'uniform')
            estimates.append(T_Cl)
        
        estimates = np.array(estimates)
        for i in range(3):
            results[i].append(estimates[:, i])
            mean_est = np.mean(estimates[:, i])
            std_est = np.std(estimates[:, i])
            theory_val = theory[f'S{i+1}']
            bias = mean_est - theory_val
            print(f"  S{i+1}: Estimé = {mean_est:.4f} ± {std_est:.4f}, "
                  f"Théorique = {theory_val:.4f}, Biais = {bias:.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        ax = axes[i]
        positions = range(len(N_values))
        bp = ax.boxplot(results[i], positions=positions, widths=0.6)
        ax.axhline(theory[f'S{i+1}'], color='r', linestyle='--', 
                   label=f'Valeur théorique = {theory[f"S{i+1}"]:.4f}')
        ax.set_xticks(positions)
        ax.set_xticklabels(N_values)
        ax.set_xlabel('Taille échantillon N')
        ax.set_ylabel(f'S{i+1} estimé')
        ax.set_title(f'Convergence de S{i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_ishigami.png', dpi=150)
    print("\nGraphique sauvegardé: convergence_ishigami.png")

def test_asymptotic_normality():
    """Test 2: Vérification de la normalité asymptotique"""
    print("\n" + "=" * 60)
    print("TEST 2: Normalité asymptotique (√N convergence)")
    print("=" * 60)
    
    estimator = SobolEstimator(ishigami, d=3)
    theory = ishigami_indices_theory()
    
    N = 5000
    n_replications = 1000
    
    estimates = []
    for rep in range(n_replications):
        T_Cl, _ = estimator.estimate_T_Cl(N, [[0]], 'uniform')
        estimates.append(T_Cl[0])
    
    estimates = np.array(estimates)
    normalized = np.sqrt(N) * (estimates - theory['S1'])
    
    # Test de normalité
    _, p_value = stats.shapiro(normalized[:1000])  # Shapiro-Wilk
    
    print(f"\nPour N = {N}, {n_replications} réplications:")
    print(f"  Moyenne de √N(S1_N - S1): {np.mean(normalized):.4f}")
    print(f"  Écart-type: {np.std(normalized):.4f}")
    print(f"  Test de Shapiro-Wilk: p-value = {p_value:.4f}")
    if p_value > 0.05:
        print("  → On ne rejette pas l'hypothèse de normalité (p > 0.05)")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogramme + densité normale
    ax = axes[0]
    ax.hist(normalized, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_range = np.linspace(normalized.min(), normalized.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, np.mean(normalized), np.std(normalized)),
            'r-', linewidth=2, label='Densité normale ajustée')
    ax.set_xlabel('√N (S1_N - S1)')
    ax.set_ylabel('Densité')
    ax.set_title('Distribution de la statistique normalisée')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1]
    stats.probplot(normalized, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalite_asymptotique.png', dpi=150)
    print("\nGraphique sauvegardé: normalite_asymptotique.png")

def test_extreme_cases():
    """Test 3: Cas extrêmes (fonction triviale)"""
    print("\n" + "=" * 60)
    print("TEST 3: Cas extrêmes")
    print("=" * 60)
    
    # Cas 1: Fonction ne dépendant que de X1
    print("\nCas 1: Y = X1 (ne dépend que de X1)")
    def only_x1(x):
        return x[0]
    
    estimator = SobolEstimator(only_x1, d=3)
    T_Cl, _ = estimator.estimate_T_Cl(5000, [[0], [1], [2]], 'gaussian')
    print(f"  S1 estimé: {T_Cl[0]:.4f} (théorique ≈ 1.0)")
    print(f"  S2 estimé: {T_Cl[1]:.4f} (théorique ≈ 0.0)")
    print(f"  S3 estimé: {T_Cl[2]:.4f} (théorique ≈ 0.0)")
    
    # Cas 2: Fonction linéaire
    print("\nCas 2: Y = 2*X1 + X2 + 0.5*X3")
    coeffs = np.array([2.0, 1.0, 0.5])
    def linear(x):
        return linear_model(x, coeffs)
    
    estimator = SobolEstimator(linear, d=3)
    T_Cl, _ = estimator.estimate_T_Cl(5000, [[0], [1], [2]], 'gaussian')
    
    theory_var = coeffs**2
    theory_S = theory_var / np.sum(theory_var)
    
    for i in range(3):
        print(f"  S{i+1} estimé: {T_Cl[i]:.4f}, théorique: {theory_S[i]:.4f}")

def test_dimension_scaling():
    """Test 4: Montée en dimension"""
    print("\n" + "=" * 60)
    print("TEST 4: Montée en dimension (fonction Sobol-G)")
    print("=" * 60)
    
    dimensions = [2, 5, 10, 20]
    N = 5000
    n_replications = 20
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, d in enumerate(dimensions):
        print(f"\nDimension d = {d}")
        
        # Paramètres a_i décroissants pour avoir des variables d'importance variable
        a = np.array([i for i in range(d)])
        
        def sobol_g_d(x):
            return sobol_g(x, a)
        
        estimator = SobolEstimator(sobol_g_d, d=d)
        theory = sobol_g_indices_theory(a)
        
        # Estimer les 5 premiers indices
        n_to_estimate = min(5, d)
        u_list = [[i] for i in range(n_to_estimate)]
        
        estimates = []
        for rep in range(n_replications):
            # Pour Sobol-G, utiliser uniform [0,1]
            X = np.random.uniform(0, 1, (N, d))
            X_prime = np.random.uniform(0, 1, (N, d))
            
            estimator_temp = SobolEstimator(sobol_g_d, d=d)
            estimator_temp.generate_samples = lambda n, dist: (X, X_prime)
            
            T_Cl, _ = estimator_temp.estimate_T_Cl(N, u_list, 'uniform')
            estimates.append(T_Cl)
        
        estimates = np.array(estimates)
        
        # Visualisation
        ax = axes[idx]
        x_pos = np.arange(n_to_estimate)
        means = np.mean(estimates, axis=0)
        stds = np.std(estimates, axis=0)
        theory_vals = theory['S'][:n_to_estimate]
        
        ax.errorbar(x_pos - 0.1, means, yerr=stds, fmt='o', 
                    label='Estimé', capsize=5)
        ax.scatter(x_pos + 0.1, theory_vals, color='red', 
                   marker='x', s=100, label='Théorique')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'S{i+1}' for i in range(n_to_estimate)])
        ax.set_ylabel('Indice de Sobol')
        ax.set_title(f'Dimension d = {d}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i in range(n_to_estimate):
            print(f"  S{i+1}: Estimé = {means[i]:.4f} ± {stds[i]:.4f}, "
                  f"Théorique = {theory_vals[i]:.4f}")
    
    plt.tight_layout()
    plt.savefig('dimension_scaling.png', dpi=150)
    print("\nGraphique sauvegardé: dimension_scaling.png")


# ==================== PROGRAMME PRINCIPAL ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ÉTUDE DES ESTIMATEURS DE SOBOL - Estimateur T^u_{N,Cl}")
    print("=" * 60)
    
    # Exécution de tous les tests
    test_convergence_ishigami()
    test_asymptotic_normality()
    test_extreme_cases()
    test_dimension_scaling()
    
    print("\n" + "=" * 60)
    print("TOUS LES TESTS TERMINÉS")
    print("=" * 60)
    print("\nGraphiques générés:")
    print("  - convergence_ishigami.png")
    print("  - normalite_asymptotique.png")
    print("  - dimension_scaling.png")
