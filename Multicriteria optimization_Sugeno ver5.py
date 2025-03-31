import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Membership Functions
# -------------------------------
def gaussian_membership(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def gaussian_membership_derivative(x, c, sigma):
    mu = gaussian_membership(x, c, sigma)
    return mu * (-(x - c) / (sigma ** 2))

# -------------------------------
# Define the Sugeno rule class
# -------------------------------
class SugenoRule:
    """
    Each rule is defined by following:
      - a0: Bias term in the linear consequent.
      - a_coeff: Coefficients for each of the 3 decision variables.
      - centers: Centers for the Gaussian membership functions (for each variable).
      - sigmas: Spreads (sigma) for the Gaussian membership functions.
    """
    def __init__(self, a0, a_coeff, centers, sigmas):
        self.a0 = a0
        self.a_coeff = np.array(a_coeff)   
        self.centers = np.array(centers)    
        self.sigmas = np.array(sigmas)    

    def firing_strength(self, x):
        """The rule's firing strength."""
        mu = gaussian_membership(x, self.centers, self.sigmas)
        return np.prod(mu)

    def firing_strength_derivative(self, x):
        """The derivative of the rule's firing strength with respect to each x_i."""
        mu = gaussian_membership(x, self.centers, self.sigmas)
        dmu = gaussian_membership_derivative(x, self.centers, self.sigmas)
        deriv = np.zeros_like(x)
        for i in range(3):
            prod = 1.0
            for j in range(3):
                if j == i:
                    prod *= dmu[j]
                else:
                    prod *= mu[j]
            deriv[i] = prod
        return deriv

    def consequent(self, x):
        return self.a0 + np.dot(self.a_coeff, x)

    def consequent_derivative(self, x):
        return self.a_coeff

# -------------------------------------------------
# 3. Define the Sugeno class for 1 criterion
# --------------------------------------------------
class SugenoFIS:
    def __init__(self, rules):
        self.rules = rules  

    def output(self, x):
        numerator = 0.0
        denominator = 0.0
        for rule in self.rules:
            w = rule.firing_strength(x)
            y = rule.consequent(x)
            numerator += w * y
            denominator += w
        return numerator / denominator if denominator != 0 else 0.0

    def output_derivative(self, x):
        numerator = 0.0
        denominator = 0.0
        dNumerator = np.zeros_like(x)
        dDenominator = np.zeros_like(x)
        for rule in self.rules:
            w = rule.firing_strength(x)
            y = rule.consequent(x)
            dw = rule.firing_strength_derivative(x)
            dy = rule.consequent_derivative(x)
            numerator += w * y
            denominator += w
            dNumerator += dw * y + w * dy
            dDenominator += dw
        if denominator == 0:
            return np.zeros_like(x)
        # Rule: (D*dN - N*dD) / D^2
        grad = (denominator * dNumerator - numerator * dDenominator) / (denominator ** 2)
        return grad

# --------------------------------------------------------------------
# 4. Define 3 Fuzzy systems for 3 criteria with 3 rules
# ---------------------------------------------------------------------
rules_c1 = [
    SugenoRule(a0=1.0, a_coeff=[0.4, 0.1, 0.3], centers=[2.0, 3.0, 2.5], sigmas=[1.0, 1.0, 1.0]),
    SugenoRule(a0=1.5, a_coeff=[-0.2, 0.3, 0.2], centers=[3.5, 2.5, 3.0], sigmas=[1.2, 1.0, 1.1]),
    SugenoRule(a0=0.8, a_coeff=[0.3, -0.1, 0.4], centers=[2.5, 3.5, 2.0], sigmas=[0.9, 1.0, 1.0])
]
fis1 = SugenoFIS(rules_c1)

rules_c2 = [
    SugenoRule(a0=0.5, a_coeff=[0.2, 0.4, -0.1], centers=[3.0, 2.0, 3.5], sigmas=[1.0, 1.2, 1.0]),
    SugenoRule(a0=1.0, a_coeff=[-0.3, 0.1, 0.5], centers=[2.0, 3.0, 2.5], sigmas=[1.1, 1.0, 1.0]),
    SugenoRule(a0=0.8, a_coeff=[0.3, 0.2, 0.2], centers=[3.5, 3.0, 3.0], sigmas=[1.0, 1.0, 1.0])
]
fis2 = SugenoFIS(rules_c2)

rules_c3 = [
    SugenoRule(a0=1.2, a_coeff=[0.1, -0.2, 0.3], centers=[2.5, 2.5, 3.0], sigmas=[1.0, 1.1, 1.0]),
    SugenoRule(a0=0.9, a_coeff=[0.4, 0.1, -0.3], centers=[3.0, 3.5, 2.5], sigmas=[1.0, 1.0, 1.2]),
    SugenoRule(a0=1.0, a_coeff=[-0.1, 0.3, 0.2], centers=[2.0, 3.0, 3.5], sigmas=[1.1, 1.0, 1.0])
]
fis3 = SugenoFIS(rules_c3)

# -------------------------------
# 5. Composite an objective function and its gradient
# -------------------------------
lambda1 = 0.4  # Weight for criterion 1
lambda2 = 0.3  # Weight for criterion 2
lambda3 = 0.3  # Weight for criterion 3

def composite_objective(x):
    y1 = fis1.output(x)
    y2 = fis2.output(x)
    y3 = fis3.output(x)
    return lambda1 * y1 - lambda2 * y2 + lambda3 * y3

def composite_objective_gradient(x):
    grad1 = fis1.output_derivative(x)
    grad2 = fis2.output_derivative(x)
    grad3 = fis3.output_derivative(x)
    return lambda1 * grad1 - lambda2 * grad2 + lambda3 * grad3

# -------------------------------
# 6. Gradient Descent Optimizer
# -------------------------------
def gradient_descent(x_init, obj_func, grad_func, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = x_init.copy()
    history = [x.copy()]
    for i in range(max_iter):
        grad = grad_func(x)
        x_new = x - learning_rate * grad
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergence reached after {i+1} iterations.")
            break
        x = x_new
    return x, history

# -------------------------------
# 7. Run the optimization and visualize results
# -------------------------------
x_init = np.array([3.0, 3.0, 3.0])
x_opt, history = gradient_descent(x_init, composite_objective, composite_objective_gradient,
                                  learning_rate=0.05, max_iter=5000, tol=1e-6)

print("Optimal decision vector:", x_opt)
print("Composite objective value at optimum:", composite_objective(x_opt))

# Plotting the optimization path in 3D
history = np.array(history)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(history[:, 0], history[:, 1], history[:, 2], marker='o', label='Optimization Path')
ax.scatter(x_opt[0], x_opt[1], x_opt[2], color='r', s=100, label='Optimal Solution')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Optimization using Gradient Descent')
ax.legend()
plt.show()
