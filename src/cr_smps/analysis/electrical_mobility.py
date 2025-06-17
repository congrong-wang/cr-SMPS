import numpy as np
from scipy.optimize import fsolve

# The exact values for e & mu are not important,
# since they are cancelled out on both sides
e = 1.602e-19  # elementary charge in coulombs
mu = 1.8e-5  # dynamic viscosity of air in Pa.s

# Constants for Cunningham correction factor
A = 1.257
B = 0.4
C = 1.1
lambd = 68  # mean free path of air in nm


# Cunningham correction factor
def Cc(Dp):
    return 1 + (2 * lambd / Dp) * (A + B * np.exp(-C * Dp / lambd))


# Mobility diameter Zp
# n: number of charges
def Zp(Dp, n):
    return n * e / (3 * np.pi * mu * Dp) * Cc(Dp)


def calc_Dp(Dp1: float | int, n1: int, n2: int) -> float | int:
    """
    Known:
    Dp1: mobility diameter of particle 1 in nm
    n1: number of charges on particle 1
    n2: number of charges on particle 2
    Returns:
    Dp2: mobility diameter of particle 2 in nm
    """

    # Zp1 = Zp2
    def equation(Dp2):
        return Zp(Dp1, n1) - Zp(Dp2, n2)

    Dp_initial_guess = 100  # Initial guess for Dp in nm
    (Dp_solution,) = fsolve(equation, Dp_initial_guess)
    return Dp_solution
