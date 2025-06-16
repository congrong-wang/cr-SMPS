import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    """
    Gaussian function for curve fitting.
    Parameters
    ----------
    x : array-like
        The input values (particle diameters).
    A : float
        The amplitude of the Gaussian peak.
    mu : float
        The mean (center) of the Gaussian peak.
    sigma : float
        The standard deviation (width) of the Gaussian peak.
    Returns
    -------
    array-like
        The Gaussian function evaluated at x.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


file_dir = "20250606_PSL_cal/data/100nm.xlsx"


def norm_curve_fitting(file_dir):
    data = pd.read_excel(file_dir)
    Dp = data.iloc[:, 0]
    dNdlogDp = data.iloc[:, 1]
    samples = np.repeat(Dp, dNdlogDp)

    # Fit the Gaussian function to the data
    # `p0`: the initial guess for the parameters [A, mu, sigma]
    popt, _ = curve_fit(
        gaussian, Dp, dNdlogDp, p0=[max(dNdlogDp), np.mean(samples), np.std(samples)]
    )
    A_fit, mu_fit, sigma_fit = popt
    x_fit = np.linspace(min(Dp) - 5, max(Dp) + 5, 200)
    y_fit = gaussian(x_fit, *popt)

    plt.figure(figsize=(10, 6))
    plt.scatter(Dp, dNdlogDp, label="Measured points")
    plt.plot(
        x_fit,
        y_fit,
        "r-",
        label=f"Fitted Gaussian\nμ={mu_fit:.2f} nm, σ={sigma_fit:.2f} nm",
    )
    plt.xlabel("Particle Diameter (nm)")
    plt.ylabel("dN/dlogDp (#/cm³)")
    plt.title("Gaussian Fit to Size Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save fig with the same name as the file directory
    plt.savefig(file_dir.replace("nm.xlsx", "nm_fit.png"))
