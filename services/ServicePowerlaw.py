from scipy.optimize import curve_fit


def fitPowerlaw(x, a):
    return x**a/2

def determinePowerlaw(probs):
    popt, pcov = curve_fit(fitPowerlaw, range(1, len(probs) + 1), probs)
    alpha = popt[0]
    print(f"Scaling exponent (α): {alpha}")
    return alpha
