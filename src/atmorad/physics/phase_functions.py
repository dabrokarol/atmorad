import numpy as np

from atmorad.constants import BOUNDARY_EPSILON


class Scattering:
    def __init__(self, pdf_array):
        """
        Takes a raw probability density array of cos_theta, normalizes it,
        and computes the cumulative distribution function (cdf) for fast sampling.

        Args:
        - pdf_array: probability density of cos_theta
        """
        self.n_precomputed = len(pdf_array)
        self.cos_grid = np.linspace(-1, 1, self.n_precomputed)
        dx = self.cos_grid[1] - self.cos_grid[0]
        pdf_array = pdf_array / (np.sum(pdf_array) * dx)
        self.distribuant = np.cumsum(pdf_array) * dx

    def __call__(self, rand_1, rand_2):
        """Computes sin and cos of theta, phi used for scattering. Uses `np.interp` to obtain reversed cdf values for given rand_1. Samples phi from uniform distribution [0,2pi].

        Args:
            rand_1 - uniform(0,1) samples used to compute cos_theta through inverse cdf
            rand_2 - uniform(0,1) samples used to compute sin_theta

        Returns:
            np.array((cos_theta, sin_theta, cos_phi, sin_phi))
        """
        phi = 2 * np.pi * rand_2

        cos_theta = np.interp(rand_1, self.distribuant, self.cos_grid)
        sin_theta = np.sqrt(1 - np.clip(cos_theta**2, 0.0, 1.0))

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return np.array((cos_theta, sin_theta, cos_phi, sin_phi))


# Source:
# L. G. Henyey, J. L. Greenstein, Diffuse radiation in the galaxy, [doi:10.1086/144246](https://doi.org/10.1086/144246)
class HenyeyGreensteinScattering(Scattering):
    def __init__(self, g: float):
        self.g = g

    def __call__(self, rand_1, rand_2):
        if abs(self.g) < BOUNDARY_EPSILON:
            cos_theta = 2.0 * rand_1 - 1.0
        else:
            sq = (1.0 - self.g**2) / (1.0 - self.g + 2.0 * self.g * rand_1)
            cos_theta = (1.0 + self.g**2 - sq**2) / (2.0 * self.g)

        sin_theta = np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0))
        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))


class IsotropicScattering(Scattering):
    def __init__(self):
        pass

    def __call__(self, rand_1, rand_2):
        cos_theta = 2.0 * rand_1 - 1.0
        sin_theta = np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0))

        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))


# Source:
# J. R. Frisvad, Importance sampling the Rayleigh phase function, [doi:10.1364/JOSAA.28.002436](https://doi.org/10.1364/JOSAA.28.002436)
class RayleighScattering(Scattering):
    def __init__(self):
        pass

    def __call__(self, rand_1, rand_2):
        u = 2.0 * rand_1 - 1.0
        w = np.cbrt(2.0 * u + np.sqrt(4.0 * u**2 + 1.0))
        cos_theta = w - 1.0 / w

        sin_theta = np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0))
        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))


SCATTERING_MODELS = {
    "hg": HenyeyGreensteinScattering,
    "isotropic": IsotropicScattering,
    "rayleigh": RayleighScattering,
}
