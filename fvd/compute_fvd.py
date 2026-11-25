"""Compute FVD (Frechet Video Distance) between two sets of pre-computed I3D features.

FVD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))

Takes .npz files produced by compute_features.py (containing 'features', 'mu', 'sigma').
Supports multiple .npz files per side (concatenated before computing stats) for mixed-bot FVD.
"""

import argparse

import numpy as np
import scipy.linalg


def load_features(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate features from one or more .npz files.

    Args:
        paths: List of .npz file paths.

    Returns:
        (features [N, D], mu [D], sigma [D, D]).
    """
    arrays = [np.load(p)["features"] for p in paths]
    features = np.concatenate(arrays, axis=0)  # [N_total, D]
    mu = np.mean(features, axis=0).astype(np.float64)
    sigma = np.cov(features, rowvar=False).astype(np.float64)
    return features, mu, sigma


def compute_fvd(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute Frechet Video Distance between two Gaussian-fitted feature sets.

    Args:
        mu1: [D] mean of distribution 1.
        sigma1: [D, D] covariance of distribution 1.
        mu2: [D] mean of distribution 2.
        sigma2: [D, D] covariance of distribution 2.

    Returns:
        FVD value (float). 0 = identical distributions.
    """
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    # sqrtm may introduce small imaginary components from numerical error
    if np.iscomplexobj(covmean):
        imag_max = np.max(np.abs(np.imag(covmean)))
        if imag_max > 1e-3:
            print(f"WARNING: sqrtm has large imaginary component ({imag_max:.6f})")
        covmean = np.real(covmean)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def main() -> None:
    """Load feature files and print the FVD between them."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--features1", type=str, nargs="+", required=True,
                        help="One or more .npz files for side 1 (concatenated if multiple)")
    parser.add_argument("--features2", type=str, nargs="+", required=True,
                        help="One or more .npz files for side 2 (concatenated if multiple)")
    args = parser.parse_args()

    feats1, mu1, sigma1 = load_features(paths=args.features1)
    feats2, mu2, sigma2 = load_features(paths=args.features2)

    n1, dim1 = feats1.shape
    n2, dim2 = feats2.shape
    assert dim1 == dim2, f"Feature dim mismatch: {dim1} vs {dim2}"

    print(f"Features 1: {args.features1} (n={n1}, d={dim1})")
    print(f"Features 2: {args.features2} (n={n2}, d={dim2})")

    fvd = compute_fvd(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
    print(f"\nFVD = {fvd:.6f}")


if __name__ == "__main__":
    main()
