"""
Stage 0: Binary Classification on the Complexity-Entropy (C-H) Plane

No ML model required - just 2D visualization with linear decision boundary.

Features:
- H: Permutation Entropy (pe)
- C: Statistical Complexity (comp)

Classification:
- Class 0: Baseline (no pain)
- Class 1: Pain (low + high)
- Rest segments: EXCLUDED

Normalization: Global z-score (valid for LOSO)

Author: Claude (AI4Pain Paper 2)
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class CHPlaneAnalyzer:
    """
    Binary pain classification on the Complexity-Entropy plane.

    Uses only 2 features (H, C) with a simple linear decision boundary.
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.data_dir = base_dir / 'data' / 'features'
        self.output_dir = base_dir / 'results' / 'stage0_binary'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimal parameters
        self.signal = 'eda'
        self.dimension = 7
        self.tau = 2

    def load_data(self) -> pd.DataFrame:
        """Load EDA feature data from all splits."""
        print("Loading EDA feature data...")
        dfs = []

        for split in ['train', 'validation']:
            filepath = self.data_dir / f'results_{split}_{self.signal}.csv'
            if filepath.exists():
                df = pd.read_csv(filepath)
                dfs.append(df)
                print(f"  Loaded {filepath.name}: {len(df)} rows")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(combined)}")
        return combined

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Extract H, C features and create binary labels.

        Excludes rest segments (baseline-only methodology).
        """
        # Filter for optimal d, tau
        mask = (df['dimension'] == self.dimension) & (df['tau'] == self.tau)
        subset = df[mask].copy()

        # CRITICAL: Exclude rest segments
        n_before = len(subset)
        subset = subset[subset['state'] != 'rest'].copy()
        n_after = len(subset)
        print(f"Excluded {n_before - n_after} rest segments")

        # Extract features
        H = subset['pe'].values  # Permutation Entropy
        C = subset['comp'].values  # Statistical Complexity

        # Extract subject ID
        subjects = subset['file_name'].apply(
            lambda x: int(x.split('/')[-1].replace('.csv', ''))
        ).values

        # Binary labels: baseline=0, pain (low+high)=1
        y = np.where(subset['state'] == 'baseline', 0, 1)

        # Remove NaN values
        valid = ~(np.isnan(H) | np.isnan(C))
        H, C, y, subjects = H[valid], C[valid], y[valid], subjects[valid]

        print(f"\nSamples after filtering: {len(y)}")
        print(f"  Baseline (0): {np.sum(y == 0)}")
        print(f"  Pain (1): {np.sum(y == 1)}")
        print(f"  Unique subjects: {len(np.unique(subjects))}")

        return H, C, y, subjects

    def apply_global_normalization(self, H: np.ndarray, C: np.ndarray) -> tuple:
        """
        Apply global z-score normalization.

        This is valid for LOSO validation (no data leakage).
        """
        print("\nApplying global z-score normalization...")

        H_mean, H_std = np.mean(H), np.std(H)
        C_mean, C_std = np.mean(C), np.std(C)

        H_norm = (H - H_mean) / (H_std + 1e-8)
        C_norm = (C - C_mean) / (C_std + 1e-8)

        print(f"  H: mean={H_mean:.4f}, std={H_std:.4f}")
        print(f"  C: mean={C_mean:.4f}, std={C_std:.4f}")

        return H_norm, C_norm

    def evaluate_classification(self, H: np.ndarray, C: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate linear classification on C-H plane."""
        X = np.column_stack([H, C])

        # Silhouette score (cluster quality)
        silhouette = silhouette_score(X, y, metric='euclidean')

        # Linear classifier (simple logistic regression)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)

        print(f"\nC-H Plane Results:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Linear Accuracy: {accuracy*100:.2f}%")

        return {
            'silhouette': silhouette,
            'accuracy': accuracy,
            'classifier': clf
        }

    def create_ch_plane_plot(
        self, H: np.ndarray, C: np.ndarray, y: np.ndarray,
        clf, silhouette: float, accuracy: float
    ) -> str:
        """Create the Complexity-Entropy plane visualization."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Colors
        colors = {0: '#2ecc71', 1: '#e74c3c'}
        labels = {
            0: f'Baseline (n={np.sum(y==0)})',
            1: f'Pain (n={np.sum(y==1)})'
        }

        # Plot points
        for class_val in [0, 1]:
            mask = y == class_val
            ax.scatter(
                H[mask], C[mask],
                c=colors[class_val],
                label=labels[class_val],
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )

        # Draw decision boundary
        h_range = np.linspace(H.min() - 0.5, H.max() + 0.5, 200)
        c_range = np.linspace(C.min() - 0.5, C.max() + 0.5, 200)
        hh, cc = np.meshgrid(h_range, c_range)
        Z = clf.predict(np.c_[hh.ravel(), cc.ravel()])
        Z = Z.reshape(hh.shape)
        ax.contour(hh, cc, Z, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')

        # Add filled contour for regions
        ax.contourf(hh, cc, Z, levels=[0, 0.5, 1], colors=['#2ecc71', '#e74c3c'], alpha=0.1)

        ax.set_xlabel('Normalized Permutation Entropy (H)', fontsize=14)
        ax.set_ylabel('Normalized Statistical Complexity (C)', fontsize=14)
        ax.set_title(
            f'Binary Pain Classification on C-H Plane\n'
            f'EDA Signal | d={self.dimension}, tau={self.tau} | Global Normalization\n'
            f'Silhouette: {silhouette:.4f} | Linear Accuracy: {accuracy*100:.2f}%',
            fontsize=14
        )
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(
            0.02, 0.02,
            'No ML model required - just linear separation in 2D',
            transform=ax.transAxes,
            fontsize=10,
            style='italic',
            alpha=0.7
        )

        plt.tight_layout()

        plot_path = self.output_dir / 'ch_plane_binary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved: {plot_path}")
        return str(plot_path)

    def create_raw_vs_normalized_comparison(
        self, H_raw: np.ndarray, C_raw: np.ndarray,
        H_norm: np.ndarray, C_norm: np.ndarray,
        y: np.ndarray
    ) -> str:
        """Create side-by-side comparison of raw vs normalized C-H plane."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        colors = {0: '#2ecc71', 1: '#e74c3c'}
        labels = {0: 'Baseline', 1: 'Pain'}

        # Raw data
        ax = axes[0]
        for class_val in [0, 1]:
            mask = y == class_val
            ax.scatter(
                H_raw[mask], C_raw[mask],
                c=colors[class_val],
                label=labels[class_val],
                alpha=0.6,
                s=40
            )
        ax.set_xlabel('Permutation Entropy (H)', fontsize=12)
        ax.set_ylabel('Statistical Complexity (C)', fontsize=12)
        ax.set_title('Raw Features', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Normalized data
        ax = axes[1]
        for class_val in [0, 1]:
            mask = y == class_val
            ax.scatter(
                H_norm[mask], C_norm[mask],
                c=colors[class_val],
                label=labels[class_val],
                alpha=0.6,
                s=40
            )
        ax.set_xlabel('Normalized H', fontsize=12)
        ax.set_ylabel('Normalized C', fontsize=12)
        ax.set_title('After Global Z-Score Normalization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Complexity-Entropy Plane: Raw vs Normalized\nEDA Signal | d={self.dimension}, tau={self.tau}',
            fontsize=14
        )
        plt.tight_layout()

        plot_path = self.output_dir / 'ch_plane_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved: {plot_path}")
        return str(plot_path)

    def run(self) -> dict:
        """Execute the C-H plane analysis."""
        print("=" * 60)
        print("STAGE 0: C-H Plane Binary Classification")
        print("=" * 60)

        # Load and prepare data
        df = self.load_data()
        H_raw, C_raw, y, subjects = self.prepare_features(df)

        # Apply normalization
        H_norm, C_norm = self.apply_global_normalization(H_raw, C_raw)

        # Evaluate classification
        results = self.evaluate_classification(H_norm, C_norm, y)

        # Create visualizations
        self.create_ch_plane_plot(
            H_norm, C_norm, y,
            results['classifier'],
            results['silhouette'],
            results['accuracy']
        )

        self.create_raw_vs_normalized_comparison(
            H_raw, C_raw, H_norm, C_norm, y
        )

        print("\n" + "=" * 60)
        print("Stage 0 C-H Plane Analysis Complete!")
        print("=" * 60)

        return results


if __name__ == '__main__':
    analyzer = CHPlaneAnalyzer()
    results = analyzer.run()
