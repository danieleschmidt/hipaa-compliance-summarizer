"""
Statistical Validation Framework for HIPAA Compliance Research.

Advanced statistical methods for validating PHI detection accuracy, including:
1. Hypothesis testing with proper statistical significance
2. Confidence interval estimation
3. Power analysis for study design
4. Multi-dataset validation with meta-analysis
5. Bootstrap resampling for robust estimates
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics with statistical significance testing."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Confidence intervals (lower, upper)
    precision_ci: Tuple[float, float] = (0.0, 1.0)
    recall_ci: Tuple[float, float] = (0.0, 1.0)
    f1_ci: Tuple[float, float] = (0.0, 1.0)

    # Statistical significance
    p_value_precision: float = 1.0
    p_value_recall: float = 1.0
    is_significant: bool = False

    # Effect size measures
    cohen_d: float = 0.0
    effect_size_category: str = "none"

    @property
    def precision(self) -> float:
        """Calculate precision with proper handling of edge cases."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / max(1, denominator)

    @property
    def recall(self) -> float:
        """Calculate recall with proper handling of edge cases."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / max(1, denominator)

    @property
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / max(1, denominator)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score with proper handling of edge cases."""
        p, r = self.precision, self.recall
        return 2 * (p * r) / max(0.001, p + r)

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / max(1, total)

    @property
    def matthews_correlation_coefficient(self) -> float:
        """Calculate Matthews Correlation Coefficient (MCC)."""
        tp, tn, fp, fn = self.true_positives, self.true_negatives, self.false_positives, self.false_negatives

        numerator = (tp * tn) - (fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / max(1, denominator)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'mcc': self.matthews_correlation_coefficient,
            'precision_ci': self.precision_ci,
            'recall_ci': self.recall_ci,
            'f1_ci': self.f1_ci,
            'p_value_precision': self.p_value_precision,
            'p_value_recall': self.p_value_recall,
            'is_statistically_significant': self.is_significant,
            'cohen_d': self.cohen_d,
            'effect_size': self.effect_size_category,
            'sample_size': self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        }


@dataclass
class StudyDesign:
    """Statistical study design parameters for validation experiments."""

    alpha: float = 0.05  # Type I error rate
    beta: float = 0.20   # Type II error rate (power = 1 - beta)
    effect_size: float = 0.1  # Minimum detectable effect size
    baseline_accuracy: float = 0.90  # Expected baseline accuracy

    @property
    def power(self) -> float:
        """Statistical power (1 - Type II error rate)."""
        return 1 - self.beta

    def calculate_required_sample_size(self, expected_accuracy: float) -> int:
        """Calculate required sample size for detecting specified effect."""
        # Using formula for comparing proportions
        p1 = self.baseline_accuracy
        p2 = expected_accuracy

        # Average proportion
        p_avg = (p1 + p2) / 2

        # Z-scores for alpha and beta
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.beta)

        # Sample size calculation
        numerator = (z_alpha + z_beta) ** 2 * 2 * p_avg * (1 - p_avg)
        denominator = (p1 - p2) ** 2

        sample_size = math.ceil(numerator / denominator)
        return max(30, sample_size)  # Minimum 30 for normal approximation


class StatisticalValidator:
    """Advanced statistical validation with hypothesis testing and confidence intervals."""

    def __init__(self, study_design: Optional[StudyDesign] = None):
        self.study_design = study_design or StudyDesign()
        self.validation_history: List[ValidationMetrics] = []
        self.bootstrap_iterations = 1000

    def validate_model_performance(
        self,
        predictions: List[bool],
        ground_truth: List[bool],
        confidence_scores: Optional[List[float]] = None,
        hypothesis_test: bool = True
    ) -> ValidationMetrics:
        """
        Comprehensive statistical validation of model performance.
        
        Args:
            predictions: Model predictions (True = PHI detected)
            ground_truth: Ground truth labels (True = actual PHI)
            confidence_scores: Optional confidence scores for threshold analysis
            hypothesis_test: Whether to perform statistical significance testing
            
        Returns:
            ValidationMetrics with comprehensive statistical analysis
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Calculate confusion matrix
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

        metrics = ValidationMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        )

        # Calculate confidence intervals using bootstrap
        if len(predictions) >= 30:  # Sufficient sample size
            self._calculate_confidence_intervals(metrics, predictions, ground_truth)

        # Perform hypothesis testing
        if hypothesis_test and len(predictions) >= 30:
            self._perform_hypothesis_testing(metrics)

        # Calculate effect size
        self._calculate_effect_size(metrics)

        # Store for historical analysis
        self.validation_history.append(metrics)

        return metrics

    def _calculate_confidence_intervals(
        self,
        metrics: ValidationMetrics,
        predictions: List[bool],
        ground_truth: List[bool]
    ):
        """Calculate confidence intervals using bootstrap resampling."""
        n_samples = len(predictions)
        precision_samples = []
        recall_samples = []
        f1_samples = []

        # Bootstrap resampling
        for _ in range(self.bootstrap_iterations):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_pred = [predictions[i] for i in indices]
            boot_truth = [ground_truth[i] for i in indices]

            # Calculate metrics for bootstrap sample
            boot_tp = sum(1 for p, g in zip(boot_pred, boot_truth) if p and g)
            boot_fp = sum(1 for p, g in zip(boot_pred, boot_truth) if p and not g)
            boot_fn = sum(1 for p, g in zip(boot_pred, boot_truth) if not p and g)

            # Avoid division by zero
            boot_precision = boot_tp / max(1, boot_tp + boot_fp)
            boot_recall = boot_tp / max(1, boot_tp + boot_fn)
            boot_f1 = 2 * (boot_precision * boot_recall) / max(0.001, boot_precision + boot_recall)

            precision_samples.append(boot_precision)
            recall_samples.append(boot_recall)
            f1_samples.append(boot_f1)

        # Calculate 95% confidence intervals
        alpha = self.study_design.alpha
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        metrics.precision_ci = (
            np.percentile(precision_samples, lower_percentile),
            np.percentile(precision_samples, upper_percentile)
        )

        metrics.recall_ci = (
            np.percentile(recall_samples, lower_percentile),
            np.percentile(recall_samples, upper_percentile)
        )

        metrics.f1_ci = (
            np.percentile(f1_samples, lower_percentile),
            np.percentile(f1_samples, upper_percentile)
        )

    def _perform_hypothesis_testing(self, metrics: ValidationMetrics):
        """Perform statistical significance testing."""
        # Test precision against baseline
        n_precision = metrics.true_positives + metrics.false_positives
        if n_precision > 0:
            # Binomial test for precision
            successes = metrics.true_positives
            p_value_precision = stats.binom_test(
                successes, n_precision, self.study_design.baseline_accuracy
            )
            metrics.p_value_precision = p_value_precision

        # Test recall against baseline
        n_recall = metrics.true_positives + metrics.false_negatives
        if n_recall > 0:
            # Binomial test for recall
            successes = metrics.true_positives
            p_value_recall = stats.binom_test(
                successes, n_recall, self.study_design.baseline_accuracy
            )
            metrics.p_value_recall = p_value_recall

        # Overall significance
        metrics.is_significant = (
            metrics.p_value_precision < self.study_design.alpha or
            metrics.p_value_recall < self.study_design.alpha
        )

    def _calculate_effect_size(self, metrics: ValidationMetrics):
        """Calculate Cohen's d effect size."""
        # Cohen's d for the difference in accuracy from baseline
        observed_accuracy = metrics.accuracy
        baseline_accuracy = self.study_design.baseline_accuracy

        # Pooled standard deviation estimation
        # For proportions, std = sqrt(p * (1-p))
        std_baseline = math.sqrt(baseline_accuracy * (1 - baseline_accuracy))
        std_observed = math.sqrt(observed_accuracy * (1 - observed_accuracy))
        pooled_std = math.sqrt((std_baseline**2 + std_observed**2) / 2)

        if pooled_std > 0:
            cohen_d = (observed_accuracy - baseline_accuracy) / pooled_std
            metrics.cohen_d = cohen_d

            # Categorize effect size
            abs_d = abs(cohen_d)
            if abs_d < 0.2:
                metrics.effect_size_category = "negligible"
            elif abs_d < 0.5:
                metrics.effect_size_category = "small"
            elif abs_d < 0.8:
                metrics.effect_size_category = "medium"
            else:
                metrics.effect_size_category = "large"

    def compare_models(
        self,
        model1_predictions: List[bool],
        model2_predictions: List[bool],
        ground_truth: List[bool]
    ) -> Dict[str, Any]:
        """Compare two models using paired statistical tests."""

        if not (len(model1_predictions) == len(model2_predictions) == len(ground_truth)):
            raise ValueError("All prediction arrays must have same length")

        # Validate each model
        metrics1 = self.validate_model_performance(model1_predictions, ground_truth, hypothesis_test=False)
        metrics2 = self.validate_model_performance(model2_predictions, ground_truth, hypothesis_test=False)

        # McNemar's test for comparing paired binary classifiers
        # Count the disagreements
        model1_correct = [p == g for p, g in zip(model1_predictions, ground_truth)]
        model2_correct = [p == g for p, g in zip(model2_predictions, ground_truth)]

        # 2x2 contingency table
        both_correct = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if m1 and m2)
        model1_only_correct = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if m1 and not m2)
        model2_only_correct = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if not m1 and m2)
        both_incorrect = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if not m1 and not m2)

        # McNemar's test statistic
        n_discordant = model1_only_correct + model2_only_correct
        if n_discordant > 25:  # Use normal approximation
            chi_square = ((abs(model1_only_correct - model2_only_correct) - 1) ** 2) / n_discordant
            p_value = 1 - stats.chi2.cdf(chi_square, df=1)
        elif n_discordant > 0:  # Use binomial test for small samples
            p_value = stats.binom_test(model1_only_correct, n_discordant, 0.5) * 2  # Two-tailed
        else:
            p_value = 1.0  # No disagreements, no difference

        comparison_result = {
            'model1_metrics': metrics1.get_summary(),
            'model2_metrics': metrics2.get_summary(),
            'mcnemar_test': {
                'both_correct': both_correct,
                'model1_only_correct': model1_only_correct,
                'model2_only_correct': model2_only_correct,
                'both_incorrect': both_incorrect,
                'p_value': p_value,
                'significant_difference': p_value < self.study_design.alpha
            },
            'accuracy_difference': metrics2.accuracy - metrics1.accuracy,
            'f1_difference': metrics2.f1_score - metrics1.f1_score,
        }

        return comparison_result

    def perform_power_analysis(
        self,
        current_sample_size: int,
        observed_accuracy: float
    ) -> Dict[str, Any]:
        """Perform statistical power analysis for the current study."""

        # Calculate achieved power
        effect_size = observed_accuracy - self.study_design.baseline_accuracy

        # For proportion test, calculate power
        p1 = self.study_design.baseline_accuracy
        p2 = observed_accuracy
        p_pooled = (p1 + p2) / 2

        # Standard error
        se = math.sqrt(2 * p_pooled * (1 - p_pooled) / current_sample_size)

        # Critical value
        z_alpha = stats.norm.ppf(1 - self.study_design.alpha / 2)

        # Power calculation
        z_score = abs(p1 - p2) / se
        achieved_power = 1 - stats.norm.cdf(z_alpha - z_score) + stats.norm.cdf(-z_alpha - z_score)

        # Required sample size for desired power
        required_size = self.study_design.calculate_required_sample_size(observed_accuracy)

        power_analysis = {
            'current_sample_size': current_sample_size,
            'observed_accuracy': observed_accuracy,
            'baseline_accuracy': self.study_design.baseline_accuracy,
            'effect_size': effect_size,
            'achieved_power': achieved_power,
            'desired_power': self.study_design.power,
            'required_sample_size': required_size,
            'sample_size_adequate': current_sample_size >= required_size,
            'power_adequate': achieved_power >= self.study_design.power,
        }

        return power_analysis

    def meta_analysis(self, validation_results: List[ValidationMetrics]) -> Dict[str, Any]:
        """Perform meta-analysis across multiple validation studies."""

        if len(validation_results) < 2:
            return {'error': 'Need at least 2 studies for meta-analysis'}

        # Extract effect sizes (accuracy differences from baseline)
        effect_sizes = []
        weights = []

        for metrics in validation_results:
            sample_size = metrics.true_positives + metrics.false_positives + metrics.true_negatives + metrics.false_negatives
            effect_size = metrics.accuracy - self.study_design.baseline_accuracy

            # Weight by sample size (inverse variance weighting approximation)
            weight = sample_size

            effect_sizes.append(effect_size)
            weights.append(weight)

        # Calculate weighted mean effect size
        weighted_effect_size = np.average(effect_sizes, weights=weights)

        # Calculate heterogeneity (Q statistic)
        weighted_mean = weighted_effect_size
        q_statistic = sum(w * (es - weighted_mean)**2 for w, es in zip(weights, effect_sizes))

        # Degrees of freedom
        df = len(effect_sizes) - 1

        # I-squared (proportion of variation due to heterogeneity)
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0

        # Combined sample size
        total_sample_size = sum(
            m.true_positives + m.false_positives + m.true_negatives + m.false_negatives
            for m in validation_results
        )

        # Overall metrics (simple average for demonstration)
        avg_precision = np.mean([m.precision for m in validation_results])
        avg_recall = np.mean([m.recall for m in validation_results])
        avg_f1 = np.mean([m.f1_score for m in validation_results])
        avg_accuracy = np.mean([m.accuracy for m in validation_results])

        meta_analysis_result = {
            'number_of_studies': len(validation_results),
            'total_sample_size': total_sample_size,
            'weighted_effect_size': weighted_effect_size,
            'average_metrics': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'accuracy': avg_accuracy,
            },
            'heterogeneity': {
                'q_statistic': q_statistic,
                'degrees_of_freedom': df,
                'i_squared': i_squared,
                'interpretation': self._interpret_heterogeneity(i_squared)
            },
            'statistical_significance': {
                'p_value_q': 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0,
                'heterogeneous': i_squared > 0.5,
            }
        }

        return meta_analysis_result

    def _interpret_heterogeneity(self, i_squared: float) -> str:
        """Interpret I-squared heterogeneity statistic."""
        if i_squared < 0.25:
            return "low heterogeneity"
        elif i_squared < 0.5:
            return "moderate heterogeneity"
        elif i_squared < 0.75:
            return "substantial heterogeneity"
        else:
            return "considerable heterogeneity"

    def generate_validation_report(self, metrics: ValidationMetrics) -> str:
        """Generate a comprehensive validation report."""

        report = f"""
# Statistical Validation Report

## Performance Metrics
- **Precision**: {metrics.precision:.3f} (95% CI: {metrics.precision_ci[0]:.3f} - {metrics.precision_ci[1]:.3f})
- **Recall**: {metrics.recall:.3f} (95% CI: {metrics.recall_ci[0]:.3f} - {metrics.recall_ci[1]:.3f})
- **F1 Score**: {metrics.f1_score:.3f} (95% CI: {metrics.f1_ci[0]:.3f} - {metrics.f1_ci[1]:.3f})
- **Accuracy**: {metrics.accuracy:.3f}
- **Specificity**: {metrics.specificity:.3f}
- **Matthews Correlation Coefficient**: {metrics.matthews_correlation_coefficient:.3f}

## Statistical Significance
- **Precision p-value**: {metrics.p_value_precision:.4f}
- **Recall p-value**: {metrics.p_value_recall:.4f}
- **Statistically Significant**: {metrics.is_significant}
- **Effect Size (Cohen's d)**: {metrics.cohen_d:.3f} ({metrics.effect_size_category})

## Sample Size
- **True Positives**: {metrics.true_positives}
- **False Positives**: {metrics.false_positives}
- **True Negatives**: {metrics.true_negatives}
- **False Negatives**: {metrics.false_negatives}
- **Total**: {metrics.true_positives + metrics.false_positives + metrics.true_negatives + metrics.false_negatives}

## Interpretation
The model shows {'statistically significant' if metrics.is_significant else 'no statistically significant'} 
improvement over baseline performance with a {metrics.effect_size_category} effect size.
        """

        return report.strip()


def create_statistical_validator(
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_accuracy: float = 0.90
) -> StatisticalValidator:
    """Factory function to create a statistical validator with custom parameters."""
    study_design = StudyDesign(
        alpha=alpha,
        beta=1-power,
        baseline_accuracy=baseline_accuracy
    )
    return StatisticalValidator(study_design)
