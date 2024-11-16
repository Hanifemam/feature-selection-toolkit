import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from minepy import MINE
from ReliefF import ReliefF
from scipy.stats import (chi2_contingency, f_oneway, fisher_exact, kendalltau,
                         kruskal, ks_2samp, mannwhitneyu, pointbiserialr,
                         spearmanr, ttest_ind)
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lars, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from statsmodels.stats.anova import anova_lm

"""
Feature Ranking and Selection Methods with Detailed Applicability and Guidelines

This module provides a comprehensive set of functions for ranking and selecting features
using various statistical tests, correlation measures, and model-based methods.
Each function includes detailed information about its applicability, assumptions,
and guidelines for use in different types of machine learning models.

"""

# =============================================================================
# CORRELATION-BASED METHODS
# =============================================================================


def compute_spearman_correlation(features, target):
    """
    Computes Spearman rank-order correlation coefficient between each feature and the target.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification (with numerically encoded target labels)
        - Regression problems
        - Non-normally distributed data
        - Ordinal data
        - Monotonic relationships (both linear and non-linear)
    - Robust to outliers
    - Captures monotonic relationships

    Notes:
    - Ensure that the target variable is numerical or ordinal.
    - For classification problems, encode target labels numerically (e.g., 0, 1, 2).
    - Spearman correlation is non-parametric and does not assume a linear relationship.

    Useful for:
    - Preliminary feature screening
    - Models sensitive to monotonic relationships (e.g., tree-based models, KNN)

    Returns:
    - Series of absolute Spearman correlation coefficients for each feature.
    """
    correlations = pd.Series(index=features.columns, dtype=float)
    for column in features.columns:
        corr, _ = spearmanr(features[column], target)
        correlations[column] = abs(corr)
    return correlations


def compute_kendall_tau(features, target):
    """
    Computes Kendall's Tau correlation coefficient between each feature and the target.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification (with numerically encoded target labels)
        - Regression problems
        - Non-normally distributed data
        - Ordinal data
        - Monotonic relationships
    - More robust to outliers than Spearman
    - Better for small sample sizes
    - Captures monotonic relationships

    Notes:
    - Ensure that the target variable is numerical or ordinal.
    - For classification problems, encode target labels numerically.

    Useful for:
    - Preliminary feature screening
    - Models sensitive to monotonic relationships

    Returns:
    - Series of absolute Kendall's Tau coefficients for each feature.
    """
    tau_values = pd.Series(index=features.columns, dtype=float)
    for column in features.columns:
        tau, _ = kendalltau(features[column], target)
        tau_values[column] = abs(tau)
    return tau_values


def compute_pearson_correlation(features, target):
    """
    Computes Pearson correlation coefficient between each feature and the target.

    Applicability:
    - Suitable for:
        - Regression problems
        - Continuous and normally distributed features and target
        - Linear relationships only
    - Sensitive to outliers

    Notes:
    - Not appropriate for classification problems with categorical target variables.
    - Assumes both features and target are continuous and normally distributed.
    - Measures linear correlation between variables.

    Useful for:
    - Linear regression models
    - Identifying linear relationships between features and target

    Returns:
    - Series of absolute Pearson correlation coefficients for each feature.
    """
    correlations = pd.Series(index=features.columns, dtype=float)
    for column in features.columns:
        corr = features[column].corr(target)
        correlations[column] = abs(corr)
    return correlations


def compute_mic(features, target):
    """
    Computes the Maximal Information Coefficient (MIC) for each feature.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Continuous features
    - Assumptions:
        - None

    Notes:
    - MIC can detect a wide range of associations, both linear and non-linear.
    - Computationally intensive for large datasets.

    Useful for:
    - Identifying any type of dependency between features and target
    - Feature selection when non-linear relationships are important

    Returns:
    - Series of MIC scores for each feature.
    """

    mine = MINE(alpha=0.6, c=15)
    mic_scores = []

    for column in features.columns:
        mine.compute_score(features[column], target)
        mic = mine.mic()
        mic_scores.append(mic)

    return pd.Series(mic_scores, index=features.columns)


def compute_phi_coefficient(features, target):
    """
    Computes the Phi Coefficient between each binary feature and the binary target.

    Applicability:
    - Suitable for:
        - Binary features
        - Binary target

    Notes:
    - Phi coefficient ranges from -1 to 1
    - Similar to Pearson correlation for dichotomous variables
    - Higher absolute values indicate stronger association

    Useful for:
    - Assessing the association between binary variables
    - Feature selection in binary classification

    Returns:
    - Series of absolute Phi coefficients for each feature
    """
    scores = {}
    # Ensure binary target is encoded as 0 and 1
    target_values = sorted(target.unique())
    target = target.replace({target_values[0]: 0, target_values[1]: 1})

    for column in features.columns:
        feature = features[column]
        if feature.nunique() == 2:
            contingency_table = pd.crosstab(feature, target)
            if contingency_table.shape == (2, 2):
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.to_numpy().sum()
                phi = np.sqrt(chi2 / n)
                scores[column] = abs(phi)
            else:
                scores[column] = np.nan
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


def compute_partial_correlation(features, target):
    """
    Computes partial correlation coefficients between each feature and the target.

    Applicability:
    - Suitable for regression problems
    - Continuous features and target

    Notes:
    - Controls for the influence of other variables
    - Assumes linear relationships

    Returns:
    - Series of partial correlation coefficients for each feature
    """
    from pingouin import partial_corr

    data = features.copy()
    data["target"] = target
    scores = {}
    for column in features.columns:
        covariates = [col for col in features.columns if col != column]
        result = partial_corr(
            data=data, x=column, y="target", covar=covariates, method="pearson"
        )
        scores[column] = abs(result["r"].values[0])
    return pd.Series(scores).sort_values(ascending=False)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================


def compute_mwu_test(features, target):
    """
    Computes Mann-Whitney U test for each feature between two classes.
    (ranking is the same is Wilcoxon)

    Applicability:
    - Suitable for:
        - Binary classification only
        - Non-normally distributed data
        - Ordinal or continuous features
        - Large sample sizes
    - Non-parametric alternative to the t-test

    Notes:
    - Tests whether distributions of the two groups are different.
    - Does not assume normality.
    - Sensitive to differences in distributions.

    Useful for:
    - Identifying features that differ significantly between two classes
    - Models where feature distributions between classes are important

    Returns:
    - Series of scores (-log10 p-value) for each feature.
    """
    mwu_scores = pd.Series(index=features.columns, dtype=float)
    unique_classes = np.unique(target)

    if len(unique_classes) != 2:
        raise ValueError(
            "Mann-Whitney U test is only applicable for binary classification."
        )

    for column in features.columns:
        group1 = features[column][target == unique_classes[0]]
        group2 = features[column][target == unique_classes[1]]
        statistic, pvalue = mannwhitneyu(group1, group2, alternative="two-sided")
        mwu_scores[column] = -np.log10(pvalue) if pvalue > 0 else np.inf
    return mwu_scores


def compute_kruskal_wallis(features, target):
    """
    Computes Kruskal-Wallis H test for each feature across multiple classes.

    Applicability:
    - Suitable for:
        - Multi-class classification (three or more classes)
        - Non-normally distributed data
        - Ordinal or continuous features
    - Non-parametric alternative to ANOVA

    Notes:
    - Tests whether samples originate from the same distribution.
    - Does not assume normality or equal variances.

    Useful for:
    - Identifying features that differ significantly across multiple classes
    - Models sensitive to distributional differences of features across classes

    Returns:
    - Series of scores (-log10 p-value) for each feature.
    """
    kw_scores = pd.Series(index=features.columns, dtype=float)
    unique_classes = np.unique(target)

    for column in features.columns:
        groups = [features[column][target == cls] for cls in unique_classes]
        statistic, pvalue = kruskal(*groups)
        kw_scores[column] = -np.log10(pvalue) if pvalue > 0 else np.inf
    return kw_scores


def compute_ks_test(features, target):
    """
    Computes Kolmogorov-Smirnov test for each feature between two classes.

    Applicability:
    - Suitable for:
        - Binary classification only
        - Non-parametric test
        - Continuous features
    - Tests if two samples come from the same distribution
    - Sensitive to differences in distributions (location, shape, etc.)

    Notes:
    - Does not assume any specific distribution.
    - More powerful than Mann-Whitney U for detecting distribution differences.

    Useful for:
    - Identifying features with different distributions between two classes
    - Models where distributional differences impact performance

    Returns:
    - Series of scores (-log10 p-value) for each feature.
    """
    ks_scores = pd.Series(index=features.columns, dtype=float)
    unique_classes = np.unique(target)

    if len(unique_classes) != 2:
        raise ValueError("KS test is only applicable for binary classification")

    for column in features.columns:
        group1 = features[column][target == unique_classes[0]]
        group2 = features[column][target == unique_classes[1]]
        statistic, pvalue = ks_2samp(group1, group2)
        ks_scores[column] = -np.log10(pvalue) if pvalue > 0 else np.inf
    return ks_scores


def compute_chi_square(features, target, bins=10):
    """
    Computes chi-square test between each feature and the target variable.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Categorical features or binned continuous features
    - Tests independence between feature and target

    Notes:
    - Features need to be categorical.
    - For continuous features, binning is required.
    - No distributional assumptions.

    Useful for:
    - Identifying features that are not independent of the target
    - Models using categorical features (e.g., Naive Bayes, Decision Trees)

    Returns:
    - Series of scores (-log10 p-value) for each feature.
    """
    chi_scores = pd.Series(index=features.columns, dtype=float)

    for column in features.columns:
        # Bin continuous features
        if features[column].nunique() > bins:
            feature_binned = pd.qcut(
                features[column], bins, labels=False, duplicates="drop"
            )
        else:
            feature_binned = features[column]
        contingency_table = pd.crosstab(feature_binned, target)
        chi2, pvalue, _, _ = chi2_contingency(contingency_table)
        chi_scores[column] = -np.log10(pvalue) if pvalue > 0 else np.inf
    return chi_scores


def compute_t_test(features, target):
    """
    Performs t-test between two classes for each feature.

    Applicability:
    - Suitable for:
        - Binary classification
        - Continuous features
    - Assumptions:
        - Data is normally distributed
        - Homogeneity of variances (can use Welch's t-test if not met)
    - Notes:
        - Tests whether the means of two groups are significantly different.

    Useful for:
    - Identifying features that show significant differences between two classes

    Returns:
    - Series of -log10(p-values) for each feature.
    """
    t_scores = pd.Series(index=features.columns, dtype=float)
    classes = target.unique()
    if len(classes) != 2:
        raise ValueError("T-test is only applicable for binary classification.")

    group1 = features[target == classes[0]]
    group2 = features[target == classes[1]]

    for column in features.columns:
        t_stat, p_value = ttest_ind(group1[column], group2[column], equal_var=False)
        t_scores[column] = -np.log10(p_value) if p_value > 0 else np.inf

    return t_scores


def compute_fishers_exact(features, target):
    """
    Computes Fisher's Exact Test for each binary or categorical feature.

    Applicability:
    - Suitable for:
        - Binary classification
    - Works with:
        - Binary features
        - Categorical features with two categories

    Notes:
    - Tests the null hypothesis of independence between the feature and the target.
    - Best for 2x2 contingency tables.
    - For larger tables, the test is computationally intensive and less commonly used.

    Useful for:
    - Feature selection in datasets with small sample sizes
    - Assessing the association between binary features and binary target

    Returns:
    - Series of -log10(p-values) for each feature
    """
    scores = {}
    # Ensure binary target is encoded as 0 and 1
    target_values = sorted(target.unique())
    target = target.replace({target_values[0]: 0, target_values[1]: 1})

    for column in features.columns:
        feature = features[column]
        if feature.nunique() == 2:
            contingency_table = pd.crosstab(feature, target)
            if contingency_table.shape == (2, 2):
                # Perform Fisher's Exact Test
                _, p_value = fisher_exact(contingency_table)
                scores[column] = -np.log10(
                    p_value + 1e-10
                )  # Small constant to avoid log(0)
            else:
                scores[column] = np.nan
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


def compute_likelihood_ratio(features, target):
    """
    Computes Likelihood Ratio Test p-values for each categorical feature in logistic regression.

    Applicability:
    - Suitable for:
        - Binary classification
    - Works with:
        - Categorical features
        - Binary features

    Notes:
    - Assesses the significance of each feature by comparing models with and without the feature.
    - Lower p-values indicate that the feature significantly improves model fit.

    Useful for:
    - Feature selection in logistic regression models
    - Assessing the significance of categorical features

    Returns:
    - Series of -log10(p-values) for each feature
    """
    scores = {}
    # Encode target as 0 and 1
    target_values = sorted(target.unique())
    target = target.replace({target_values[0]: 0, target_values[1]: 1})

    for column in features.columns:
        feature = features[[column]]
        if feature[column].dtype.kind in {"O", "b", "i", "u"}:
            X_full = pd.get_dummies(feature, drop_first=True)
            if X_full.shape[1] == 0:
                continue
            X_full = sm.add_constant(X_full)
            model_full = sm.Logit(target, X_full).fit(disp=0)

            # Null model (intercept only)
            X_null = sm.add_constant(pd.DataFrame(index=target.index))
            model_null = sm.Logit(target, X_null).fit(disp=0)

            # Likelihood Ratio Test
            lr_stat = 2 * (model_full.llf - model_null.llf)
            df_diff = model_full.df_model - model_null.df_model
            p_value = sm.stats.stattools.stats.chisqprob(lr_stat, df_diff)
            scores[column] = -np.log10(p_value + 1e-10)
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


def compute_eta_coefficient(features, target):
    """
    Computes the Eta Coefficient for each categorical feature against a continuous target.

    Applicability:
    - Suitable for:
        - Regression problems
    - Works with:
        - Categorical features
        - Continuous target

    Notes:
    - Eta coefficient ranges from 0 to 1
    - Higher values indicate a stronger association
    - Related to the proportion of variance in the continuous variable explained by the categorical variable

    Returns:
    - Series of Eta coefficients for each feature
    """

    scores = {}
    for column in features.columns:
        feature = features[column]
        if feature.dtype.kind in {"O", "b", "i", "u"}:
            categories = feature.unique()
            mean_total = target.mean()
            ss_between = sum(
                [
                    len(target[feature == cat])
                    * (target[feature == cat].mean() - mean_total) ** 2
                    for cat in categories
                ]
            )
            ss_total = sum((target - mean_total) ** 2)
            eta_squared = ss_between / ss_total if ss_total != 0 else 0
            eta = np.sqrt(eta_squared)
            scores[column] = eta
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


def compute_anova_f_test(features, target):
    """
    Computes ANOVA F-test between each feature and the target variable.

    Applicability:
    - Suitable for:
        - Multi-class classification
        - Continuous features
        - Normally distributed features within each class
        - Assumes linear relationships
    - Assumption of equal variances across classes

    Notes:
    - Tests if the means of different groups are significantly different.
    - Sensitive to outliers and non-normal distributions.
    - May require transformations (e.g., log, Box-Cox) to meet assumptions.

    Useful for:
    - Feature selection for linear models
    - Situations where assumptions of normality are met

    Returns:
    - Series of scores (-log10 p-value) for each feature.
    """
    f_scores, pvalues = f_classif(features, target)
    scores = -np.log10(pvalues)
    return pd.Series(scores, index=features.columns)


def compute_fisher_score(features, target):
    """
    Computes Fisher Score (ANOVA F-value) for each feature.

    Applicability:
    - Suitable for:
        - Binary classification
        - Continuous features
    - Assumes:
        - Features are continuous
        - Classes are separable
        - Features are statistically independent

    Notes:
    - Fisher Score is similar to ANOVA F-test for binary classification.
    - Higher scores indicate a feature with better class discriminative power.
    - Assumes normal distribution and homogeneity of variances.
    - May require data transformation (e.g., log, Box-Cox) to meet assumptions.

    Useful for:
    - Feature selection for linear models (e.g., logistic regression, SVM)
    - Situations where assumptions of normality are met

    Returns:
    - Series of Fisher Scores for each feature.
    """
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(features, target)
    scores = selector.scores_
    return pd.Series(scores, index=features.columns)


def compute_anova_categorical(features, target):
    """
    Computes ANOVA F-test for categorical features against a continuous target.

    Applicability:
    - Suitable for:
        - Regression problems
        - Continuous target variable
        - Categorical features

    Notes:
    - Tests if the means of the target variable are significantly different across feature categories.
    - Assumes normal distribution of the target variable within groups and homogeneity of variances.

    Useful for:
    - Identifying categorical features with significant impact on a continuous target

    Returns:
    - Series of F-statistics for each feature
    """
    scores = {}
    for column in features.columns:
        feature = features[column]
        if feature.dtype.kind in {"O", "b", "i", "u"}:
            groups = [target[feature == cat] for cat in feature.unique()]
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                scores[column] = f_stat
            else:
                scores[column] = np.nan
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


# =============================================================================
# INFORMATION THEORY BASED
# =============================================================================


def compute_mutual_information(features, target):
    """
    Computes mutual information between each feature and the target variable.
    (The same as Information Gain for ranking features)

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Captures non-linear relationships
        - Works with any feature type (discrete or continuous)
    - No distribution assumptions
    - Scale-invariant

    Notes:
    - Mutual Information measures the amount of information one variable provides about another.
    - Higher values indicate a stronger relationship.

    Useful for:
    - Identifying features with strong predictive power
    - Models that can capture non-linear relationships (e.g., tree-based models)

    Returns:
    - Series of mutual information scores for each feature.
    """
    mi_scores = mutual_info_classif(features, target, random_state=42)
    return pd.Series(mi_scores, index=features.columns)


def compute_iv_woe(features, target, bins=10):
    """
    Computes Information Value (IV) and Weight of Evidence (WOE) for categorical features.

    Applicability:
    - Suitable for:
        - Binary classification
    - Works with:
        - Categorical features
        - Binned continuous features

    Notes:
    - IV indicates the predictive power of a feature:
        - IV < 0.02: Not predictive
        - 0.02 <= IV < 0.1: Weak predictor
        - 0.1 <= IV < 0.3: Medium predictor
        - 0.3 <= IV < 0.5: Strong predictor
        - IV >= 0.5: Suspicious or too good to be true (potential data issues)
    - WOE provides a method for encoding categorical variables.

    Useful for:
    - Feature selection in credit scoring and risk modeling
    - Encoding categorical variables for logistic regression

    Returns:
    - iv_scores: Series containing IV scores for each feature
    """
    iv_scores = {}
    woe_dict = {}

    # Ensure binary target is encoded as 0 and 1
    target_values = sorted(target.unique())
    target = target.replace({target_values[0]: 0, target_values[1]: 1})

    for column in features.columns:
        # Handle categorical and continuous features
        if (
            features[column].dtype.kind in {"O", "b", "i", "u"}
            or features[column].nunique() <= bins
        ):
            # Treat as categorical
            feature = features[column].astype(str)
        else:
            # Bin continuous variables
            feature = pd.qcut(features[column], bins, duplicates="drop").astype(str)

        df = pd.DataFrame({"feature": feature, "target": target})
        groups = df.groupby("feature")["target"]

        # Calculate event (bad) and non-event (good) counts
        stats = groups.agg(["count", "sum"])
        stats.columns = ["total", "bad"]
        stats["good"] = stats["total"] - stats["bad"]

        # Replace zeros to avoid division by zero
        stats.replace({"bad": {0: 0.5}, "good": {0: 0.5}}, inplace=True)

        # Calculate distributions
        stats["dist_bad"] = stats["bad"] / stats["bad"].sum()
        stats["dist_good"] = stats["good"] / stats["good"].sum()

        # Calculate WOE
        stats["woe"] = np.log(stats["dist_good"] / stats["dist_bad"])

        # Calculate IV
        stats["iv"] = (stats["dist_good"] - stats["dist_bad"]) * stats["woe"]

        # Total IV for the feature
        iv = stats["iv"].sum()
        iv_scores[column] = iv

        # Store WOE mapping
        woe_mapping = stats["woe"].to_dict()
        woe_dict[column] = woe_mapping

    iv_scores = pd.Series(iv_scores).sort_values(ascending=False)
    return iv_scores


def compute_pmi(features, target):
    """
    Computes Pointwise Mutual Information (PMI) between each categorical feature and the target.

    Applicability:
    - Suitable for:
        - Categorical features
        - Binary and multi-class classification

    Notes:
    - PMI highlights how much the actual co-occurrence of two values deviates from expectation under independence.
    - High PMI indicates a strong association between feature category and target class.

    Useful for:
    - Feature selection in text analysis, NLP, and categorical data

    Returns:
    - Series of average PMI scores for each feature
    """
    scores = {}
    total_samples = len(target)
    target_counts = target.value_counts(normalize=True)

    for column in features.columns:
        feature = features[column]
        if feature.dtype.kind in {"O", "b", "i", "u"}:
            df = pd.DataFrame({"feature": feature, "target": target})
            contingency_table = pd.crosstab(df["feature"], df["target"])
            feature_probs = contingency_table.sum(axis=1) / total_samples
            pmi_total = 0
            count = 0
            for feature_value, row in contingency_table.iterrows():
                for target_value, joint_count in row.items():
                    if joint_count > 0:
                        joint_prob = joint_count / total_samples
                        exp_prob = (
                            feature_probs[feature_value] * target_counts[target_value]
                        )
                        pmi = np.log2(joint_prob / exp_prob)
                        pmi_total += pmi
                        count += 1
            avg_pmi = pmi_total / count if count != 0 else 0
            scores[column] = avg_pmi
        else:
            scores[column] = np.nan

    return pd.Series(scores).dropna().sort_values(ascending=False)


# =============================================================================
# MODEL-BASED METHODS
# =============================================================================


def compute_rf_importance(features, target):
    """
    Computes feature importance using a Random Forest classifier.
    (Gini Importance)

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Handles mixed feature types
        - Captures non-linear relationships and feature interactions
    - No distribution assumptions

    Notes:
    - Random Forest feature importance is based on mean decrease in impurity.
    - Can be biased towards features with more levels.
    - Features should be numerical and appropriately encoded.

    Useful for:
    - Feature selection for tree-based models
    - Understanding feature impact in ensemble methods

    Returns:
    - Series of feature importances from the Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, target)
    importances = pd.Series(rf.feature_importances_, index=features.columns)
    return importances


def compute_lasso_importance(features, target):
    """
    Computes feature importance using Logistic Regression with L1 regularization (Lasso).

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Features should be standardized or scaled
        - Assumes linear relationships between features and target
    - Encourages sparsity in feature selection
    - Less effective for highly correlated features

    Notes:
    - Lasso can zero out coefficients for less important features.
    - For multi-class classification, using 'saga' solver is required with 'multinomial' option.
    - Scaling or standardizing features is important to ensure that coefficient sizes reflect feature importance.

    Useful for:
    - Linear models requiring feature selection
    - Situations where interpretability and simplicity are important

    Returns:
    - Series of absolute coefficient values (importances) for each feature.
    """

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    features_std = pd.DataFrame(features_std, columns=features.columns)

    # For multi-class classification, need to use 'saga' solver with 'multinomial' option
    unique_classes = np.unique(target)
    if len(unique_classes) > 2:
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            multi_class="multinomial",
            random_state=42,
            max_iter=10000,
        )
    else:
        clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    clf.fit(features_std, target)
    # For multi-class, coef_ is an array of shape (n_classes, n_features)
    # We can take the mean of absolute coefficients across classes
    coef = np.abs(clf.coef_)
    if len(coef.shape) > 1:
        importances = np.mean(coef, axis=0)
    else:
        importances = coef.flatten()
    return pd.Series(importances, index=features.columns)


def compute_permutation_importance(features, target, model=None, n_repeats=10):
    """
    Computes permutation feature importance.

    Applicability:
    - Suitable for:
        - Any classification problem
        - Model-agnostic method
        - Captures actual feature impact on model performance
    - Computationally intensive

    Notes:
    - Permutation importance measures the increase in prediction error after permuting the feature.
    - Requires a trained model.
    - Handles non-linear relationships if the base model does.

    Useful for:
    - Understanding feature importance in any model
    - Validating feature impact post-model training

    Returns:
    - Series of permutation importance scores for each feature.
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    result = permutation_importance(
        model, features, target, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=features.columns)


def compute_lars_coefficients(features, target):
    """
    Computes feature importance using Least Angle Regression (LARS).

    Applicability:
    - Suitable for:
        - Regression problems
        - Continuous features
    - Assumptions:
        - Linear relationships
        - Features and target are continuous

    Notes:
    - Efficient for high-dimensional data.
    - Selects variables to include in the model in a step-wise fashion.

    Useful for:
    - Feature selection in linear regression models
    - High-dimensional data with many features

    Returns:
    - Series of absolute coefficients for each feature.
    """
    model = Lars(n_nonzero_coefs=features.shape[1])
    model.fit(features, target)
    coef = np.abs(model.coef_)
    return pd.Series(coef, index=features.columns)


def compute_lda(features, target):
    """
    Performs Linear Discriminant Analysis and returns the coefficients for each feature.

    Applicability:
    - Suitable for:
        - Classification tasks (binary and multiclass)
        - Continuous features
    - Assumptions:
        - Normally distributed features
        - Equal covariance matrices across classes
    - Notes:
        - LDA maximizes class separability.
        - Coefficients indicate the importance of features in discriminating classes.

    Useful for:
        - Feature selection in classification models
        - Dimensionality reduction

    Returns:
    - Series of absolute coefficients from LDA for each feature.
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, target)
    coef = np.abs(lda.coef_).sum(axis=0)
    return pd.Series(coef, index=features.columns)


def compute_pls_feature_importance(features, target, n_components=2):
    """
    Performs Partial Least Squares Regression and returns feature importances.

    Applicability:
    - Suitable for:
        - Regression problems
        - When predictors are highly collinear
    - Assumptions:
        - Linear relationships
    - Notes:
        - Projects predictors and response to a new space.
        - Handles multicollinearity well.

    Useful for:
        - Reducing dimensionality
        - Feature extraction in regression models

    Returns:
    - Series of feature importances based on PLS.
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(features, target)
    # Feature importance is approximated by the sum of the absolute coefficients
    coef = np.abs(pls.coef_).flatten()
    return pd.Series(coef, index=features.columns)


def compute_stability_selection(
    features, target, model=None, n_subsamples=100, sample_fraction=0.75, threshold=0.6
):
    """
    Performs feature selection using Stability Selection.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Regression problems
        - High-dimensional datasets
    - Assumptions:
        - Base estimator should be appropriate for the data
    - Notes:
        - Stability Selection involves running a feature selection algorithm multiple times
          on bootstrapped samples and aggregating the results.
        - Features selected consistently across subsamples are considered stable.
        - Helps to improve robustness and interpretability of feature selection.
        - Handles high-dimensional data well.

    Useful for:
    - Identifying robust features that consistently contribute to the model
    - Reducing overfitting by selecting stable features
    - Models sensitive to feature selection variance

    Parameters:
    - features (DataFrame): Feature matrix
    - target (Series or array-like): Target variable
    - model: Base estimator with 'coef_' or 'feature_importances_' attribute (default: LogisticRegression with L1 penalty)
    - n_subsamples (int): Number of bootstrap samples (default: 100)
    - sample_fraction (float): Fraction of samples to include in each bootstrap sample (default: 0.75)
    - threshold (float): Proportion threshold to select features (default: 0.6)

    Returns:
    - selection_frequency (Series): Frequency of feature selection across subsamples
    """
    if model is None:
        # Default to Logistic Regression with L1 penalty
        model = LogisticRegression(
            penalty="l1", solver="liblinear", random_state=42, max_iter=1000
        )

    n_samples = features.shape[0]
    n_features = features.shape[1]
    selection_counts = pd.Series(0, index=features.columns, dtype=int)

    for i in range(n_subsamples):
        # Bootstrap resampling
        X_sample, y_sample = resample(
            features,
            target,
            n_samples=int(sample_fraction * n_samples),
            replace=True,
            random_state=i,
        )

        # Fit the model
        try:
            model.fit(X_sample, y_sample)
        except Exception as e:
            print(f"Model failed to fit on subsample {i}: {e}")
            continue

        # Get feature importances or coefficients
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_).sum(axis=0)
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            raise ValueError(
                "Model must have 'coef_' or 'feature_importances_' attribute."
            )

        # Select features based on non-zero importance
        selected_features = features.columns[importance > 1e-6]

        # Update selection counts
        selection_counts[selected_features] += 1

    # Calculate selection frequency
    selection_frequency = selection_counts / n_subsamples

    return selection_frequency


def compute_stepwise_regression_with_scores(features, target, direction="both"):
    """

    Performs stepwise regression to select features.

    Applicability:
    - Suitable for:
        - Regression problems
    - Assumptions:
        - Linear relationships
        - Normally distributed residuals
    - Notes:
        - Iteratively adds or removes features based on statistical criteria (e.g., AIC).
        - Direction can be 'forward', 'backward', or 'both'.

    Useful for:
        - Identifying a subset of features that contribute significantly to the model

    Returns:
    - Series of feature importance scores based on cumulative AIC contribution.
    """

    initial_features = features.columns.tolist()
    feature_scores = {col: 0 for col in initial_features}
    if direction == "forward":
        current_features = []
    elif direction == "backward":
        current_features = initial_features.copy()
    else:
        current_features = []

    def calculate_aic(features_list):
        X = (
            sm.add_constant(features[features_list])
            if features_list
            else sm.add_constant(pd.DataFrame(index=features.index))
        )
        model = sm.OLS(target, X).fit()
        return model.aic

    base_aic = calculate_aic(current_features)
    changed = True

    while changed:
        changed = False
        if direction in ["forward", "both"]:
            remaining_features = list(set(initial_features) - set(current_features))
            best_aic = base_aic
            best_candidate = None
            for candidate in remaining_features:
                features_to_try = current_features + [candidate]
                new_aic = calculate_aic(features_to_try)
                aic_reduction = base_aic - new_aic
                if aic_reduction > 0 and new_aic < best_aic:
                    best_aic = new_aic
                    best_candidate = candidate
            if best_candidate:
                current_features.append(best_candidate)
                feature_scores[best_candidate] += (
                    base_aic - best_aic
                )  # Accumulate score
                base_aic = best_aic
                changed = True

        if direction in ["backward", "both"] and len(current_features) > 1:
            worst_aic = base_aic
            worst_candidate = None
            for candidate in current_features:
                features_to_try = [f for f in current_features if f != candidate]
                new_aic = calculate_aic(features_to_try)
                aic_reduction = base_aic - new_aic
                if aic_reduction > 0 and new_aic < worst_aic:
                    worst_aic = new_aic
                    worst_candidate = candidate
            if worst_candidate:
                current_features.remove(worst_candidate)
                feature_scores[worst_candidate] += (
                    base_aic - worst_aic
                )  # Accumulate score
                base_aic = worst_aic
                changed = True

    return pd.Series(feature_scores)


def compute_autoencoder_feature_importance(
    features, encoding_dim=None, max_iter=200, **kwargs
):
    """
    Performs feature selection using an Autoencoder neural network.

    Applicability:
    - Suitable for:
        - Unsupervised feature selection
        - High-dimensional data
        - Continuous features
    - Assumptions:
        - Features are continuous and scaled appropriately
    - Notes:
        - Autoencoders are neural networks trained to reconstruct their input.
        - The encoder captures essential features in a compressed representation.
        - Feature importance can be derived from the weights connecting input to the encoding layer.
        - Captures non-linear relationships.

    Useful for:
    - Unsupervised feature learning
    - Dimensionality reduction
    - Capturing complex patterns in data

    Parameters:
    - features (DataFrame): Feature matrix
    - encoding_dim (int): Number of neurons in the hidden encoding layer (default: half the number of features)
    - max_iter (int): Maximum number of iterations for training (default: 200)

    Returns:
    - feature_importances (Series): Importance scores for each feature based on encoder weights
    """
    from sklearn.base import clone

    n_features = features.shape[1]
    if encoding_dim is None:
        encoding_dim = n_features // 2  # Default encoding dimension

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Define Autoencoder model
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(encoding_dim, n_features),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
    )

    # Train autoencoder
    autoencoder.fit(X_scaled, X_scaled)

    # Extract encoder weights
    encoder_weights = autoencoder.coefs_[0]  # Weights from input to encoding layer

    # Compute feature importances
    importances = np.sum(np.abs(encoder_weights), axis=1)

    return pd.Series(importances, index=features.columns)


# =============================================================================
# RECURSIVE METHODS
# =============================================================================


def compute_rfe_eliminate_worst_cv(
    features,
    target,
    estimator=None,
    step=1,
    cv=5,
    scoring=None,
    min_features_to_select=1,
    random_state=None,
):
    """
    Performs Recursive Feature Elimination with Cross-Validation by eliminating the least important feature at each iteration.

    Applicability:
    - Suitable for:
        - Classification and regression problems
        - Estimators with 'feature_importances_' or 'coef_' attribute
    - Assumptions:
        - The estimator provides reliable feature importance measures
        - Features are independent (optional)

    Notes:
    - At each iteration, the least important feature is eliminated based on the estimator's feature importance.
    - Uses cross-validation to evaluate model performance at each step.
    - Returns a ranking where higher ranks indicate more important features.
    - Selects the optimal number of features based on cross-validation scores.

    Useful for:
    - Feature selection and ranking in datasets with many features
    - Understanding feature importance hierarchy with validation

    Parameters:
    - features: DataFrame of feature variables
    - target: Array-like target variable
    - estimator: Estimator object (default is RandomForestClassifier or RandomForestRegressor)
    - step: Number of features to remove at each iteration
    - cv: Cross-validation splitting strategy. Integer or CV splitter.
    - scoring: A string or a scorer callable object/function.
    - min_features_to_select: Minimum number of features to select
    - random_state: Controls the randomness of the estimator and CV splitter

    Returns:
    - feature_ranking: Series with features and their elimination ranking (higher rank indicates higher importance)
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    # Encode target if necessary
    y = target.copy()
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = features.copy()

    # Determine default estimator if not provided
    if estimator is None:
        # Choose estimator based on problem type
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            estimator = RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
            default_scoring = "neg_mean_squared_error"
        else:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=random_state
            )
            default_scoring = "accuracy"
    else:
        estimator = clone(estimator)
        default_scoring = (
            "neg_mean_squared_error"
            if hasattr(estimator, "predict_proba")
            else "accuracy"
        )

    # Use provided scoring or default
    if scoring is None:
        scoring = default_scoring

    # Cross-validation splitter
    if isinstance(cv, int):
        if hasattr(estimator, "predict_proba"):
            cv_splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    current_features = list(X.columns)
    n_features = len(current_features)
    eliminated_features = []
    ranking_dict = {}  # Key: feature name, Value: elimination rank
    elimination_round = 0
    cv_scores = []
    n_features_list = []
    rank = n_features  # Start ranking from the total number of features

    while len(current_features) >= min_features_to_select:
        elimination_round += 1

        # Evaluate estimator with cross-validation
        scores = cross_val_score(
            estimator,
            X[current_features],
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
        )
        mean_score = np.mean(scores)
        cv_scores.append(mean_score)
        n_features_list.append(len(current_features))

        # Fit estimator to compute feature importances
        estimator_clone = clone(estimator)
        estimator_clone.fit(X[current_features], y)

        # Get feature importances
        if hasattr(estimator_clone, "feature_importances_"):
            importances = estimator_clone.feature_importances_
        elif hasattr(estimator_clone, "coef_"):
            importances = np.abs(estimator_clone.coef_).flatten()
        else:
            raise ValueError(
                "Estimator must have 'feature_importances_' or 'coef_' attribute."
            )

        # Identify features to eliminate
        n_features_current = len(current_features)
        if n_features_current == min_features_to_select:
            # Assign ranks to remaining features and break
            for feature in current_features:
                ranking_dict[feature] = rank
                rank -= 1
            break

        n_features_to_remove = min(step, n_features_current - min_features_to_select)
        # Get indices of features sorted by importance (ascending)
        sorted_idx = np.argsort(importances)
        features_to_remove = [
            current_features[i] for i in sorted_idx[:n_features_to_remove]
        ]

        # Assign ranks to eliminated features
        for feature in features_to_remove:
            ranking_dict[feature] = rank
            rank -= 1
            eliminated_features.append(feature)

        # Remove features
        current_features = [f for f in current_features if f not in features_to_remove]

    # Create DataFrame for feature rankings
    feature_ranking = pd.DataFrame(
        {
            "Feature": X.columns,
            "Elimination_Rank": [
                ranking_dict.get(feature, None) for feature in X.columns
            ],
        }
    )

    # Sort features by elimination rank (higher rank means more important)
    feature_ranking.sort_values(by="Elimination_Rank", ascending=False, inplace=True)
    max_rank = feature_ranking["Elimination_Rank"].max()
    scores = max_rank - feature_ranking["Elimination_Rank"] + 1
    return pd.Series(scores.values, index=feature_ranking["Feature"])


def compute_rfe_eliminate_best_cv(
    features,
    target,
    estimator=None,
    step=1,
    cv=5,
    scoring=None,
    min_features_to_select=1,
    random_state=None,
):
    """
    Performs Reverse Recursive Feature Elimination with Cross-Validation by eliminating the most important feature at each iteration.

    Applicability:
    - Suitable for:
        - Classification and regression problems
        - Estimators with 'feature_importances_' or 'coef_' attribute
    - Assumptions:
        - The estimator provides reliable feature importance measures
        - Features are independent (optional)

    Notes:
    - At each iteration, the most important feature is eliminated based on the estimator's feature importance.
    - Uses cross-validation to evaluate model performance at each step.
    - Returns a ranking where higher ranks indicate more important features.
    - Selects the optimal number of features based on cross-validation scores.

    Useful for:
    - Assessing the impact of key features on model performance with validation
    - Identifying features the model relies on most heavily

    Parameters:
    - features: DataFrame of feature variables
    - target: Array-like target variable
    - estimator: Estimator object (default is RandomForestClassifier or RandomForestRegressor)
    - step: Number of features to remove at each iteration
    - cv: Cross-validation splitting strategy. Integer or CV splitter.
    - scoring: A string or a scorer callable object/function.
    - min_features_to_select: Minimum number of features to select
    - random_state: Controls the randomness of the estimator and CV splitter

    Returns:
    - feature_ranking: DataFrame with features and their elimination ranking (higher rank indicates higher importance)
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    # Encode target if necessary
    y = target.copy()
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = features.copy()

    # Determine default estimator if not provided
    if estimator is None:
        # Choose estimator based on problem type
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            estimator = RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
            default_scoring = "neg_mean_squared_error"
        else:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=random_state
            )
            default_scoring = "accuracy"
    else:
        estimator = clone(estimator)
        default_scoring = (
            "neg_mean_squared_error"
            if hasattr(estimator, "predict_proba")
            else "accuracy"
        )

    # Use provided scoring or default
    if scoring is None:
        scoring = default_scoring

    # Cross-validation splitter
    if isinstance(cv, int):
        if hasattr(estimator, "predict_proba"):
            cv_splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    current_features = list(X.columns)
    n_features = len(current_features)
    eliminated_features = []
    ranking_dict = {}  # Key: feature name, Value: elimination rank
    elimination_round = 0
    cv_scores = []
    n_features_list = []
    rank = 1  # Start ranking from 1

    while len(current_features) >= min_features_to_select:
        elimination_round += 1

        # Evaluate estimator with cross-validation
        scores = cross_val_score(
            estimator,
            X[current_features],
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
        )
        mean_score = np.mean(scores)
        cv_scores.append(mean_score)
        n_features_list.append(len(current_features))

        # Fit estimator to compute feature importances
        estimator_clone = clone(estimator)
        estimator_clone.fit(X[current_features], y)

        # Get feature importances
        if hasattr(estimator_clone, "feature_importances_"):
            importances = estimator_clone.feature_importances_
        elif hasattr(estimator_clone, "coef_"):
            importances = np.abs(estimator_clone.coef_).flatten()
        else:
            raise ValueError(
                "Estimator must have 'feature_importances_' or 'coef_' attribute."
            )

        # Identify features to eliminate
        n_features_current = len(current_features)
        if n_features_current == min_features_to_select:
            # Assign ranks to remaining features and break
            for feature in current_features:
                ranking_dict[feature] = rank
                rank += 1
            break

        n_features_to_remove = min(step, n_features_current - min_features_to_select)
        # Get indices of features sorted by importance (descending)
        sorted_idx = np.argsort(importances)[::-1]
        features_to_remove = [
            current_features[i] for i in sorted_idx[:n_features_to_remove]
        ]

        # Assign ranks to eliminated features
        for feature in features_to_remove:
            ranking_dict[feature] = rank
            rank += 1
            eliminated_features.append(feature)

        # Remove features
        current_features = [f for f in current_features if f not in features_to_remove]

    # Create DataFrame for feature rankings
    feature_ranking = pd.DataFrame(
        {
            "Feature": X.columns,
            "Elimination_Rank": [
                ranking_dict.get(feature, None) for feature in X.columns
            ],
        }
    )

    # Sort features by elimination rank (higher rank means more important)
    feature_ranking.sort_values(by="Elimination_Rank", ascending=False, inplace=True)
    max_rank = feature_ranking["Elimination_Rank"].max()
    scores = max_rank - feature_ranking["Elimination_Rank"] + 1
    return pd.Series(scores.values, index=feature_ranking["Feature"])


# =============================================================================
# EFFECT SIZE MEASURES
# =============================================================================


def compute_cohens_d(features, target):
    """
    Computes Cohen's d effect size between two classes for each feature.

    Applicability:
    - Suitable for:
        - Binary classification
        - Continuous features
        - Normally distributed data with equal variances
    - Measures standardized mean difference

    Notes:
    - Cohen's d indicates the magnitude of differences between two groups.
    - Rule of thumb: 0.2 (small), 0.5 (medium), 0.8 (large) effect

    Useful for:
    - Understanding the practical significance of feature differences
    - Feature selection when effect size is important

    Returns:
    - Series of Cohen's d values for each feature.
    """
    cohens_d_scores = pd.Series(index=features.columns, dtype=float)
    classes = np.unique(target)

    if len(classes) != 2:
        raise ValueError("Cohen's d is only applicable for binary classification.")

    for column in features.columns:
        group1 = features[column][target == classes[0]]
        group2 = features[column][target == classes[1]]

        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Avoid division by zero
        if pooled_std == 0:
            d = 0.0
        else:
            d = np.abs(np.mean(group1) - np.mean(group2)) / pooled_std
        cohens_d_scores[column] = d

    return cohens_d_scores


def compute_odds_ratio(features, target, bins=10):
    """
    Computes maximum odds ratio between binned feature values and binary target.

    Applicability:
    - Suitable for:
        - Binary classification
        - Binary or binned continuous features
    - No distribution assumptions
    - Interpretable effect size for categorical data

    Notes:
    - Odds ratios are typically used with categorical data.
    - Binning continuous features can lead to loss of information.
    - Interpretation requires caution.

    Useful for:
    - Situations where features are categorical
    - Interpreting the strength of association between feature levels and target

    Returns:
    - Series of maximum absolute log odds ratios for each feature.
    """
    odds_ratios = pd.Series(index=features.columns, dtype=float)

    classes = np.unique(target)
    if len(classes) != 2:
        raise ValueError("Odds ratio is only applicable for binary classification.")

    for column in features.columns:
        # Bin continuous features
        if features[column].nunique() > bins:
            feature_binned = pd.qcut(
                features[column], bins, labels=False, duplicates="drop"
            )
        else:
            feature_binned = features[column]

        # Create contingency table
        contingency = pd.crosstab(feature_binned, target)

        # Calculate odds ratio for each bin
        odds_ratios_bin = []
        for i in contingency.index:
            a = (
                contingency.loc[i, classes[1]]
                if classes[1] in contingency.columns
                else 0
            )
            b = (
                contingency.loc[i, classes[0]]
                if classes[0] in contingency.columns
                else 0
            )
            c = (
                contingency[classes[1]].sum() - a
                if classes[1] in contingency.columns
                else 0
            )
            d = (
                contingency[classes[0]].sum() - b
                if classes[0] in contingency.columns
                else 0
            )

            # Add small constant to avoid division by zero
            odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            odds_ratios_bin.append(np.abs(np.log(odds_ratio)))

        odds_ratios[column] = np.max(odds_ratios_bin)

    return odds_ratios


def compute_cliffs_delta(features, target):
    """
    Computes Cliff's Delta effect size between two classes for each feature.

    Applicability:
    - Suitable for:
        - Binary classification
        - Non-parametric
        - Ordinal or continuous features
    - No distribution assumptions
    - Robust effect size measure

    Notes:
    - Values range between -1 and 1, with 0 indicating no difference.
    - Rule of thumb: 0.147 (small), 0.33 (medium), 0.474 (large) effect.
    - Good for ordinal data.

    Useful for:
    - Understanding differences in feature distributions between classes
    - Feature selection in non-parametric contexts

    Returns:
    - Series of absolute Cliff's Delta values for each feature.
    """
    cliffs_delta_scores = pd.Series(index=features.columns, dtype=float)
    classes = np.unique(target)

    if len(classes) != 2:
        raise ValueError("Cliff's Delta is only applicable for binary classification.")

    for column in features.columns:
        group1 = features[column][target == classes[0]].values
        group2 = features[column][target == classes[1]].values

        n1, n2 = len(group1), len(group2)

        # For large datasets, use sampling to speed up computation
        max_comparisons = 100000  # Adjust as needed
        if n1 * n2 > max_comparisons:
            sample_size = int(np.sqrt(max_comparisons))
            group1 = np.random.choice(group1, sample_size, replace=False)
            group2 = np.random.choice(group2, sample_size, replace=False)
            n1 = n2 = sample_size

        # Calculate Cliff's Delta
        dominance = 0
        for x in group1:
            dominance += np.sum(group2 < x) - np.sum(group2 > x)
        cliffs_delta = dominance / (n1 * n2)
        cliffs_delta_scores[column] = np.abs(cliffs_delta)

    return cliffs_delta_scores


def compute_cramers_v(features, target, bins=10):
    """
    Computes Cramér's V statistic for association between each feature and the target.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Categorical or binned continuous features
    - Effect size measure for chi-square

    Notes:
    - Values range from 0 to 1, with higher values indicating stronger association.
    - For continuous features, binning is necessary.
    - Rule of thumb: 0.1 (small), 0.3 (medium), 0.5 (large) effect

    Useful for:
    - Assessing association between categorical features and target
    - Feature selection in models using categorical data

    Returns:
    - Series of Cramér's V values for each feature.
    """
    cramers_v_scores = pd.Series(index=features.columns, dtype=float)

    for column in features.columns:
        # Bin continuous features
        if features[column].nunique() > bins:
            feature_binned = pd.qcut(
                features[column], bins, labels=False, duplicates="drop"
            )
        else:
            feature_binned = features[column]

        contingency = pd.crosstab(feature_binned, target)
        chi2, p, dof, expected = chi2_contingency(contingency)

        n = contingency.sum().sum()
        phi2 = chi2 / n
        r, k = contingency.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        cramers_v_scores[column] = cramers_v

    return cramers_v_scores


def compute_point_biserial(features, target):
    """
    Computes point-biserial correlation between each feature and a binary target.

    Applicability:
    - Suitable for:
        - Binary classification
        - Continuous features
    - Special case of Pearson correlation
    - Effect size measure for binary target

    Notes:
    - Measures the relationship between a binary variable and a continuous variable.
    - Assumes the continuous variable is normally distributed within groups.

    Useful for:
    - Linear models where linearity assumption holds
    - Understanding linear relationships in binary classification

    Returns:
    - Series of absolute point-biserial correlation coefficients for each feature.
    """
    pb_scores = pd.Series(index=features.columns, dtype=float)

    if len(np.unique(target)) != 2:
        raise ValueError(
            "Point-biserial correlation is only applicable for binary classification."
        )

    for column in features.columns:
        correlation, _ = pointbiserialr(target, features[column])
        pb_scores[column] = abs(correlation)

    return pb_scores


# =============================================================================
# DISTANCE-BASED MEASURE
# =============================================================================


def compute_relieff(features, target, n_neighbors=100):
    """
    Computes feature importance using the ReliefF algorithm.

    Applicability:
    - Suitable for:
        - Binary and multi-class classification
        - Continuous and categorical features
    - Assumptions:
        - None; non-parametric
    - Handles:
        - Non-linear relationships
        - Feature interactions

    Notes:
    - ReliefF estimates feature importance based on the difference between nearest neighbors.
    - Computationally efficient for medium-sized datasets.
    - Requires the 'ReliefF' package (pip install ReliefF).

    Useful for:
    - Situations where feature interactions are important
    - Any classification model

    Returns:
    - Series of ReliefF scores for each feature.
    """
    fs = ReliefF(n_neighbors=n_neighbors)
    fs.fit(features.values, target.values)
    n_features = len(features.columns)
    scores = np.zeros(n_features)
    for rank, feature_idx in enumerate(fs.top_features):
        scores[feature_idx] = n_features - rank
    
    return pd.Series(data=scores, index=features.columns)


def compute_cfd(features, target):
    """
    Computes Correlation-based Feature Distance (CFD) scores.

    Applicability:
    - Suitable for:
        - Classification and regression problems
        - Continuous and discrete features
        - Linear and nonlinear relationships
    - Assumptions:
        - None (non-parametric)

    Notes:
    - Uses distance correlation instead of traditional correlation
    - Captures both linear and nonlinear relationships
    - Computationally more intensive than traditional correlation

    Returns:
    - Series of CFD scores for each feature
    """
    import dcor

    scores = {}
    target_array = target.values if isinstance(target, pd.Series) else target

    for column in features.columns:
        feature_array = features[column].values
        score = dcor.distance_correlation(feature_array, target_array)
        scores[column] = score

    cfd_score = pd.Series(scores)
    return cfd_score


def compute_laplacian_score(features, **kwargs):
    """
    Computes the Laplacian Score for feature selection.

    Applicability:
    - Suitable for:
        - Unsupervised learning tasks (e.g., clustering)
        - Continuous features
    - Assumptions:
        - None
    - Notes:
        - Evaluates features based on locality-preserving power.
        - A lower Laplacian Score indicates a more important feature but here we changed to higher better.

    Useful for:
    - Identifying features that respect the intrinsic geometry of data
    - Unsupervised feature selection

    Returns:
    - Series of Laplacian Scores for each feature.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_samples = X.shape[0]

    # Build adjacency graph using k-nearest neighbors
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    W = nbrs.kneighbors_graph(X, mode="connectivity").toarray()

    D = np.diag(W.sum(axis=1))
    L = D - W  # Laplacian matrix

    f_scores = {}
    for i, column in enumerate(features.columns):
        f = X[:, i]
        numerator = np.dot(f.T, np.dot(L, f))
        denominator = (
            np.dot(f.T, np.dot(D, f)) + 1e-12
        )  # Add small constant to avoid division by zero
        score = numerator / denominator
        f_scores[column] = score

    laplacian_scores = pd.Series(f_scores)
    max_score = laplacian_scores.max()
    laplacian_scores = max_score - laplacian_scores
    return laplacian_scores


def compute_vdm(features, target):
    """
    Computes Value Difference Metric (VDM) scores for feature importance.

    Applicability:
    - Suitable for:
        - Any classification problems (binary, multiclass, multilabel)
        - Categorical features (can handle numeric features but works best with categorical)
        - Nominal and ordinal data
    - Assumptions:
        - None (non-parametric)

    Notes:
    - Measures importance based on class-conditional probability differences
    - Higher scores indicate features with values that better discriminate between classes
    - Handles missing values by treating them as a separate category
    - More robust to noisy categorical data compared to chi-square or mutual information
    - Can capture non-linear relationships

    Returns:
    - Series of VDM scores for each feature
    """
    scores = {}

    # Convert target to array if it's a pandas Series
    target_array = target.values if isinstance(target, pd.Series) else target

    for column in features.columns:
        feature_array = features[column].values

        # Get unique values for feature and target
        feature_values = np.unique(feature_array)
        target_values = np.unique(target_array)

        total_vdm = 0
        value_counts = 0

        # Calculate VDM for each pair of feature values
        for i in range(len(feature_values)):
            for j in range(i + 1, len(feature_values)):
                val1, val2 = feature_values[i], feature_values[j]
                vdm = 0

                # Calculate conditional probabilities for each class
                for c in target_values:
                    # P(class|value1)
                    p1 = np.mean((feature_array == val1) & (target_array == c))
                    # P(class|value2)
                    p2 = np.mean((feature_array == val2) & (target_array == c))

                    # Add to VDM using Manhattan distance (n=1)
                    vdm += abs(p1 - p2)

                total_vdm += vdm
                value_counts += 1

        # Average VDM across all value pairs
        scores[column] = total_vdm / value_counts if value_counts > 0 else 0

    vdm_scores = pd.Series(scores)
    return vdm_scores


# =============================================================================
# ENSEMBLE RANKING METHOD
# =============================================================================


def rank_features(features, target, methods=None, weights=None):
    """
    Comprehensive feature ranking combining multiple methods.

    Parameters:
    - features: DataFrame of feature variables.
    - target: Array-like target variable.
    - methods: List of method names to use for ranking.
    - weights: Dictionary of weights for each method.

    Available methods:
    - 'spearman': Spearman correlation (non-parametric, any classification)
    - 'kendall': Kendall's Tau (non-parametric, any classification)
    - 'pearson': Pearson correlation (regression, continuous target)
    - 'mwu': Mann-Whitney U test (binary classification, non-parametric)
    - 'kruskal': Kruskal-Wallis H test (multi-class classification, non-parametric)
    - 'ks': Kolmogorov-Smirnov test (binary classification, non-parametric)
    - 'chi2': Chi-square test (any classification, categorical features)
    - 'mi': Mutual Information (any classification, non-parametric)
    - 'rf': Random Forest importance (any classification, non-parametric)
    - 'lasso': Lasso coefficients (binary and multi-class classification, linear)
    - 'permutation': Permutation importance (any classification)
    - 'rfe_worst': Recursive Feature Elimination worst feature (any classification, regression)
    - 'rfe_best': Recursive Feature Elimination best feature (any classification, regression)
    - 'anova': ANOVA F-test (normal distribution, linear)
    - 'cohens_d': Cohen's d effect size (binary classification, normal distribution)
    - 'odds_ratio': Odds ratio (binary classification, categorical features)
    - 'cliffs_delta': Cliff's delta (binary classification, non-parametric)
    - 'cramers_v': Cramér's V (any classification, categorical features)
    - 'point_biserial': Point-biserial correlation (binary classification)
    - 'vdm': Value Difference Metric (any classification, categorical features preferred, non-parametric)
    - 'fishers_score: ANOVA F-test for binary classification (normal distribution, linear)
    - 'relief': Relief based festure importance (non-parametric, any classification)
    - 'mic': MIC (non-parametric, any classification)
    - 'laplacian': Laplacian score (any classification, regression, clustering, non-parametric)
    - 't_test': T-est (normal distribution, binary classification)
    - 'lda': LDA coefficient (normal distribution, any classification)
    - 'stepwise_regression': Step wise regression importance (regression)
    - 'pls': PLS importance (non-parametric, regression)
    - 'stability': Feature selection based (any classification, regression)
    - 'autoencoder_feature_selection': Unsupervised selection (regression, any classification)
    - 'cfd': correlation based feature distance (any classification and regression, non-parametric)
    - 'iv_woe': Information Value (IV) and Weight of Evidence (binary classificaion)
    - 'fishers_exact': Fisher's exact test (binary classification, binary feature)
    - 'phi_coefficient': Phi coefficient (binary classification, binary feature)
    - 'pmi': Pointwise Mutual Information (any classificaion, categorical feature)
    - 'anova_categorical': ANOVA categorical (regression, categorical feature, target normal distribution)
    - 'likelihood_ratio_test': Likelihood ratio test (binary classification, categorical features)
    - 'eta_coefficient': Eta coefficient (regression, categorical feature)
    - 'partial_correlation': Pearson correlation based (regression, continuous target)
    - 'lars': Lars importance (regression)


    Notes:
    - Ensure that the methods chosen are appropriate for your specific problem and data.
    - Weights can be used to emphasize certain methods over others in the final ranking.
    - Features should be preprocessed as required by each method (e.g., scaling for Lasso).

    Returns:
    - DataFrame with rankings from each method and the mean score.

    Example:
    ```
    rankings = rank_features(features, target, methods=['spearman', 'mi', 'rf'], weights={'spearman': 1.0, 'mi': 0.5, 'rf': 1.5})
    ```
    """

    available_methods = {
        "spearman": compute_spearman_correlation,
        "kendall": compute_kendall_tau,
        "pearson": compute_pearson_correlation,
        "mwu": compute_mwu_test,
        "kruskal": compute_kruskal_wallis,
        "ks": compute_ks_test,
        "chi2": compute_chi_square,
        "mi": compute_mutual_information,
        "rf": compute_rf_importance,
        "lasso": compute_lasso_importance,
        "permutation": compute_permutation_importance,
        "rfe_worst": compute_rfe_eliminate_worst_cv,
        "rfe_best": compute_rfe_eliminate_best_cv,
        "cohens_d": compute_cohens_d,
        "odds_ratio": compute_odds_ratio,
        "cramers_v": compute_cramers_v,
        "relief": compute_relieff,
        "mic": compute_mic,
        "t_test": compute_t_test,
        "partial_correlation": compute_partial_correlation,
        "lars": compute_lars_coefficients,
        "iv_woe": compute_iv_woe,
        "pmi": compute_pmi,
        "laplacian": compute_laplacian_score,
        "stability": compute_stability_selection,
        "autoencoder_feature_selection": compute_autoencoder_feature_importance,
        "anova": compute_anova_f_test,
        "lda": compute_lda,
        "pls": compute_pls_feature_importance,
        "stepwise_regression": compute_stepwise_regression_with_scores,
        "point_biserial": compute_point_biserial,
        "vdm": compute_vdm,
        "fishers_score": compute_fisher_score,
        "cliffs_delta": compute_cliffs_delta,
        "cfd": compute_cfd,
        "fishers_exact": compute_fishers_exact,
        "phi_coefficient": compute_phi_coefficient,
        "anova_categorical": compute_anova_categorical,
        "likelihood_ratio_test": compute_likelihood_ratio,
        "eta_coefficient": compute_eta_coefficient,
    }

    if methods is None:
        methods = available_methods.keys()  # Default methods

    if weights is None:
        weights = {method: 1.0 for method in methods}

    rankings = pd.DataFrame()

    # Compute scores for selected methods
    for method in methods:
        if method in available_methods:
            # Check if the method requires the target variable
            func = available_methods[method]
            scores = func(features=features, target=target)
            # Normalize scores to [0,1]
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                scores_normalized = (scores - min_score) / (max_score - min_score)
            else:
                scores_normalized = scores - scores
            rankings[method] = scores_normalized * weights.get(method, 1.0)
        else:
            raise ValueError(f"Method '{method}' is not recognized or not implemented.")

    # Compute weighted mean score
    total_weight = sum(weights.get(method, 1.0) for method in methods)
    rankings["average_score"] = rankings.sum(axis=1) / total_weight

    # Compute ranks for each method
    for method in methods:
        rankings[f"{method}_rank"] = rankings[method].rank(
            ascending=False, method="average"
        )

        # Compute weighted average rank
        rank_columns = [f"{method}_rank" for method in methods]
    method_weights = [weights.get(method, 1.0) for method in methods]
    rankings["average_rank"] = rankings[rank_columns].multiply(
        method_weights, axis=1
    ).sum(axis=1) / sum(method_weights)

    # Sort features by average rank ascending (since lower rank is better)
    rankings = rankings.sort_values("average_rank")

    # Rearrange columns as desired
    score_columns = methods
    rankings = rankings[
        score_columns + ["average_score"] + rank_columns + ["average_rank"]
    ]

    return rankings


"""
Mean Decrease Accuracy (MDA) Feature Ranking
Is MDA the same as Permutation Importance?
Similarities:
Both MDA and permutation importance involve measuring the change in model performance when a feature's values are permuted.
They assess feature importance based on the impact on the model's predictive accuracy.
Differences:
Context of Use:
MDA is commonly associated with Random Forests and is computed during the model training process.
Permutation Importance is a more general method and can be applied to any trained model (model-agnostic).
Implementation Details:
In MDA, the decrease in accuracy is computed internally during the Random Forest's out-of-bag (OOB) error calculation for each tree.
Permutation Importance can be computed post-training by permuting features and measuring the impact on a chosen performance metric.
Conclusion:

While MDA and permutation importance are conceptually similar, MDA is traditionally associated with Random Forests and uses the OOB error for calculation.
Permutation Importance can be applied to any model and is considered model-agnostic.
"""

"""
Chi-Squared Automatic Interaction Detection (CHAID)
Explanation:

Purpose: A decision tree-based method that uses the chi-squared test to determine splits.
Applicability:
Suitable for categorical features.
Handles nominal and ordinal features.
Assumptions:
Assumes independence between observations.
Code Implementation:

Note: Implementing CHAID is complex and requires specialized libraries (e.g., CHAID in Python is limited). Due to its complexity, I will provide a high-level explanation rather than full code.
"""

"""
Association Rules Mining
Explanation:

Purpose: Identifies interesting relationships (associations) between variables in large datasets.
Applicability:
Suitable for categorical features.
Commonly used in market basket analysis but can be adapted for feature selection.
Assumptions:
Data should be in transactional format.
Code Implementation:

Note: Since association rules mining is a broad topic and involves transforming data into a suitable format (e.g., one-hot encoded transactions), implementing it here would be extensive.
"""
