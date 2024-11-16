def suggest_methods_for_model(model_type, data_characteristics, method_characteristics):
    """
    Suggests appropriate feature selection methods based on model type and data characteristics.

    Parameters:
    -----------
    model_type : str
        The type of model you plan to use. Example values:
        - 'logistic_regression'
        - 'random_forest'
        - 'svm'
        - 'neural_networks'
        - 'decision_trees'
        - 'naive_bayes'
        - 'knn'
        - 'linear_regression'
    data_characteristics : dict
        Dictionary containing characteristics of your data. Example keys:
        - 'data_type_continuous': True/False
        - 'data_type_nominal': True/False
        - 'sample_size_small': True/False
        - 'assumption_normal_distribution': True/False
        - 'handles_nonlinear': True/False
        - 'target_type_binary': True/False
        - 'multiclass': True/False
    method_characteristics : DataFrame
        The DataFrame returned by get_method_characteristics() function.

    Returns:
    --------
    List of dictionaries containing recommended methods and their suitability scores.

    Example:
    --------
    recommended_methods = suggest_methods_for_model('random_forest', data_characteristics, method_characteristics)
    """

    suitable_methods = []

    for method, props in method_characteristics.iterrows():
        suitability_score = _calculate_suitability(
            model_type, data_characteristics, props
        )
        if suitability_score > 0:
            suitable_methods.append(
                {
                    "method": method,
                    "suitability_score": suitability_score,
                    "notes": props["notes"],
                }
            )

    # Sort methods by suitability score in descending order
    suitable_methods = sorted(
        suitable_methods, key=lambda x: x["suitability_score"], reverse=True
    )

    return suitable_methods


def _calculate_suitability(model_type, data_chars, method_props):
    """
    Helper function to calculate a suitability score for a method based on model type and data characteristics.
    Assigns points for each matching characteristic.

    Parameters:
    -----------
    model_type : str
        The type of model you plan to use.
    data_chars : dict
        Dictionary containing characteristics of your data.
    method_props : Series
        A pandas Series representing the characteristics of a method from the method_characteristics DataFrame.

    Returns:
    --------
    int
        The suitability score for the method.
    """
    score = 0

    # Check if the method works with the model
    model_key = f"works_with_{model_type}"
    if model_key in method_props and method_props[model_key]:
        score += 3  # Higher weight if method is known to work well with the model

    # Check if the method is applicable to the target type
    for target_type in [
        "target_type_continuous",
        "target_type_binary",
        "target_type_multiclass",
    ]:
        if data_chars.get(target_type, False) and method_props.get(target_type, False):
            score += 2

    # Check if the method works with the data types
    for data_type in [
        "data_type_continuous",
        "data_type_ordinal",
        "data_type_nominal",
        "data_type_binary",
    ]:
        if data_chars.get(data_type, False) and method_props.get(data_type, False):
            score += 1

    # Check sample size suitability
    for sample_size in ["sample_size_small", "sample_size_medium", "sample_size_large"]:
        if data_chars.get(sample_size, False) and method_props.get(sample_size, False):
            score += 1

    # Check if method handles non-linear relationships
    if data_chars.get("handles_nonlinear", False) and method_props.get(
        "handles_nonlinear", False
    ):
        score += 1

    # Deduct points if the method requires assumptions that are not met
    for assumption in [
        "assumption_normal_distribution",
        "assumption_equal_variance",
        "assumption_linear_relationship",
    ]:
        if method_props.get(assumption, False) and not data_chars.get(assumption, True):
            score -= 1

    # Check if method requires transformation
    if method_props.get("transformation_required", False):
        score -= 1  # Penalize methods requiring transformation

    # Handle multiclass support
    if data_chars.get("multiclass", False) and not method_props.get(
        "multiclass", False
    ):
        score -= 2  # Penalize if method does not support multiclass when needed

    return score
