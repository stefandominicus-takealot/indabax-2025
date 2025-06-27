class CustomerFeatures:
    ID = "customer_id"


class ProductFeatures:
    ID = "product_id"
    TITLE = "product_title"
    BRAND = "product_brand"
    DEPARTMENT = "product_department"


class ReviewFeatures:
    ID = "review_id"
    RATING = "review_rating"
    TEXT = "review_text"
    TIMESTAMP = "review_timestamp"


class MetaFeatures:
    RETRIEVAL_SAMPLE_WEIGHT = "retrieval_sample_weight"


def xf(feature_name: str):
    """
    Appends a suffix '_xf' to the given feature name, indicating that this is a
    transformed feature.
    """
    return f"{feature_name}_xf"
