"""
Fetch reviews data from BigQuery and generate the CSV files used for these exercises.

```sh
python -m admin.fetch_data
```
"""

from pathlib import Path

import pandas as pd
from google.cloud.bigquery import Client

from recommender_systems.features import (
    CustomerFeatures,
    ProductFeatures,
    ReviewFeatures,
)
from recommender_systems.splits import Splits

SPLIT = "split"

QUERY = f"""WITH
    reviews AS (
    SELECT
        -- Reviews can be edited, or have their state changed, hence the need for these ANY_VALUE functions
        ANY_VALUE(customer_id) AS customer_id_pii,
        ANY_VALUE(tsin_id
        HAVING
            MAX updated_at) AS tsin,
        uuid AS {ReviewFeatures.ID},
        ANY_VALUE(rating
        HAVING
            MAX updated_at) AS {ReviewFeatures.RATING},
        ANY_VALUE(body
        HAVING
            MAX updated_at) AS {ReviewFeatures.TEXT},
        MAX(updated_at) AS {ReviewFeatures.TIMESTAMP},
    FROM
        `tal-production-ml.automated_review_moderation.reviews`
    WHERE
        moderation_state_id = 0 -- ACCEPTED
        AND DATE(updated_at) BETWEEN "2025-01-01" AND "2025-06-30" -- RECENT
    GROUP BY
        uuid ),
    customers AS (
    SELECT
        customer_id_pii,
        GENERATE_UUID() AS {CustomerFeatures.ID},
    FROM (
        SELECT
            DISTINCT customer_id_pii
        FROM
            reviews )),
    products AS (
    SELECT
        variant.id AS tsin,
        ANY_VALUE(CONCAT('PLID', productline.id)
        HAVING
            MAX insert_timestamp) AS {ProductFeatures.ID},
        ANY_VALUE(productline.title
        HAVING
            MAX insert_timestamp) AS {ProductFeatures.TITLE},
        ANY_VALUE(productline.attributes.brand
        HAVING
            MAX insert_timestamp) AS {ProductFeatures.BRAND},
        ANY_VALUE(extended.hierarchy.merchandising.forest.names[SAFE_OFFSET(0)]
        HAVING
            MAX insert_timestamp) AS {ProductFeatures.DEPARTMENT},
    FROM
        `tal-production-ml.search_indexing.productline_documents`,
        UNNEST(productline.variants) AS variant
    GROUP BY
        -- BUG: TSINs are globally unique, but this table has some TSINs associated with multiple PLIDs
        tsin
    HAVING
        product_department IN ('Home & Kitchen',
            'Garden, Pool & Patio',
            'Health',
            'Beauty',
            'Computers & Tablets',
            'Office & Stationery',
            'Toys',
            'Fashion',
            'Sport',
            'Cellphones & Wearables'))
SELECT
    reviews.* EXCEPT(customer_id_pii,
        tsin),
    customers.* EXCEPT(customer_id_pii),
    products.* EXCEPT(tsin),
    -- stratified split based on customer_id and timestamp
    -- 80% train, 10% validation, 10% test
    CASE MOD(ROW_NUMBER() OVER (PARTITION BY {CustomerFeatures.ID} ORDER BY {ReviewFeatures.TIMESTAMP}), 10)
        WHEN 8 THEN "{Splits.VALIDATION}"
        WHEN 9 THEN "{Splits.TEST}"
        ELSE "{Splits.TRAIN}"
END
    AS {SPLIT},
FROM
    reviews
JOIN
    customers
USING
    (customer_id_pii)
JOIN
    products
USING
    (tsin)
QUALIFY
    -- Only consider customers who have left at least 10 reviews
    COUNT(*) OVER (PARTITION BY {CustomerFeatures.ID} ) >= 10
    -- Only consider the most recent 20 reviews per customer
    AND ROW_NUMBER() OVER (PARTITION BY {CustomerFeatures.ID} ORDER BY {ReviewFeatures.TIMESTAMP} DESC ) <= 20
ORDER BY
    RAND()
"""


if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    # All reviews
    reviews_df: pd.DataFrame = Client().query(QUERY).to_dataframe()

    # Split reviews into train, validation, and test sets
    print("🤓 TFRS Datasets")
    reviews_path = data_path / "reviews"
    reviews_path.mkdir(parents=True, exist_ok=True)
    for split in [Splits.TRAIN, Splits.VALIDATION, Splits.TEST]:
        print(f"🧐 {split} Split")
        split_df = reviews_df[reviews_df[SPLIT] == split].drop(columns=[SPLIT])
        print(split_df.head())
        print(f"Split: {split}, Count: {len(split_df)}")
        split_df.to_csv(reviews_path / f"{split}.csv", index=False)

    # Drop the "split" column, as it's only relevant for the TFRS datasets
    reviews_df.drop(columns=[SPLIT], inplace=True)

    # All customers present in all reviews
    print("🤓 Customers")
    customers_df = reviews_df.groupby(CustomerFeatures.ID, as_index=False).first()[
        [CustomerFeatures.ID]
    ]
    print(customers_df.head())
    print(f"Customer Count: {len(customers_df)}")
    customers_df.to_csv(data_path / "customers.csv", index=False)

    # All products present in all reviews
    print("🤓 Products")
    products_df = reviews_df.groupby(ProductFeatures.ID, as_index=False).first()[
        [
            ProductFeatures.ID,
            ProductFeatures.TITLE,
            ProductFeatures.BRAND,
            ProductFeatures.DEPARTMENT,
        ]
    ]
    print(products_df.head())
    print(f"Product Count: {len(products_df)}")
    products_df.to_csv(data_path / "products.csv", index=False)

    # All reviews (ignoring additional product features)
    print("🤓 Reviews")
    reviews_df = reviews_df.drop(
        columns=[
            ProductFeatures.TITLE,
            ProductFeatures.BRAND,
            ProductFeatures.DEPARTMENT,
        ]
    )
    print(reviews_df.head())
    print(f"Review Count: {len(reviews_df)}")
    reviews_df.to_csv(data_path / "reviews.csv", index=False)
