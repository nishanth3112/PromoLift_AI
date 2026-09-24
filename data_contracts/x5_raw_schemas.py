"""Pandera schema contracts for the raw X5 RetailHero dataset files.

Each schema encodes two different kinds of facts:

- dtype and nullability: discovered from the real data (see
  ``notebooks/01_data_audit.ipynb``), not assumed.
- value-range / domain checks: sensible real-world constraints, used to
  *flag* data-quality problems rather than to describe whatever values
  happen to be present. For example, ``clients.csv`` contains 313 rows with
  an age outside [0, 120] (including negative ages and an age of 1901) --
  the schema intentionally rejects those rather than widening the range to
  fit them.
"""

from __future__ import annotations

import pandera.polars as pa
from pandera.typing.polars import Series

from promolift.data.loader import Dataset


class ClientsSchema(pa.DataFrameModel):
    client_id: Series[str]
    first_issue_date: Series[str]
    first_redeem_date: Series[str] = pa.Field(nullable=True)
    age: Series[int] = pa.Field(ge=0, le=120)
    gender: Series[str] = pa.Field(isin=["F", "M", "U"])

    class Config:
        strict = True
        coerce = False


class ProductsSchema(pa.DataFrameModel):
    product_id: Series[str]
    level_1: Series[str] = pa.Field(nullable=True)
    level_2: Series[str] = pa.Field(nullable=True)
    level_3: Series[str] = pa.Field(nullable=True)
    level_4: Series[str] = pa.Field(nullable=True)
    segment_id: Series[float] = pa.Field(nullable=True, ge=0)
    brand_id: Series[str] = pa.Field(nullable=True)
    vendor_id: Series[str] = pa.Field(nullable=True)
    netto: Series[float] = pa.Field(nullable=True, ge=0)
    is_own_trademark: Series[int] = pa.Field(isin=[0, 1])
    is_alcohol: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = False


class PurchasesSchema(pa.DataFrameModel):
    client_id: Series[str]
    transaction_id: Series[str]
    # Kept as a raw string here; parsing/validity is checked by the
    # referential_integrity date-integrity check, not this schema.
    transaction_datetime: Series[str]
    regular_points_received: Series[float] = pa.Field(ge=0)
    express_points_received: Series[float] = pa.Field(ge=0)
    regular_points_spent: Series[float] = pa.Field(le=0)
    express_points_spent: Series[float] = pa.Field(le=0)
    purchase_sum: Series[float] = pa.Field(ge=0)
    store_id: Series[str]
    product_id: Series[str]
    product_quantity: Series[float] = pa.Field(ge=0)
    trn_sum_from_iss: Series[float] = pa.Field(ge=0)
    # Inferred as a string column (93.35% null) rather than numeric -- a
    # known quirk of the raw file, left as-is until investigated further.
    trn_sum_from_red: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = False


class UpliftTrainSchema(pa.DataFrameModel):
    client_id: Series[str]
    treatment_flg: Series[int] = pa.Field(isin=[0, 1])
    target: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = False


class UpliftTestSchema(pa.DataFrameModel):
    client_id: Series[str]

    class Config:
        strict = True
        coerce = False


class UpliftSampleSubmissionSchema(pa.DataFrameModel):
    client_id: Series[str]
    uplift: Series[float] = pa.Field(ge=0, le=1)

    class Config:
        strict = True
        coerce = False


SCHEMA_REGISTRY: dict[Dataset, type[pa.DataFrameModel]] = {
    Dataset.CLIENTS: ClientsSchema,
    Dataset.PRODUCTS: ProductsSchema,
    Dataset.PURCHASES: PurchasesSchema,
    Dataset.UPLIFT_TRAIN: UpliftTrainSchema,
    Dataset.UPLIFT_TEST: UpliftTestSchema,
    Dataset.UPLIFT_SAMPLE_SUBMISSION: UpliftSampleSubmissionSchema,
}
