import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew

# ── Imports only needed for commented-out classes (kept for reference) ─────────
# import statsmodels.api as sm
# from gensim.models import Word2Vec


# =============================================================================
# ACTIVE — Used in the loan default prediction pipeline
# =============================================================================

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Cleaning Step: Automatically applies Yeo-Johnson power transform to any
    numeric column whose absolute skewness exceeds `threshold`.
    Skips non-numeric columns entirely.
    """
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # PROTECTION: Only look at columns that are actually numeric
        # This prevents the step from ever seeing a categorical string
        numeric_df = X.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return self

        # Only calculate skewness for numeric columns
        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
            
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature Engineering — Final Selection Step:
      1. Drops columns with missing rate above `missing_threshold`
      2. Drops categorical columns with cardinality above `cardinality_threshold`
      3. Drops numeric columns whose absolute correlation with target is
         below `corr_threshold`
    All logic is learned in fit() and applied in transform() — no leakage.
    """
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 1. Missing Values Filter
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        # 2. High Cardinality Filter (only for categorical/object columns)
        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        # 3. Correlation Filter (only for numeric columns)
        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = y
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Cleaning Step — Drop-in replacement for SimpleImputer that preserves
    DataFrame column names all the way through the pipeline.
    SimpleImputer converts output to a numpy array, which strips column names
    and causes SHAP plots to show 'feature_0', 'feature_1' etc. instead of
    real names like 'int_rate', 'dti', 'fico_mid'.

    Strategy options: 'median' (default), 'mean', 'most_frequent', 'constant'
    For numeric columns uses the chosen strategy.
    For object/categorical columns always uses 'most_frequent'.
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.fill_values_ = {}
        for col in X.columns:
            if X[col].dtype == object:
                self.fill_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 'missing'
            else:
                if self.strategy == 'median':
                    self.fill_values_[col] = X[col].median()
                elif self.strategy == 'mean':
                    self.fill_values_[col] = X[col].mean()
                elif self.strategy == 'most_frequent':
                    self.fill_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 0
                else:
                    self.fill_values_[col] = 0
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in X.columns:
            if col in self.fill_values_:
                X[col] = X[col].fillna(self.fill_values_[col])
        return X  # Returns a DataFrame — column names are preserved!


class RecodeCategoricals(BaseEstimator, TransformerMixin):
    """
    Cleaning Steps 1 & 2:
      - term:       ' 36 months' -> 36  (extract numeric digits)
      - emp_length: '10+ years'  -> 10, '< 1 year' -> 0  (extract digits)
    Both columns arrive as strings and must be numeric before any math is done.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'term' in X.columns:
            X['term'] = (X['term'].astype(str)
                           .str.extract(r'(\d+)')
                           .astype(float))
        if 'emp_length' in X.columns:
            X['emp_length'] = (X['emp_length']
                                .astype(str)
                                .str.replace('10+ years', '10', regex=False)
                                .str.replace('< 1 year',  '0',  regex=False)
                                .str.extract(r'(\d+)')
                                .astype(float))
        return X


class ParseCreditDate(BaseEstimator, TransformerMixin):
    """
    Cleaning Step 3:
    Converts `earliest_cr_line` (e.g. 'Aug-2003') into
    `credit_history_years` — the number of years between that date and
    the end of the dataset (Dec-2018). Drops the original string column.
    """
    REFERENCE_DATE = pd.Timestamp('2018-12-31')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'earliest_cr_line' in X.columns:
            dates = pd.to_datetime(X['earliest_cr_line'], format='%b-%Y', errors='coerce')
            X['credit_history_years'] = (self.REFERENCE_DATE - dates).dt.days / 365.25
            X.drop(columns=['earliest_cr_line'], inplace=True)
        return X


class EncodeGrade(BaseEstimator, TransformerMixin):
    """
    Cleaning Step 4:
    Ordinal-encodes the loan grade (A=1 … G=7). Grade is already
    ordered by risk so ordinal encoding preserves that relationship
    without creating 7 dummy columns.
    """
    GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'grade' in X.columns:
            X['grade'] = X['grade'].map(self.GRADE_MAP).fillna(4)
        return X


class OneHotEncodeCats(BaseEstimator, TransformerMixin):
    """
    Cleaning Step 5:
    One-hot encodes the remaining nominal categorical columns:
    home_ownership, verification_status, and purpose.
    Column alignment is handled so train and test always have the same shape.
    """
    CAT_COLS = ['home_ownership', 'verification_status', 'purpose']

    def fit(self, X, y=None):
        self.dummies_cols_ = [c for c in self.CAT_COLS if c in X.columns]
        dummy_frame = pd.get_dummies(X[self.dummies_cols_], drop_first=True)
        self.encoded_cols_ = dummy_frame.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        dummies = pd.get_dummies(X[self.dummies_cols_], drop_first=True)
        # Align to columns seen during fit — fills unseen categories with 0
        dummies = dummies.reindex(columns=self.encoded_cols_, fill_value=0)
        X.drop(columns=self.dummies_cols_, inplace=True)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return X


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineering Steps FE1 – FE10 for loan tabular data:
      FE1  — Drop columns with > `missing_thresh` missing values
      FE2  — FICO midpoint  (avg of fico_range_low + fico_range_high)
      FE3  — Installment-to-monthly-income ratio
      FE4  — Loan-to-annual-income ratio
      FE5  — Log transform of annual_inc  (reduce right skew)
      FE6  — Log transform of revol_bal   (reduce right skew)
      FE7  — Binary flag: any public records  (pub_rec > 0)
      FE8  — Binary flag: any delinquencies   (delinq_2yrs > 0)
      FE9  — Drop original fico_range_low / fico_range_high (replaced by FE2)
      FE10 — Drop original revol_bal (replaced by log version in FE6)
    """
    def __init__(self, missing_thresh=0.40):
        self.missing_thresh = missing_thresh

    def fit(self, X, y=None):
        # FE1 — record high-missing columns on training data only (no leakage)
        missing_rate = X.isnull().mean()
        self.high_missing_cols_ = missing_rate[missing_rate > self.missing_thresh].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()

        # FE1 — Drop high-missing columns
        drop_missing = [c for c in self.high_missing_cols_ if c in X.columns]
        X.drop(columns=drop_missing, inplace=True)

        # FE2 — FICO midpoint
        if 'fico_range_low' in X.columns and 'fico_range_high' in X.columns:
            X['fico_mid'] = (X['fico_range_low'] + X['fico_range_high']) / 2

        # FE3 — Installment-to-monthly-income ratio
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            X['installment_to_income'] = X['installment'] / (X['annual_inc'] / 12 + 1e-9)

        # FE4 — Loan-to-annual-income ratio
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_income'] = X['loan_amnt'] / (X['annual_inc'] + 1e-9)

        # FE5 — Log annual income
        if 'annual_inc' in X.columns:
            X['log_annual_inc'] = np.log1p(X['annual_inc'])

        # FE6 — Log revolving balance
        if 'revol_bal' in X.columns:
            X['log_revol_bal'] = np.log1p(X['revol_bal'])

        # FE7 — Public record flag
        if 'pub_rec' in X.columns:
            X['has_pub_rec'] = (X['pub_rec'] > 0).astype(int)

        # FE8 — Delinquency flag
        if 'delinq_2yrs' in X.columns:
            X['has_delinq'] = (X['delinq_2yrs'] > 0).astype(int)

        # FE9 — Drop original fico range columns (replaced by midpoint)
        X.drop(columns=[c for c in ['fico_range_low', 'fico_range_high'] if c in X.columns],
               inplace=True)

        # FE10 — Drop original revol_bal (replaced by log version)
        X.drop(columns=[c for c in ['revol_bal'] if c in X.columns], inplace=True)

        return X


# =============================================================================
# NOT USED in this project — kept for reference / future projects
# =============================================================================

# class FeatureEngineer(BaseEstimator, TransformerMixin):
#     """
#     Time-series feature engineering for stock/price data.
#     Creates rolling EMA, ROC, Momentum, RSI, and MA features.
#     NOT applicable to loan tabular data — use LoanFeatureEngineer instead.
#     """
#     def __init__(self, windows=[5, 10, 20]):
#         self.windows = windows
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if isinstance(X, np.ndarray):
#             X_df = pd.DataFrame(X)
#         else:
#             X_df = X.copy()
#         data = X_df.squeeze()
#         X_out = pd.DataFrame(index=X_df.index)
#         for w in self.windows:
#             X_out[f'EMA_{w}'] = data.ewm(span=w, min_periods=w).mean()
#             M = data.diff(w - 1)
#             N = data.shift(w - 1)
#             X_out[f'ROC_{w}'] = (M / N) * 100
#             X_out[f'MOM_{w}'] = data.diff(w)
#             delta = data.diff()
#             u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
#             d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
#             avg_gain = u.ewm(com=w - 1, adjust=False).mean()
#             avg_loss = d.ewm(com=w - 1, adjust=False).mean()
#             rs = avg_gain / avg_loss
#             X_out[f'RSI_{w}'] = 100 - (100 / (1 + rs))
#             X_out[f'MA_{w}'] = data.rolling(w, min_periods=w).mean()
#         return X_out


# class PairFeatureEngineer(BaseEstimator, TransformerMixin):
#     """
#     Pairs trading feature engineering — rolling OLS regression between
#     two asset price series to compute spread, beta, and z-score.
#     NOT applicable to loan tabular data.
#     Requires: import statsmodels.api as sm
#     """
#     def __init__(self, window=60):
#         self.window = window
#         self.last_beta_ = None
#         self.last_alpha_ = None
#         self.is_fitted_ = False
#
#     def fit(self, X, y=None):
#         if len(X) < self.window:
#             raise ValueError(f"Data length {len(X)} is less than window size {self.window}")
#         self.is_fitted_ = True
#         return self
#
#     def transform(self, X):
#         if not self.is_fitted_:
#             raise RuntimeError("Extractor must be fitted before calling transform.")
#         if isinstance(X, np.ndarray):
#             df = pd.DataFrame(X, columns=['price_a', 'price_b'])
#         else:
#             df = X.copy()
#             df.columns = ['price_a', 'price_b']
#         df[['spread', 'beta']] = self._compute_rolling_regression(df)
#         df['z_score'] = self._calculate_z_score(df['spread'])
#         df['spread_std'] = df['spread'].rolling(self.window).std()
#         df['beta_stability'] = df['beta'].rolling(self.window).std()
#         return df
#
#     def _compute_rolling_regression(self, df):
#         import statsmodels.api as sm
#         spreads = np.full(len(df), np.nan)
#         betas = np.full(len(df), np.nan)
#         a_vals = df['price_a'].values
#         b_vals = df['price_b'].values
#         for i in range(self.window, len(df)):
#             y = a_vals[i-self.window:i]
#             x = b_vals[i-self.window:i]
#             x_with_const = sm.add_constant(x)
#             model = sm.OLS(y, x_with_const).fit()
#             alpha, beta = model.params[0], model.params[1]
#             betas[i] = beta
#             spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
#             self.last_alpha_, self.last_beta_ = alpha, beta
#         return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)
#
#     def _calculate_z_score(self, spread_series):
#         rolling_mean = spread_series.rolling(self.window).mean()
#         rolling_std = spread_series.rolling(self.window).std()
#         return (spread_series - rolling_mean) / rolling_std


# class Word2VecTransformer(BaseEstimator, TransformerMixin):
#     """
#     Converts text (e.g. news headlines) into averaged Word2Vec embeddings.
#     NOT applicable to loan tabular data — no free-text features are used.
#     Requires: from gensim.models import Word2Vec
#     """
#     def __init__(self, vector_size=100, window=5, min_count=1):
#         self.vector_size = vector_size
#         self.window = window
#         self.min_count = min_count
#         self.model = None
#
#     def fit(self, X, y=None):
#         sentences = [str(row[0]).split() for row in X]
#         self.model = Word2Vec(sentences, vector_size=self.vector_size,
#                               window=self.window, min_count=self.min_count)
#         return self
#
#     def transform(self, X):
#         def get_mean_vector(text):
#             words = str(text).split()
#             vectors = [self.model.wv[w] for w in words if w in self.model.wv]
#             if not vectors:
#                 return np.zeros(self.vector_size)
#             return np.mean(vectors, axis=0)
#         return np.array([get_mean_vector(row[0]) for row in X])
