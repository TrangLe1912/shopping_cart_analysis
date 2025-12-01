# -*- coding: utf-8 -*-
"""
Shopping Cart Library

This library contains classes for data cleaning, feature engineering,
and association rule analysis for shopping cart.
"""

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. DATA CLEANER
# =========================================================

class DataCleaner:
    """
    A class for cleaning and preprocessing retail transaction data.

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path):
        """
        Initialize the DataCleaner with data path.

        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        """
        Load and display basic information about the dataset.

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Chuyển CustomerID thành format 6 ký tự
        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

        print(f"Kích thước dữ liệu: {self.df.shape}")
        print(f"Số bản ghi: {len(self.df):,}")

        return self.df

    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing on UK customers.

        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Thêm cột TotalPrice
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Loại bỏ các hóa đơn bị hủy (bắt đầu bằng 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Chỉ tập trung vào khách hàng UK
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Loại bỏ các sản phẩm có quantity hoặc price không hợp lệ
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        # Bỏ description NA
        self.df_uk = self.df_uk.dropna(subset=["Description"])

        return self.df_uk

    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def add_total_price(self):
        """
        Add TotalPrice column (Quantity * UnitPrice) to cleaned UK data.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["TotalPrice"] = self.df_uk["Quantity"] * self.df_uk["UnitPrice"]
        return self.df_uk

    def compute_rfm(self, snapshot_date=None):
        """
        Compute RFM (Recency, Frequency, Monetary) for each customer based on cleaned UK data.

        Args:
            snapshot_date (datetime or str, optional):
                Reference date for Recency calculation.
                - If None: use max(InvoiceDate) + 1 day.

        Returns:
            pd.DataFrame: RFM dataframe with columns [CustomerID, Recency, Frequency, Monetary]
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        df = self.df_uk.copy()

        # Đảm bảo có TotalPrice
        if "TotalPrice" not in df.columns:
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # Xác định snapshot_date
        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        else:
            # Cho phép truyền vào dạng string 'YYYY-MM-DD'
            if isinstance(snapshot_date, str):
                snapshot_date = pd.to_datetime(snapshot_date)

        # Tính RFM
        rfm = df.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": "nunique",  # Frequency
                "TotalPrice": "sum",     # Monetary
            }
        )

        rfm.rename(
            columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary",
            },
            inplace=True,
        )

        self.rfm_data = rfm.reset_index()
        return self.rfm_data

    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.

        Args:
            output_dir (str): Output directory path
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/cleaned_uk_data.csv"
        self.df_uk.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đã làm sạch: {output_path}")


# =========================================================
# 2. BASKET PREPARER
# =========================================================

class BasketPreparer:
    """
    A class for preparing basket data for association rule mining.

    This class transforms transaction data into a format suitable for
    applying the Apriori algorithm.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        invoice_col: str = "InvoiceNo",
        item_col: str = "Description",
        quantity_col: str = "Quantity",
    ):
        """
        Initialize the BasketPreparer with cleaned dataframe.

        Args:
            df (pd.DataFrame): Cleaned transaction-level dataframe
            invoice_col (str): Column name for invoice number
            item_col (str): Column name for item description
            quantity_col (str): Column name for item quantity
        """
        self.df = df
        self.invoice_col = invoice_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.basket = None
        self.basket_bool = None

    def create_basket(self):
        """
        Create a basket format dataframe for Apriori algorithm.

        Returns:
            pd.DataFrame: Basket format dataframe
        """

        basket = (
            self.df.groupby([self.invoice_col, self.item_col])[self.quantity_col]
            .sum()
            .unstack()
            .fillna(0)
        )

        self.basket = basket
        return self.basket

    def encode_basket(self, threshold: int = 1):
        """
        Encode the basket dataframe into boolean format.

        Args:
            threshold (int): Minimum quantity to consider an item as present

        Returns:
            pd.DataFrame: Boolean encoded basket dataframe
        """

        if self.basket is None:
            raise ValueError("Basket not created. Please run create_basket() first.")
        basket_bool = self.basket.applymap(lambda x: 1 if x >= threshold else 0)
        basket_bool = basket_bool.astype(bool)
        self.basket_bool = basket_bool
        return self.basket_bool

    def save_basket_bool(self, output_path: str):
        """
        Save the boolean encoded basket dataframe to a Parquet file.

        Args:
            output_path (str): Path to save the Parquet file
        """
        if self.basket_bool is None:
            raise ValueError("Basket not encoded. Please call encode_basket() first.")
        basket_bool_to_save = self.basket_bool.reset_index(drop=True)

        basket_bool_to_save.to_parquet(output_path, index=False)
        print(f"Đã lưu basket boolean: {output_path}")


# =========================================================
# 3. APRIORI ASSOCIATION RULES MINER
# =========================================================

class AssociationRulesMiner:
    """
    A class for mining association rules using the Apriori algorithm.

    This class applies the Apriori algorithm to the basket data and extracts
    association rules based on specified metrics.
    """

    def __init__(self, basket_bool: pd.DataFrame):
        """
        Initialize the AssociationRulesMiner with basket data.

        Args:
            basket_bool (pd.DataFrame): Boolean encoded basket dataframe
        """
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None

    def mine_frequent_itemsets(
        self,
        min_support: float = 0.01,
        max_len: int = None,
        use_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using the Apriori algorithm.

        Returns:
            pd.DataFrame: DataFrame of frequent itemsets
        """

        fi = apriori(
            self.basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )

        fi.sort_values(by="support", ascending=False, inplace=True)
        self.frequent_itemsets = fi
        return self.frequent_itemsets

    def generate_rules(
        self,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        Args:
            metric (str): Metric to evaluate the rules
            min_threshold (float): Minimum threshold for the metric

        Returns:
            pd.DataFrame: DataFrame of association rules
        """

        if self.frequent_itemsets is None:
            raise ValueError(
                "Frequent itemsets not mined. Please run mine_frequent_itemsets() first."
            )

        rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )

        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules
        return self.rules

    @staticmethod
    def _frozenset_to_str(fs: frozenset) -> str:
        return ", ".join(sorted(list(fs)))

    def add_readable_rule_str(self) -> pd.DataFrame:
        """
        Add human-readable columns for antecedents, consequents, and rule_str
        to the rules dataframe.

        Returns:
            pd.DataFrame: Rules dataframe with extra readable columns
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        rules = self.rules.copy()
        rules["antecedents_str"] = rules["antecedents"].apply(self._frozenset_to_str)
        rules["consequents_str"] = rules["consequents"].apply(self._frozenset_to_str)
        rules["rule_str"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

        self.rules = rules
        return self.rules

    def filter_rules(
        self,
        min_support: float = None,
        min_confidence: float = None,
        min_lift: float = None,
        max_len_antecedents: int = None,
        max_len_consequents: int = None,
    ) -> pd.DataFrame:
        """
        Filter rules based on support, confidence, lift and length of antecedents/consequents.
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        filtered = self.rules.copy()

        if min_support is not None:
            filtered = filtered[filtered["support"] >= min_support]
        if min_confidence is not None:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if min_lift is not None:
            filtered = filtered[filtered["lift"] >= min_lift]
        if max_len_antecedents is not None:
            filtered = filtered[
                filtered["antecedents"].apply(len) <= max_len_antecedents
            ]
        if max_len_consequents is not None:
            filtered = filtered[
                filtered["consequents"].apply(len) <= max_len_consequents
            ]

        filtered = filtered.reset_index(drop=True)
        return filtered

    def save_rules(self, output_path: str, rules_df: pd.DataFrame = None):
        """
        Save rules dataframe to CSV.

        Args:
            output_path (str): CSV path
            rules_df (pd.DataFrame): Rules dataframe to save (if None, use self.rules)
        """
        if rules_df is None:
            if self.rules is None:
                raise ValueError("No rules to save.")
            rules_df = self.rules

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rules_df.to_csv(output_path, index=False)
        print(f"Đã lưu luật vào: {output_path}")


# =========================================================
# 4. DATA VISUALIZER
# =========================================================

class DataVisualizer:
    """
    A class for creating visualizations for customer segmentation and
    shopping behavior analysis.

    This class provides methods for plotting various aspects of the data
    including temporal patterns, customer behavior, and RFM analysis.
    """

    def __init__(self):
        """Initialize the DataVisualizer with plotting settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Doanh thu hàng ngày")
        plt.xlabel("Ngày")
        plt.ylabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Doanh thu hàng tháng")
        plt.xlabel("Tháng")
        plt.ylabel("Doanh thu (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.

        Args:
            df (pd.DataFrame): Dataframe with time features:
                DayOfWeek, HourOfDay
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Hoạt động mua hàng theo ngày và giờ")
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe (có Quantity, TotalPrice)
            top_n (int): Number of top products to show
        """
        # Top sản phẩm theo số lượng
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} sản phẩm theo số lượng bán")
        plt.xlabel("Số lượng bán")
        plt.tight_layout()
        plt.show()

        # Top sản phẩm theo doanh thu
        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} sản phẩm theo doanh thu")
        plt.xlabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.

        Args:
            df (pd.DataFrame): Transaction dataframe with CustomerID, InvoiceNo, TotalPrice
        """
        # Số giao dịch trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Phân phối số giao dịch trên mỗi khách hàng")
        plt.xlabel("Số giao dịch")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

        # Chi tiêu trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Phân phối tổng chi tiêu trên mỗi khách hàng")
        plt.xlabel("Tổng chi tiêu (GBP)")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data (pd.DataFrame): RFM dataframe with
                columns ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        """
        # RFM distributions
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Phân phối Recency (Ngày kể từ lần mua cuối)")
        axes[0].set_xlabel("Ngày")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Phân phối Frequency (Số giao dịch)")
        axes[1].set_xlabel("Số giao dịch")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Phân phối Monetary (Tổng chi tiêu)")
        axes[2].set_xlabel("Tổng chi tiêu (GBP)")

        plt.tight_layout()
        plt.show()
