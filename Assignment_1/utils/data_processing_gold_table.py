import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(
    snapshot_date_str,
    silver_loan_daily_directory,
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark
):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_suffix = snapshot_date_str.replace('-', '_')
    
    # 1. Load Silver LMS table (Driver) and create loan-level behavioral features
    lms_filepath = silver_loan_daily_directory + "silver_loan_daily_" + date_suffix + '.parquet'
    lms_df = spark.read.parquet(lms_filepath)
    
    lms_df = lms_df.withColumn(
        "paid_to_due_ratio", 
        F.when(col("due_amt") > 0, col("paid_amt") / col("due_amt")).otherwise(1.0)
    ).withColumn(
        "overdue_to_balance_ratio", 
        F.when(col("balance") > 0, col("overdue_amt") / col("balance")).otherwise(0.0)
    )

    # 2. Load Customer Silver tables
    attr_filepath = silver_attributes_directory + "silver_attributes_" + date_suffix + '.parquet'
    fin_filepath = silver_financials_directory + "silver_financials_" + date_suffix + '.parquet'
    click_filepath = silver_clickstream_directory + "silver_clickstream_" + date_suffix + '.parquet'

    attr_df = spark.read.parquet(attr_filepath).select(
        "Customer_ID", "snapshot_date", "Age_clean", "Occupation"
    )
    
    fin_df = spark.read.parquet(fin_filepath).select(
        "Customer_ID", "snapshot_date", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts_clean", "Num_Credit_Card_clean", "Interest_Rate_clean",
        "Num_of_Loan_clean", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age_in_mths", "delayed_payment_rate", "Monthly_Balance_clean"
    )
    
    click_df = spark.read.parquet(click_filepath)

    # 3. Combine loan features with customer features on Customer_ID and snapshot_date
    gold_df = lms_df \
        .join(attr_df, on=["Customer_ID", "snapshot_date"], how="left") \
        .join(fin_df, on=["Customer_ID", "snapshot_date"], how="left") \
        .join(click_df, on=["Customer_ID", "snapshot_date"], how="left")

    # 4. Save combined Gold feature store table
    partition_name = "gold_feature_store_" + date_suffix + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    
    gold_df.write.mode("overwrite").parquet(filepath)
    print('saved combined gold feature store to:', filepath, 'row count:', gold_df.count())
    
    return gold_df