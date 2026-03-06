import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def plot_revenue_change(db_name="sp500_financials.db"):
    logger.info("Connecting to database and fetching revenue data...")
    
    try:
        with sqlite3.connect(db_name) as conn:
            # We fetch symbol, end_date, and revenues. 
            # We order by end_date DESC so the most recent periods are first.
            query = """
                SELECT symbol, end_date, revenues 
                FROM quarterly_financials 
                WHERE revenues IS NOT NULL 
                ORDER BY symbol, end_date DESC
            """
            df = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError as e:
        logger.error(f"Failed to read from database. Have you run the scraper yet? Error: {e}")
        return

    if df.empty:
        logger.warning("No revenue data found in the database.")
        return

    logger.info(f"Loaded {len(df)} total revenue records.")

    # 1. Keep only the two most recent records per symbol
    top2 = df.groupby('symbol').head(2)

    # 2. Ensure we have exactly 2 records to calculate a percent change
    counts = top2['symbol'].value_counts()
    valid_symbols = counts[counts == 2].index
    top2 = top2[top2['symbol'].isin(valid_symbols)]
    
    logger.info(f"Calculating QoQ percent change for {len(valid_symbols)} companies...")

    # 3. Sort chronologically (ascending) so pct_change calculates (Current - Previous) / Previous
    top2_asc = top2.sort_values(by=['symbol', 'end_date'])
    
    # 4. Calculate percent change for each group and multiply by 100 for percentage
    top2_asc['pct_change'] = top2_asc.groupby('symbol')['revenues'].pct_change() * 100

    # Clean up any infinities (e.g., if previous revenue was exactly 0) and drop the NaN rows 
    # (the first row for each symbol will naturally be NaN because there is no prior data to compare to)
    top2_asc = top2_asc.replace([np.inf, -np.inf], np.nan)
    recent_changes = top2_asc.dropna(subset=['pct_change'])

    # 5. Handle Outliers for Plotting
    # Financial data often has massive outliers (e.g. +3000% growth) which distort histograms.
    # We clip the top 5% and bottom 5% for a much cleaner visualization.
    q_low = recent_changes['pct_change'].quantile(0.05)
    q_hi = recent_changes['pct_change'].quantile(0.95)
    filtered_data = recent_changes[(recent_changes['pct_change'] >= q_low) & (recent_changes['pct_change'] <= q_hi)]

    logger.info(f"Plotting distribution for {len(filtered_data)} companies (excluding outliers outside {q_low:.2f}% to {q_hi:.2f}%)...")

    # --- Statistical Normality Test ---
    # D'Agostino's K-squared test combines skew and kurtosis to test for normality
    stat, p_value = stats.normaltest(filtered_data['pct_change'])
    logger.info(f"Original Normality Test (D'Agostino's K-squared): Statistic={stat:.4f}, p-value={p_value:.4e}")
    
    if p_value < 0.05:
        logger.info("Result: The original distribution is NOT perfectly normal (reject null hypothesis).")
    else:
        logger.info("Result: The original distribution looks normal (fail to reject null hypothesis).")

    # --- Yeo-Johnson Transformation ---
    # We create a copy of the dataframe slice to avoid SettingWithCopyWarning
    filtered_data = filtered_data.copy()
    transformed_data, lambda_val = stats.yeojohnson(filtered_data['pct_change'])
    filtered_data['yj_pct_change'] = transformed_data

    logger.info(f"Applied Yeo-Johnson Transformation with Lambda: {lambda_val:.4f}")

    # --- Identify Top 10 and Bottom 10 Scorers ---
    sorted_data = filtered_data.sort_values(by='yj_pct_change', ascending=False)
    
    top_10 = sorted_data.head(10)
    logger.info("--- Top 10 Companies by Transformed Growth ---")
    for index, row in top_10.iterrows():
        logger.info(f"  - {row['symbol']}: Transformed = {row['yj_pct_change']:.2f} (Original QoQ = {row['pct_change']:.2f}%)")

    bottom_10 = sorted_data.tail(10).sort_values(by='yj_pct_change', ascending=True)
    logger.info("--- Bottom 10 Companies by Transformed Growth ---")
    for index, row in bottom_10.iterrows():
        logger.info(f"  - {row['symbol']}: Transformed = {row['yj_pct_change']:.2f} (Original QoQ = {row['pct_change']:.2f}%)")

    # Retest for normality on the transformed data
    stat_yj, p_value_yj = stats.normaltest(filtered_data['yj_pct_change'])
    logger.info(f"Transformed Normality Test: Statistic={stat_yj:.4f}, p-value={p_value_yj:.4e}")
    
    if p_value_yj < 0.05:
        logger.info("Result: The transformed distribution is still NOT perfectly normal (reject null hypothesis).")
    else:
        logger.info("Result: The transformed distribution looks normal (fail to reject null hypothesis).")

    # 6. Plotting the distribution and Q-Q Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Top Left: Original Histogram ---
    sns.histplot(filtered_data['pct_change'], bins=40, kde=True, color='skyblue', edgecolor='black', ax=axes[0, 0])
    median_val = filtered_data['pct_change'].median()
    axes[0, 0].axvline(median_val, color='red', linestyle='dashed', linewidth=2, label=f"Median QoQ Growth: {median_val:.2f}%")
    axes[0, 0].set_title('Original Distribution of Revenue Growth', fontsize=12)
    axes[0, 0].set_xlabel('Revenue Percent Change (%)', fontsize=10)
    axes[0, 0].set_ylabel('Number of Companies', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # --- Top Right: Original Q-Q Plot ---
    stats.probplot(filtered_data['pct_change'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Original Q-Q Plot', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # --- Bottom Left: Transformed Histogram ---
    sns.histplot(filtered_data['yj_pct_change'], bins=40, kde=True, color='lightgreen', edgecolor='black', ax=axes[1, 0])
    median_yj = filtered_data['yj_pct_change'].median()
    axes[1, 0].axvline(median_yj, color='red', linestyle='dashed', linewidth=2, label=f"Median Transformed: {median_yj:.2f}")
    axes[1, 0].set_title('Yeo-Johnson Transformed Growth', fontsize=12)
    axes[1, 0].set_xlabel('Transformed Value', fontsize=10)
    axes[1, 0].set_ylabel('Number of Companies', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # --- Bottom Right: Transformed Q-Q Plot ---
    stats.probplot(filtered_data['yj_pct_change'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Transformed Q-Q Plot', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Quarter-over-Quarter Revenue Growth (S&P 500) - Before & After Transformation', fontsize=16)
    plt.tight_layout()
    output_filename = 'revenue_growth_distribution.png'
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Plot saved successfully to {output_filename}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_revenue_change()
