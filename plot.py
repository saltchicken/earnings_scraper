import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import stats

def get_recent_net_income(db_name="sp500_financials.db"):
    """
    Connects to the database and retrieves the most recent net_income 
    for each symbol based on the end_date.
    """
    query = """
        SELECT symbol, net_income
        FROM (
            SELECT 
                symbol, 
                net_income,
                ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY end_date DESC) as rn
            FROM quarterly_financials
            WHERE net_income IS NOT NULL
        )
        WHERE rn = 1
    """
    
    with sqlite3.connect(db_name) as conn:
        df = pd.read_sql_query(query, conn)
        
    return df

def currency_formatter(x, pos):
    """Formats axis values into Billions or Millions for readability."""
    if abs(x) >= 1e9:
        return f'${x * 1e-9:.1f}B'
    elif abs(x) >= 1e6:
        return f'${x * 1e-6:.1f}M'
    else:
        return f'${x:,.0f}'

def plot_distribution(df):
    """Plots the net income distribution using Linear  Yeo-Johnson scales."""
    data = df['net_income'].dropna()
    
    # Yeo-Johnson Transformation
    yj_data, lmbda = stats.yeojohnson(data.astype(float))
    
    # Set up a side-by-side figure (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Default Linear Scale ---
    sns.histplot(data, bins=60, kde=True, color='teal', ax=axes[0])
    axes[0].set_title('Linear Scale (Original)', fontsize=14, pad=10)
    axes[0].set_xlabel('Net Income (USD)', fontsize=12)
    axes[0].set_ylabel('Number of Companies', fontsize=12)
    axes[0].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].tick_params(axis='x', rotation=45)
    
    stats_text = (
        f"Count: {len(data)}\n"
        f"Median: {currency_formatter(data.median(), None)}\n"
        f"Min: {currency_formatter(data.min(), None)}"
    )
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Plot 2: Yeo-Johnson Transformation ---
    sns.histplot(yj_data, bins=60, kde=True, color='purple', ax=axes[1])
    axes[1].set_title(f'Yeo-Johnson Transform (λ ≈ {lmbda:.3f})', fontsize=14, pad=10)
    axes[1].set_xlabel('Transformed Value (Unitless)', fontsize=12)
    axes[1].set_ylabel('Number of Companies', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    yj_stats = f"Optimal Lambda: {lmbda:.3f}\nMaximized Normality"
    axes[1].text(0.95, 0.95, yj_stats, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Fetching the most recent net income data...")
    df_recent = get_recent_net_income()
    
    if df_recent.empty:
        print("No data found! Make sure 'sp500_financials.db' has data.")
    else:
        print(f"Data retrieved for {len(df_recent)} companies. Plotting distributions...")
        plot_distribution(df_recent)
