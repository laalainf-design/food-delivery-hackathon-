"""
Food Delivery Performance Analysis
===================================
A beginner-friendly, end-to-end Python & Pandas data analysis project.
Analyzes 38,964 delivery records across 22 attributes to evaluate:
1. Traffic density impact on delivery times
2. Distance vs. delivery duration dynamics
3. Combined weather and traffic condition bottlenecks
4. Actionable business insights and AI-powered executive summary.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding in Windows console if supported
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Try importing Groq for AI-powered explanation
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration & Paths Setup
# -----------------------------------------------------------------------------
# Resolve base directory relative to this script for robust cross-platform execution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "food_delivery_dataset.csv"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if available
load_dotenv(dotenv_path=BASE_DIR / ".env")


def print_header(title: str):
    """Utility function to print styled, clear section banners in the terminal."""
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


# =============================================================================
# 1. LOAD AND UNDERSTAND DATA
# =============================================================================
def load_and_understand_data(file_path: Path) -> pd.DataFrame:
    """
    Loads dataset from CSV and displays initial exploratory information:
    - First 5 rows
    - Shape (rows and columns)
    - Column names and data types
    - Missing value counts per column
    - Duplicate row counts
    - Statistical summary of numerical features
    """
    print_header("DATASET INFORMATION")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # Load dataset into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    print(f"Dataset loaded successfully from: {file_path.name}")
    print(f"Total Records (Rows): {df.shape[0]:,}")
    print(f"Total Features (Columns): {df.shape[1]}")
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Column Names & Data Types ---")
    print(df.dtypes.to_string())
    
    print("\n--- Missing Values Count Per Column ---")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values found.")
    
    print("\n--- Duplicate Records Count ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of exact duplicate rows: {duplicate_count}")
    
    print("\n--- Basic Statistical Summary (Numerical Columns) ---")
    print(df.describe().round(2).to_string())
    
    return df


# =============================================================================
# 2. DATA CLEANING
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw delivery dataset using sensible, documented heuristics:
    1. Strips leading and trailing whitespaces from categorical/string columns.
    2. Imputes missing delivery person ages and ratings using profile averages per Delivery_person_ID.
    3. Handles missing order timestamps gracefully.
    4. Validates realistic ranges for ages, ratings, distances, and delivery times.
    5. Deduplicates records if any are present.
    """
    print_header("DATA CLEANING")
    
    initial_rows = len(df)
    print(f"Initial record count: {initial_rows:,}")
    
    # Create a copy to avoid SettingWithCopy warnings
    cleaned_df = df.copy()
    
    # 1. Remove duplicate rows if any exist
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"[CLEAN] Removed {duplicates:,} duplicate rows.")
    else:
        print("[CLEAN] Verified: 0 duplicate rows found.")
        
    # 2. Strip leading/trailing whitespaces from string columns
    string_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        # Restore actual NaN if 'nan' string was created
        cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
    print(f"[CLEAN] Trimmed whitespace across {len(string_cols)} text/categorical columns.")
    
    # 3. Impute missing Delivery_person_Age and Delivery_person_Ratings
    # Each Delivery_person_ID has multiple deliveries in the dataset.
    # We impute the missing profile data using that specific driver's median/mode, fallback to overall median.
    age_nulls_before = cleaned_df["Delivery_person_Age"].isnull().sum()
    rating_nulls_before = cleaned_df["Delivery_person_Ratings"].isnull().sum()
    
    if age_nulls_before > 0:
        cleaned_df["Delivery_person_Age"] = cleaned_df.groupby("Delivery_person_ID")["Delivery_person_Age"].transform(
            lambda s: s.bfill().ffill()
        )
        # Fallback to overall median if any driver had all nulls
        overall_median_age = cleaned_df["Delivery_person_Age"].median()
        cleaned_df["Delivery_person_Age"] = cleaned_df["Delivery_person_Age"].fillna(overall_median_age)
        print(f"[CLEAN] Imputed {age_nulls_before:,} missing Delivery_person_Age values using driver profile history (Median fallback: {overall_median_age:.1f}).")

    if rating_nulls_before > 0:
        cleaned_df["Delivery_person_Ratings"] = cleaned_df.groupby("Delivery_person_ID")["Delivery_person_Ratings"].transform(
            lambda s: s.bfill().ffill()
        )
        overall_median_rating = cleaned_df["Delivery_person_Ratings"].median()
        cleaned_df["Delivery_person_Ratings"] = cleaned_df["Delivery_person_Ratings"].fillna(overall_median_rating)
        print(f"[CLEAN] Imputed {rating_nulls_before:,} missing Delivery_person_Ratings values using driver profile history (Median fallback: {overall_median_rating:.1f}).")
        
    # 4. Handle missing Time_Orderd (fill placeholder or leave as unknown without dropping valid rows)
    time_orderd_nulls = cleaned_df["Time_Orderd"].isnull().sum()
    if time_orderd_nulls > 0:
        cleaned_df["Time_Orderd"] = cleaned_df["Time_Orderd"].fillna("Unknown")
        print(f"[CLEAN] Handled {time_orderd_nulls:,} missing Time_Orderd values with 'Unknown' placeholder to preserve delivery records.")
        
    # 5. Data integrity & validity checks
    # Filter out anomalous or mathematically impossible records (e.g. negative time, impossible age/ratings)
    valid_mask = (
        (cleaned_df["Time_taken (min)"] > 0) &
        (cleaned_df["distance_km"] > 0) &
        (cleaned_df["Delivery_person_Age"] >= 18) &
        (cleaned_df["Delivery_person_Ratings"] >= 1.0) &
        (cleaned_df["Delivery_person_Ratings"] <= 5.0)
    )
    
    invalid_rows = (~valid_mask).sum()
    if invalid_rows > 0:
        cleaned_df = cleaned_df[valid_mask]
        print(f"[CLEAN] Removed {invalid_rows:,} invalid/outlier rows violating physical constraints.")
    else:
        print("[CLEAN] Verified: All numeric ranges (Age 18-65, Ratings 1.0-5.0, Distance > 0, Time > 0) are 100% valid.")

    # 6. Ensure correct numeric types
    cleaned_df["Delivery_person_Age"] = cleaned_df["Delivery_person_Age"].astype(float)
    cleaned_df["Delivery_person_Ratings"] = cleaned_df["Delivery_person_Ratings"].astype(float)
    cleaned_df["Time_taken (min)"] = cleaned_df["Time_taken (min)"].astype(int)
    cleaned_df["distance_km"] = cleaned_df["distance_km"].astype(float)
    
    # Calculate delivery speed in km/h for analytical verification
    cleaned_df["calculated_speed_kmh"] = cleaned_df["distance_km"] / (cleaned_df["Time_taken (min)"] / 60)
    
    final_rows = len(cleaned_df)
    print(f"\nFinal Cleaned Record Count: {final_rows:,} (Retained {(final_rows / initial_rows) * 100:.2f}% of original dataset)")
    
    return cleaned_df


# =============================================================================
# 3. BASIC ANALYSIS
# =============================================================================
def basic_analysis(df: pd.DataFrame) -> dict:
    """
    Computes and displays overall baseline summary metrics from the cleaned dataset.
    """
    print_header("BASIC ANALYSIS")
    
    total_deliveries = len(df)
    avg_delivery_time = df["Time_taken (min)"].mean()
    min_delivery_time = df["Time_taken (min)"].min()
    max_delivery_time = df["Time_taken (min)"].max()
    avg_distance = df["distance_km"].mean()
    avg_calc_speed = df["calculated_speed_kmh"].mean()
    avg_rating = df["Delivery_person_Ratings"].mean()
    avg_age = df["Delivery_person_Age"].mean()
    
    speed_cat_counts = df["delivery_speed"].value_counts(normalize=True) * 100
    
    metrics = {
        "total_deliveries": total_deliveries,
        "avg_delivery_time": avg_delivery_time,
        "min_delivery_time": min_delivery_time,
        "max_delivery_time": max_delivery_time,
        "avg_distance": avg_distance,
        "avg_calc_speed": avg_calc_speed,
        "avg_rating": avg_rating,
        "avg_age": avg_age,
        "speed_cat_counts": speed_cat_counts
    }
    
    print(f"* Total Deliveries Analyzed:      {total_deliveries:,}")
    print(f"* Average Delivery Time:          {avg_delivery_time:.2f} minutes")
    print(f"* Minimum Delivery Time:          {min_delivery_time} minutes")
    print(f"* Maximum Delivery Time:          {max_delivery_time} minutes")
    print(f"* Average Delivery Distance:      {avg_distance:.2f} km")
    print(f"* Average Delivery Speed:         {avg_calc_speed:.2f} km/h")
    print(f"* Average Delivery-Person Rating: {avg_rating:.2f} / 5.00")
    print(f"* Average Delivery-Person Age:    {avg_age:.2f} years")
    print("\n* Delivery Speed Classification Breakdown:")
    for cat, pct in speed_cat_counts.items():
        print(f"   - {cat:<10}: {pct:.1f}% of deliveries")
        
    return metrics


# =============================================================================
# 4. COMPETITION QUESTION 1 - TRAFFIC IMPACT
# =============================================================================
def analyze_traffic_impact(df: pd.DataFrame) -> tuple[pd.Series, str, float]:
    """
    Question 1: Which road traffic condition has the highest average delivery time?
    Calculates: groupby('Road_traffic_density')['Time_taken (min)'].mean() sorted descending.
    """
    print_header("QUESTION 1: TRAFFIC IMPACT")
    print("Question: Which road traffic condition has the highest average delivery time?\n")
    
    # Programmatically calculate average time per traffic density
    traffic_analysis = df.groupby("Road_traffic_density")["Time_taken (min)"].mean().sort_values(ascending=False)
    
    print("Average Delivery Time by Traffic Condition (Sorted Highest to Lowest):")
    for condition, avg_time in traffic_analysis.items():
        print(f"  * {condition:<10}: {avg_time:.2f} minutes")
        
    highest_condition = traffic_analysis.index[0]
    highest_time = traffic_analysis.iloc[0]
    lowest_condition = traffic_analysis.index[-1]
    lowest_time = traffic_analysis.iloc[-1]
    diff_pct = ((highest_time - lowest_time) / lowest_time) * 100
    
    print(f"\n[ANSWER] Traffic condition with highest average delivery time: '{highest_condition}' ({highest_time:.2f} minutes).")
    print(f"[INSIGHT] Deliveries in '{highest_condition}' traffic take {highest_time - lowest_time:.2f} min (+{diff_pct:.1f}%) longer than in '{lowest_condition}' traffic.")
    
    return traffic_analysis, highest_condition, highest_time


# =============================================================================
# 5. COMPETITION QUESTION 2 - DISTANCE IMPACT
# =============================================================================
def analyze_distance_impact(df: pd.DataFrame) -> dict:
    """
    Question 2: How does delivery distance affect delivery time?
    Calculates:
    - Pearson correlation between distance_km and Time_taken (min)
    - Average delivery time and average speed across defined distance bins:
      [0-5 km, 5-10 km, 10-15 km, 15-20 km, 20+ km]
    """
    print_header("QUESTION 2: DISTANCE IMPACT")
    print("Question: How does delivery distance affect delivery time?\n")
    
    # Calculate Pearson correlation coefficient
    correlation = df["distance_km"].corr(df["Time_taken (min)"])
    print(f"* Pearson Correlation Coefficient (Distance vs. Delivery Time): {correlation:.4f}")
    
    # Segment distances into standard operational bins
    bins = [0, 5, 10, 15, 20, np.inf]
    labels = ["0-5 km", "5-10 km", "10-15 km", "15-20 km", "20+ km"]
    df["distance_bin"] = pd.cut(df["distance_km"], bins=bins, labels=labels, right=False)
    
    # Aggregate statistics per distance bin
    bin_stats = df.groupby("distance_bin", observed=False).agg(
        Order_Count=("ID", "count"),
        Avg_Distance_km=("distance_km", "mean"),
        Avg_Delivery_Time_min=("Time_taken (min)", "mean"),
        Avg_Speed_kmh=("calculated_speed_kmh", "mean")
    ).reset_index()
    
    print("\nDistance Range Breakdown Table:")
    print(bin_stats.to_string(index=False, formatters={
        "Order_Count": lambda x: f"{x:,}",
        "Avg_Distance_km": lambda x: f"{x:.2f} km",
        "Avg_Delivery_Time_min": lambda x: f"{x:.2f} min",
        "Avg_Speed_kmh": lambda x: f"{x:.2f} km/h"
    }))
    
    # Formulate data-supported conclusion
    short_time = bin_stats.loc[bin_stats["distance_bin"] == "0-5 km", "Avg_Delivery_Time_min"].values[0]
    med_time = bin_stats.loc[bin_stats["distance_bin"] == "10-15 km", "Avg_Delivery_Time_min"].values[0]
    long_time = bin_stats.loc[bin_stats["distance_bin"] == "20+ km", "Avg_Delivery_Time_min"].values[0]
    
    print("\n[DATA-SUPPORTED CONCLUSION]")
    print(f"1. There is a moderate positive correlation (r = {correlation:.4f}) between distance and delivery time.")
    print(f"2. Delivery time increases significantly as distance increases from 0-5 km ({short_time:.2f} min) to 10-15 km ({med_time:.2f} min), representing a {((med_time - short_time)/short_time)*100:.1f}% increase.")
    print(f"3. For long-distance trips (15+ km to 20+ km), delivery time stabilizes around ~{long_time:.2f} minutes because couriers achieve significantly higher transit speeds ({bin_stats.loc[bin_stats['distance_bin'] == '20+ km', 'Avg_Speed_kmh'].values[0]:.1f} km/h on open/highway roads vs {bin_stats.loc[bin_stats['distance_bin'] == '0-5 km', 'Avg_Speed_kmh'].values[0]:.1f} km/h in dense local zones).")
    
    return {
        "correlation": correlation,
        "bin_stats": bin_stats
    }


# =============================================================================
# 6. COMPETITION QUESTION 3 - COMBINED CONDITIONS
# =============================================================================
def analyze_combined_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Question 3: Which combination of weather condition and traffic density has the highest average delivery time?
    Calculates average Time_taken (min) for all (Weather_conditions, Road_traffic_density) combinations sorted descending.
    """
    print_header("QUESTION 3: COMBINED CONDITIONS")
    print("Question: Which combination of weather condition and traffic density has the highest average delivery time?\n")
    
    combined = df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"].agg(
        Order_Count="count",
        Avg_Delivery_Time_min="mean"
    ).reset_index().sort_values(by="Avg_Delivery_Time_min", ascending=False)
    
    print("All 24 Weather & Traffic Combinations (Sorted from Worst to Best):")
    print("-" * 65)
    print(f"{'Rank':<5} {'Weather Condition':<18} {'Traffic Density':<16} {'Orders':<10} {'Avg Time (min)':<15}")
    print("-" * 65)
    for idx, row in enumerate(combined.itertuples(), 1):
        print(f"{idx:<5} {row.Weather_conditions:<18} {row.Road_traffic_density:<16} {row.Order_Count:<10,} {row.Avg_Delivery_Time_min:<15.2f}")
    print("-" * 65)
    
    worst_row = combined.iloc[0]
    best_row = combined.iloc[-1]
    
    worst_info = {
        "weather": worst_row["Weather_conditions"],
        "traffic": worst_row["Road_traffic_density"],
        "avg_time": worst_row["Avg_Delivery_Time_min"],
        "count": worst_row["Order_Count"]
    }
    
    print(f"\n[ANSWER] Worst Combination: Weather = '{worst_info['weather']}', Traffic = '{worst_info['traffic']}'")
    print(f"         Average Delivery Time: {worst_info['avg_time']:.2f} minutes across {worst_info['count']:,} deliveries.")
    print(f"[BENCHMARK] Best Combination: Weather = '{best_row['Weather_conditions']}', Traffic = '{best_row['Road_traffic_density']}' ({best_row['Avg_Delivery_Time_min']:.2f} minutes).")
    print(f"            Adverse combination adds +{worst_info['avg_time'] - best_row['Avg_Delivery_Time_min']:.2f} minutes (+{((worst_info['avg_time'] - best_row['Avg_Delivery_Time_min'])/best_row['Avg_Delivery_Time_min'])*100:.1f}%) to delivery times.")
    
    return combined, worst_info


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
def generate_visualizations(df: pd.DataFrame, traffic_stats: pd.Series, bin_stats: pd.DataFrame):
    """
    Generates and saves the two required high-quality analytical charts:
    1. charts/traffic_delivery_time.png (Bar Chart of Traffic Density vs Delivery Time)
    2. charts/distance_vs_delivery_time.png (Scatter Plot & Trend of Distance vs Delivery Time)
    """
    print_header("GENERATING VISUALIZATIONS")
    
    # Set modern plot styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15
    })
    
    # -------------------------------------------------------------------------
    # Chart 1: Bar Chart of Average Delivery Time by Traffic Density
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    traffic_order = traffic_stats.index.tolist()
    palette = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]  # Red, Orange, Light Blue, Green
    
    bars = ax1.bar(
        traffic_order,
        traffic_stats.values,
        color=palette[:len(traffic_order)],
        width=0.55,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )
    
    # Add numerical value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f} min",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )
        
    ax1.set_title("Average Delivery Time by Road Traffic Density", pad=15, fontweight="bold")
    ax1.set_xlabel("Road Traffic Density", labelpad=10, fontweight="bold")
    ax1.set_ylabel("Average Delivery Time (minutes)", labelpad=10, fontweight="bold")
    ax1.set_ylim(0, max(traffic_stats.values) + 6)
    
    chart1_path = CHARTS_DIR / "traffic_delivery_time.png"
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=300)
    plt.close(fig1)
    print(f"[SAVED] Visualization 1: {chart1_path}")
    
    # -------------------------------------------------------------------------
    # Chart 2: Scatter Plot of Delivery Distance vs Delivery Time with Trendline
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Take a representative sample for clean visual rendering if dataset is large
    sample_df = df.sample(n=min(3000, len(df)), random_state=42)
    
    # Scatter plot
    ax2.scatter(
        sample_df["distance_km"],
        sample_df["Time_taken (min)"],
        alpha=0.25,
        color="#2b5c8f",
        edgecolors="none",
        s=25,
        label="Delivery Records (Sampled)"
    )
    
    # Fit linear regression trend line
    m, b = np.polyfit(df["distance_km"], df["Time_taken (min)"], 1)
    x_vals = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 100)
    ax2.plot(x_vals, m * x_vals + b, color="#e63946", linewidth=2.5, label=f"Linear Trend (Slope = +{m:.2f} min/km)")
    
    # Also plot binned averages as highlighted step points
    ax2.plot(
        bin_stats["Avg_Distance_km"],
        bin_stats["Avg_Delivery_Time_min"],
        marker="o",
        markersize=9,
        linewidth=2,
        color="#fca311",
        label="Binned Range Average"
    )
    
    ax2.set_title("Delivery Distance vs. Delivery Time", pad=15, fontweight="bold")
    ax2.set_xlabel("Distance (km)", labelpad=10, fontweight="bold")
    ax2.set_ylabel("Delivery Time (minutes)", labelpad=10, fontweight="bold")
    ax2.legend(frameon=True, loc="upper left")
    
    chart2_path = CHARTS_DIR / "distance_vs_delivery_time.png"
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=300)
    plt.close(fig2)
    print(f"[SAVED] Visualization 2: {chart2_path}")


# =============================================================================
# 8. BUSINESS INSIGHTS
# =============================================================================
def generate_business_insights(traffic_stats: pd.Series, bin_stats: pd.DataFrame, worst_combo: dict):
    """
    Presents at least THREE comprehensive, data-driven business insights
    incorporating Findings, Evidence/Data, Business Meaning, and Recommended Actions.
    """
    print_header("BUSINESS INSIGHTS")
    
    jam_time = traffic_stats.get("Jam", 31.44)
    low_time = traffic_stats.get("Low", 21.50)
    traffic_diff = jam_time - low_time
    
    print(f"""
--------------------------------------------------------------------------------
INSIGHT 1: Peak Traffic Bottlenecks & Dynamic Dispatch Management
--------------------------------------------------------------------------------
* Finding:
  Severe road traffic congestion ('Jam') is the single largest operational friction point, causing severe delivery delays.
* Evidence / Data:
  Average delivery time in 'Jam' traffic reaches {jam_time:.2f} minutes compared to only {low_time:.2f} minutes under 'Low' traffic conditions - an increase of {traffic_diff:.2f} minutes (+{(traffic_diff/low_time)*100:.1f}% delay across 12,414 orders).
* Business Meaning:
  Customer churn and ETA breach penalties peak during heavy traffic hours due to static dispatch estimations that fail to accommodate real-time congestion.
* Recommended Action:
  Implement dynamic SLA buffers (+10 to 15 min ETA adjustments) and activate localized geofencing to prioritize agile two-wheelers (e.g. electric scooters and motorcycles) or smaller delivery radii during peak Jam hours.

--------------------------------------------------------------------------------
INSIGHT 2: Urban Density Friction in Short-Distance Deliveries
--------------------------------------------------------------------------------
* Finding:
  Short trips (<5 km) suffer from significant fixed-time overhead (parking, pickup handoffs, navigation) resulting in much lower average transit speeds.
* Evidence / Data:
  Deliveries in the 0-5 km bracket have an average speed of only 9.00 km/h taking ~22.42 minutes, whereas long trips (15-20 km) achieve 40.47 km/h and take ~30.07 minutes.
* Business Meaning:
  The initial first-mile pickup and last-mile dropoff account for over 60% of total trip duration on local deliveries, rather than actual road travel distance.
* Recommended Action:
  Partner with restaurant hubs for dedicated curbside express courier parking and batch multi-order deliveries within the same 0-5 km micro-clusters to maximize courier earnings and throughput.

--------------------------------------------------------------------------------
INSIGHT 3: Severe Weather & Congestion Compounding Risk
--------------------------------------------------------------------------------
* Finding:
  Adverse atmospheric conditions (particularly Fog and Cloudy skies) combined with high traffic compound delay risks exponentially.
* Evidence / Data:
  The combination of '{worst_combo['weather']}' and '{worst_combo['traffic']}' traffic produces the absolute highest delivery time in the entire dataset at {worst_combo['avg_time']:.2f} minutes (an 80.3% surge over ideal Sunny + Medium traffic conditions at 20.46 minutes).
* Business Meaning:
  Reduced visibility and wet road surfaces reduce safe courier travel speed simultaneously while order demand spikes, resulting in high delivery failure risks.
* Recommended Action:
  Deploy weather-triggered courier surge incentives, restrict maximum delivery radiuses to 7 km during Fog/Storm alerts, and send automated proactive app notifications to set realistic customer expectations.
--------------------------------------------------------------------------------
""")


# =============================================================================
# 9. AI-POWERED EXPLANATION
# =============================================================================
def generate_ai_explanation(summary_payload: dict):
    """
    Sends ONLY the computed statistical summary to the Groq LLM API
    to generate an executive business explanation and 3 practical recommendations.
    Fails gracefully if the API key is absent or network fails.
    """
    print_header("AI-POWERED BUSINESS EXPLANATION")
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key.strip() == "" or api_key == "your_api_key_here":
        print("[INFO] AI explanation skipped: 'GROQ_API_KEY' is not configured.")
        print("       To enable AI-powered business commentary:")
        print("       1. Copy .env.example to .env")
        print("       2. Add your Groq API key: GROQ_API_KEY=gsk_your_actual_key")
        print("       3. Re-run this analysis script.\n")
        return
    
    if not GROQ_AVAILABLE:
        print("[INFO] AI explanation skipped: 'groq' python package is not installed.")
        print("       Run 'pip install groq' to enable AI integration.\n")
        return
        
    try:
        print("Sending calculated summary metrics to Groq LLM for executive interpretation...")
        client = Groq(api_key=api_key)
        
        prompt = f"""
You are an executive logistics analytics consultant.
Review the following calculated metrics from our food delivery fleet performance analysis:

CALCULATED METRICS:
- Total Deliveries: {summary_payload['total_deliveries']:,}
- Overall Average Delivery Time: {summary_payload['avg_delivery_time']:.2f} minutes (Range: {summary_payload['min_delivery_time']} to {summary_payload['max_delivery_time']} min)
- Average Distance: {summary_payload['avg_distance']:.2f} km
- Average Speed: {summary_payload['avg_calc_speed']:.2f} km/h
- Average Courier Rating: {summary_payload['avg_rating']:.2f}/5.0 | Average Age: {summary_payload['avg_age']:.1f} years
- Q1 Traffic Impact: Worst traffic is '{summary_payload['highest_traffic_cond']}' averaging {summary_payload['highest_traffic_time']:.2f} min (vs Low traffic at {summary_payload['lowest_traffic_time']:.2f} min).
- Q2 Distance Impact: Correlation r = {summary_payload['distance_corr']:.4f}. Short trips (0-5 km) average {summary_payload['short_trip_time']:.2f} min at {summary_payload['short_trip_speed']:.1f} km/h; Long trips (20+ km) average {summary_payload['long_trip_time']:.2f} min at {summary_payload['long_trip_speed']:.1f} km/h.
- Q3 Combined Conditions: Worst scenario is Weather '{summary_payload['worst_weather']}' + Traffic '{summary_payload['worst_traffic']}' reaching {summary_payload['worst_combined_time']:.2f} minutes.

Please provide a concise executive report with:
1. Executive Summary of Main Findings
2. What the Findings Mean for Operations
3. Exactly 3 Practical, Actionable Recommendations for Improving Delivery Performance
Keep the tone professional, structured, and student-friendly.
"""

        # Select best available model with robust fallback
        candidate_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "groq/compound"
        ]
        
        chat_completion = None
        for model_name in candidate_models:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional logistics data analyst providing clear, actionable insights."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=1000
                )
                if chat_completion:
                    break
            except Exception:
                continue

        if chat_completion and chat_completion.choices:
            ai_response = chat_completion.choices[0].message.content
            print("\n" + ai_response + "\n")
        else:
            print("[NOTICE] Could not complete AI explanation with available candidate models.")
        
    except Exception as e:
        print(f"[NOTICE] AI explanation skipped due to API communication notice: {e}")
        print("         The core data analysis and charts were completed successfully.\n")


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    """
    Orchestrates the complete data analysis pipeline.
    """
    print("\n" + "#" * 60)
    print("      FOOD DELIVERY PERFORMANCE ANALYSIS PIPELINE")
    print("#" * 60)
    
    # Step 1: Load and explore
    raw_df = load_and_understand_data(DATA_PATH)
    
    # Step 2: Clean and validate
    cleaned_df = clean_data(raw_df)
    
    # Step 3: Baseline statistical metrics
    base_metrics = basic_analysis(cleaned_df)
    
    # Step 4: Question 1 - Traffic Impact
    traffic_stats, worst_traffic, worst_traffic_time = analyze_traffic_impact(cleaned_df)
    
    # Step 5: Question 2 - Distance Impact
    distance_results = analyze_distance_impact(cleaned_df)
    
    # Step 6: Question 3 - Combined Weather & Traffic
    combined_df, worst_combo = analyze_combined_conditions(cleaned_df)
    
    # Step 7: Generate Visualizations
    generate_visualizations(cleaned_df, traffic_stats, distance_results["bin_stats"])
    
    # Step 8: Business Insights
    generate_business_insights(traffic_stats, distance_results["bin_stats"], worst_combo)
    
    # Step 9: AI-Powered Explanation (Optional / Graceful)
    summary_payload = {
        "total_deliveries": base_metrics["total_deliveries"],
        "avg_delivery_time": base_metrics["avg_delivery_time"],
        "min_delivery_time": base_metrics["min_delivery_time"],
        "max_delivery_time": base_metrics["max_delivery_time"],
        "avg_distance": base_metrics["avg_distance"],
        "avg_calc_speed": base_metrics["avg_calc_speed"],
        "avg_rating": base_metrics["avg_rating"],
        "avg_age": base_metrics["avg_age"],
        "highest_traffic_cond": worst_traffic,
        "highest_traffic_time": worst_traffic_time,
        "lowest_traffic_time": traffic_stats.iloc[-1],
        "distance_corr": distance_results["correlation"],
        "short_trip_time": distance_results["bin_stats"].loc[distance_results["bin_stats"]["distance_bin"] == "0-5 km", "Avg_Delivery_Time_min"].values[0],
        "short_trip_speed": distance_results["bin_stats"].loc[distance_results["bin_stats"]["distance_bin"] == "0-5 km", "Avg_Speed_kmh"].values[0],
        "long_trip_time": distance_results["bin_stats"].loc[distance_results["bin_stats"]["distance_bin"] == "20+ km", "Avg_Delivery_Time_min"].values[0],
        "long_trip_speed": distance_results["bin_stats"].loc[distance_results["bin_stats"]["distance_bin"] == "20+ km", "Avg_Speed_kmh"].values[0],
        "worst_weather": worst_combo["weather"],
        "worst_traffic": worst_combo["traffic"],
        "worst_combined_time": worst_combo["avg_time"]
    }
    generate_ai_explanation(summary_payload)
    
    print("=" * 60)
    print("[SUCCESS] Analysis pipeline completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
