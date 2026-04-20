import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# =========================
# 1. LOAD DATA
# =========================
# get prepared, combined data from data_merging .py file 
from data_merging import get_complete_dataset

df = get_complete_dataset()

# =========================
# 2. BASIC INSPECTION
# =========================
print(f"Dataset Shape: {df.shape}")
print(f"Total NaN: {df.isna().sum().sum()}")

print("\n=== TARGET DISTRIBUTIONS ===")
print("At Least 1 Case:")
print(df['Target_At_Least_1'].value_counts(normalize=True))

print("\nAt Least 2 Cases:")
print(df['Target_At_Least_2'].value_counts(normalize=True))

# =========================
# 3. CLEAN NUMERIC COLUMNS
# =========================
cols_to_clean = ['Estimate (%)', 'Estimated hesitant']

for col in cols_to_clean:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# 4. IMPUTATION
# =========================
# State-level median imputation
df['Estimated hesitant'] = df.groupby('STNAME')['Estimated hesitant'].transform(
    lambda x: x.fillna(x.median())
)
df['Estimated hesitant'] = df['Estimated hesitant'].fillna(df['Estimated hesitant'].median())

# =========================
# 5. FEATURE ENGINEERING
# =========================
df['cases_per_100k'] = (df['total_measles_cases'] / df['POPESTIMATE']) * 100000

# Optional: clip extreme values for visualization stability
df['cases_per_100k_clipped'] = df['cases_per_100k'].clip(upper=df['cases_per_100k'].quantile(0.99))

# =========================
# 6. VISUALIZATION
# =========================
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# -------------------------
# Plot 1: Target Comparison
# -------------------------
target_df = pd.DataFrame({
    '≥1 Case': df['Target_At_Least_1'].mean(),
    '≥2 Cases': df['Target_At_Least_2'].mean()
}, index=['Positive Rate']).T

target_df.plot(kind='bar', ax=axes[0, 0], legend=False)
axes[0, 0].set_title("Target Definition Comparison")
axes[0, 0].set_ylabel("Proportion of Counties")

# -------------------------
# Plot 2: Vaccination Distribution
# -------------------------
sns.histplot(df['Estimate (%)'], kde=True, ax=axes[0, 1], color='teal')
axes[0, 1].set_title("MMR Vaccination Rate Distribution")

# -------------------------
# Plot 3: Hesitancy Distribution
# -------------------------
sns.histplot(df['Estimated hesitant'], kde=True, ax=axes[1, 0], color='orange')
axes[1, 0].set_title("Vaccine Hesitancy Distribution")

# -------------------------
# Plot 4: Vaccination vs Cases
# -------------------------
sns.scatterplot(
    data=df,
    x='Estimate (%)',
    y='cases_per_100k_clipped',
    hue='Target_At_Least_2',
    alpha=0.6,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Vaccination Rate vs Cases (colored by ≥2 cases)")

# -------------------------
# Plot 5: Hesitancy vs Cases
# -------------------------
sns.scatterplot(
    data=df,
    x='Estimated hesitant',
    y='cases_per_100k_clipped',
    hue='Target_At_Least_2',
    alpha=0.6,
    ax=axes[2, 0]
)
axes[2, 0].set_title("Hesitancy vs Cases")

# -------------------------
# Plot 6: Correlation Heatmap
# -------------------------
corr_cols = [
    'cases_per_100k',
    'Estimate (%)',
    'Estimated hesitant',
    'POPESTIMATE'
]

corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2, 1])
axes[2, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.show()

# =========================
# 7. DUPLICATE CHECK
# =========================
duplicates = df.duplicated(subset='FIPS').sum()
print(f"\nDuplicate FIPS entries: {duplicates}")
