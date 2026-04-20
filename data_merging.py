import pandas as pd
import numpy as np


def build_complete_dataset(save_csv=True, output_path='Data/Complete_Dataset_For_Measles_Detection.csv'):
    # 1. LOAD DATA
    df1 = pd.read_csv('Data/cc-est2024-agesex-all.csv', encoding='latin1')
    df2 = pd.read_csv('Data/measles_county_all_updates_detailed.csv')
    df3 = pd.read_csv('Data/Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260320.csv')
    df4 = pd.read_csv('Data/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates_20260320.csv')

    # 2. CLEAN CENSUS DATA
    max_year = df1['YEAR'].max()
    census_clean = df1[(df1['YEAR'] == max_year) & (df1['COUNTY'] != 0)].copy()

    census_clean['FIPS'] = (
        census_clean['STATE'].astype(int).astype(str).str.zfill(2) +
        census_clean['COUNTY'].astype(int).astype(str).str.zfill(3)
    )

    # =========================
    # 3. CLEAN MEASLES DATA
    # =========================
    df2_cases = df2[df2['outcome_type'].isin(['case_imported', 'case_local'])].copy()

    df2_cases['FIPS'] = (
        pd.to_numeric(df2_cases['location_id'], errors='coerce')
        .astype('Int64')
        .astype(str)
        .str.zfill(5)
    )
    df2_cases = df2_cases[df2_cases['FIPS'] != '<NA>'].copy()

    agg_measles = (
        df2_cases.groupby('FIPS', as_index=False)
        .agg(total_measles_cases=('value', 'sum'))
    )

    # =========================
    # 4. MERGE CENSUS + MEASLES
    # =========================
    df_full = pd.merge(
        census_clean,
        agg_measles,
        on='FIPS',
        how='left'
    )

    df_full['total_measles_cases'] = df_full['total_measles_cases'].fillna(0)

    # =========================
    # 5. CREATE TARGET VARIABLES
    # =========================
    df_full['Target_At_Least_1'] = np.where(df_full['total_measles_cases'] >= 1, 1, 0)
    df_full['Target_At_Least_2'] = np.where(df_full['total_measles_cases'] >= 2, 1, 0)

    # =========================
    # 6. CLEAN VACCINATION DATA
    # =========================
    df3 = df3[df3['Vaccine/Exemption'] == 'MMR'].copy()

    df3['Year_end'] = (
        df3['School Year']
        .astype(str)
        .str.split('-')
        .str[-1]
        .astype(int)
    )
    df3['Year_end'] = 2000 + df3['Year_end']

    df3_latest = df3.sort_values('Year_end').groupby('Geography').tail(1).copy()
    df3_latest['STNAME'] = df3_latest['Geography']

    df_full_complete = pd.merge(
        df_full,
        df3_latest,
        on='STNAME',
        how='left'
    )

    # =========================
    # 7. CLEAN HESITANCY DATA
    # =========================
    df4['FIPS'] = (
        pd.to_numeric(df4['FIPS Code'], errors='coerce')
        .astype('Int64')
        .astype(str)
        .str.zfill(5)
    )
    df4 = df4[df4['FIPS'] != '<NA>'].copy()

    # =========================
    # 8. FINAL MERGE
    # =========================
    dfOW = pd.merge(
        df_full_complete,
        df4,
        on='FIPS',
        how='left'
    )

    if save_csv:
        dfOW.to_csv(output_path, index=False)

    return dfOW


def get_complete_dataset():
    return build_complete_dataset(save_csv=False)


if __name__ == "__main__":
    df = build_complete_dataset(save_csv=True)
    print(f"Saved dataset with shape: {df.shape}")
    print(df[['FIPS', 'total_measles_cases', 'Target_At_Least_1', 'Target_At_Least_2']].head())
