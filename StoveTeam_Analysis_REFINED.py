"""
StoveTeam Charitable Giving Analysis - REFINED VERSION
========================================================
This script addresses academic feedback including:
1. Strict data validation against answer key
2. Proper state-level proxy variable naming
3. Corrected past giving variable definition
4. Robust standard errors (HC1)
5. Multicollinearity diagnostics (VIF and correlation matrix)

Author: [Your Name]
Date: December 2025
"""

# ============================================================
# SETUP AND LIBRARY IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print("="*60)
print("StoveTeam Analysis - REFINED VERSION")
print("="*60)
print("Libraries loaded successfully!\n")


# ============================================================
# DATA LOADING
# ============================================================

# For Google Colab - Mount Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive/Research Methods/Paper Publish/StoveTeam_Analysis/'
    print(f"✅ Google Drive mounted at: {base_path}\n")
except:
    # For local execution
    base_path = './'
    print(f"⚠️  Running locally. Files should be in current directory: {base_path}\n")

# Load May 2023 data
print("Loading data files...")
accts_may23 = pd.read_csv(f'{base_path}usedata_accts_may23.csv')
gifts_may23 = pd.read_csv(f'{base_path}usedata_gifts_may23.csv')

# Load November 2023 data
accts_nov23 = pd.read_csv(f'{base_path}usedata_accts_nov23.csv')
gifts_nov23 = pd.read_csv(f'{base_path}usedata_gifts_nov23.csv')

# Load May 2024 data
accts_may24 = pd.read_csv(f'{base_path}usedata_accts_may24.csv')
gifts_may24 = pd.read_csv(f'{base_path}usedata_gifts_may24.csv')

# Load state characteristics data
state_charac = pd.read_csv(f'{base_path}state_charac.csv')

print("✅ All data files loaded successfully!")
print(f"   accts_may23: {accts_may23.shape}")
print(f"   gifts_may23: {gifts_may23.shape}")
print(f"   accts_nov23: {accts_nov23.shape}")
print(f"   gifts_nov23: {gifts_nov23.shape}")
print(f"   accts_may24: {accts_may24.shape}")
print(f"   gifts_may24: {gifts_may24.shape}")
print(f"   state_charac: {state_charac.shape}\n")


# ============================================================
# DATA MERGING
# ============================================================

print("Merging accounts and gifts data...")

# Merge May 2023
merged_may23 = pd.merge(
    accts_may23,
    gifts_may23,
    on=['acct.id', 'period'],
    how='outer'
)

# Merge November 2023
merged_nov23 = pd.merge(
    accts_nov23,
    gifts_nov23,
    on=['acct.id', 'period'],
    how='outer'
)

# Merge May 2024
merged_may24 = pd.merge(
    accts_may24,
    gifts_may24,
    on=['acct.id', 'period'],
    how='outer'
)

print(f"✅ Merging complete!")
print(f"   May 2023 merged: {merged_may23.shape}")
print(f"   November 2023 merged: {merged_nov23.shape}")
print(f"   May 2024 merged: {merged_may24.shape}\n")


# ============================================================
# CRITICAL: ANSWER KEY VALIDATION (MAY 2023)
# ============================================================

print("="*60)
print("ANSWER KEY VALIDATION (MAY 2023 DATASET)")
print("="*60)

# Check 1: Total Account Count
total_accounts = len(merged_may23)
expected_accounts = 6249
check1_pass = (total_accounts == expected_accounts)

print(f"Check 1: Total Account Count")
print(f"   Expected: {expected_accounts}")
print(f"   Actual:   {total_accounts}")
print(f"   Status:   {'✅ PASS' if check1_pass else '❌ FAIL'}\n")

# Check 2: Mean of gift.amount
mean_gift_amount = merged_may23['gift.amount'].mean()
expected_mean = 186.15
# Allow small floating point tolerance
check2_pass = abs(mean_gift_amount - expected_mean) < 0.01

print(f"Check 2: Mean of gift.amount")
print(f"   Expected: {expected_mean}")
print(f"   Actual:   {mean_gift_amount:.2f}")
print(f"   Status:   {'✅ PASS' if check2_pass else '❌ FAIL'}\n")

# Check 3: Count of "Active" Accounts
# Active = appeal_sent_yes==1 OR email_sent_yes==1
active_count = len(merged_may23[
    (merged_may23['appeal_sent_yes'] == 1) | 
    (merged_may23['email_sent_yes'] == 1)
])
expected_active = 4488
check3_pass = (active_count == expected_active)

print(f"Check 3: Active Accounts Count")
print(f"   Expected: {expected_active}")
print(f"   Actual:   {active_count}")
print(f"   Status:   {'✅ PASS' if check3_pass else '❌ FAIL'}\n")

# Overall validation status
all_checks_pass = check1_pass and check2_pass and check3_pass

if all_checks_pass:
    print("🎉 ALL VALIDATION CHECKS PASSED! Data matches Answer Key.\n")
else:
    print("⚠️  WARNING: Some validation checks failed!")
    print("   Please review merge/filter logic to match the Answer Key.\n")
    if not check1_pass:
        print(f"   → Account count mismatch: {total_accounts} vs {expected_accounts}")
    if not check2_pass:
        print(f"   → Mean gift amount mismatch: {mean_gift_amount:.2f} vs {expected_mean}")
    if not check3_pass:
        print(f"   → Active accounts mismatch: {active_count} vs {expected_active}")
    print()


# ============================================================
# DATE CALCULATIONS AND FEATURE ENGINEERING
# ============================================================

print("="*60)
print("FEATURE ENGINEERING")
print("="*60)

def process_dates_and_features(df, period_name):
    """
    Process dates and create engineered features for a given period.
    
    NOTE ON STATE-LEVEL PROXIES:
    The demographic variables (income, age, education, foreign-born) are
    STATE-LEVEL aggregates from census data, not individual donor metrics.
    These serve as proxies for individual characteristics based on the
    donor's state of residence. This is a limitation of the available data.
    """
    
    df = df.copy()
    
    # Convert date columns to datetime
    date_cols = ['date.sent', 'gift.first.date', 'gift.most.recent.date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate giving length (days from first gift to most recent gift)
    df['giving.length'] = (df['gift.most.recent.date'] - df['gift.first.date']).dt.days
    df['giving.length'] = df['giving.length'].fillna(0)
    
    # Calculate gift duration (days from appeal sent to first gift in period)
    df['gift.duration'] = (df['gift.first.date'] - df['date.sent']).dt.days
    
    # ===========================================================
    # FIX 3: CORRECTED PAST GIVING VARIABLE
    # ===========================================================
    # Past giving should indicate if donor gave BEFORE the current appeal
    # Logic: If gift.first.date < date.sent, then they are a past donor
    
    df['gift_past_yes'] = 0
    mask_past = (df['gift.first.date'] < df['date.sent']) & df['gift.first.date'].notna()
    df.loc[mask_past, 'gift_past_yes'] = 1
    
    print(f"\n{period_name}:")
    print(f"   Past donors identified: {df['gift_past_yes'].sum()}")
    
    # Create binary outcome variable (gave in current period)
    df['gift_current_yes'] = (df['gift.amount'] > 0).astype(int)
    df['gift_current_yes'] = df['gift_current_yes'].fillna(0)
    
    # Merge with state characteristics
    df = pd.merge(df, state_charac, on='state', how='left')
    
    # ===========================================================
    # FIX 2: RENAMED STATE-LEVEL PROXY VARIABLES
    # ===========================================================
    # These variables are STATE-LEVEL proxies, not individual measurements
    
    # Z-score for median household income (state-level)
    df['z_income_state_proxy'] = stats.zscore(df['med.hh.income'].fillna(df['med.hh.income'].median()))
    
    # Z-score for median age (state-level)
    df['z_age_state_proxy'] = stats.zscore(df['med.age'].fillna(df['med.age'].median()))
    
    # Z-score for bachelor's degree attainment (state-level)
    df['z_edu_state_proxy'] = stats.zscore(df['bachelors'].fillna(df['bachelors'].median()))
    
    # Z-score for foreign-born population percentage (state-level)
    df['foreign_born_state_proxy'] = stats.zscore(
        df['foreign.born'].fillna(df['foreign.born'].median())
    )
    
    # Appeal characteristic binary flags
    df['appeal.message.health'] = df['appeal.message.health'].fillna(0).astype(int)
    df['appeal.message.responsibility'] = df['appeal.message.responsibility'].fillna(0).astype(int)
    df['appeal.suggest.amount'] = df['appeal.suggest.amount'].fillna(0).astype(int)
    df['November_Appeal'] = (df['period'] == 'nov23').astype(int)
    
    # Message quality score
    df['appeal.message.quality'] = df['appeal.message.quality'].fillna(
        df['appeal.message.quality'].median()
    )
    
    # Tax incentive variable (state-level policy)
    df['Tax_Incentive'] = df['deduction'].fillna(0)
    
    return df

# Process all three periods
print("\nProcessing datasets...")
merged_may23 = process_dates_and_features(merged_may23, "May 2023")
merged_nov23 = process_dates_and_features(merged_nov23, "November 2023")
merged_may24 = process_dates_and_features(merged_may24, "May 2024")

print("\n✅ Feature engineering complete!\n")


# ============================================================
# PREPARE REGRESSION DATASETS
# ============================================================

print("="*60)
print("REGRESSION DATASET PREPARATION")
print("="*60)

def prepare_regression_data(df, period_name):
    """
    Prepare clean dataset for regression analysis.
    Filter to active accounts only (those who received appeals).
    """
    # Filter to active accounts (received appeal or email)
    df_reg = df[
        (df['appeal_sent_yes'] == 1) | (df['email_sent_yes'] == 1)
    ].copy()
    
    # Select relevant variables for regression
    reg_vars = [
        'gift_current_yes',           # Dependent variable
        'gift_past_yes',              # Fixed past giving variable
        'z_income_state_proxy',       # Renamed state proxies
        'z_age_state_proxy',
        'z_edu_state_proxy',
        'foreign_born_state_proxy',
        'appeal.message.health',
        'appeal.message.responsibility',
        'appeal.suggest.amount',
        'appeal.message.quality',
        'Tax_Incentive',
        'November_Appeal'
    ]
    
    df_reg = df_reg[reg_vars].dropna()
    
    print(f"\n{period_name}:")
    print(f"   Active accounts: {len(df_reg)}")
    print(f"   Donors (gave in period): {df_reg['gift_current_yes'].sum()}")
    print(f"   Non-donors: {(1 - df_reg['gift_current_yes']).sum()}")
    print(f"   Donation rate: {df_reg['gift_current_yes'].mean():.2%}")
    
    return df_reg

# Prepare regression datasets
reg_may23 = prepare_regression_data(merged_may23, "May 2023")
reg_nov23 = prepare_regression_data(merged_nov23, "November 2023")
reg_may24 = prepare_regression_data(merged_may24, "May 2024")

# Create pooled dataset
reg_pooled = pd.concat([reg_may23, reg_nov23, reg_may24], ignore_index=True)
print(f"\nPooled Dataset:")
print(f"   Total observations: {len(reg_pooled)}")
print(f"   Total donors: {reg_pooled['gift_current_yes'].sum()}")
print(f"   Overall donation rate: {reg_pooled['gift_current_yes'].mean():.2%}\n")


# ============================================================
# FIX 5: MULTICOLLINEARITY DIAGNOSTICS
# ============================================================

print("="*60)
print("MULTICOLLINEARITY DIAGNOSTICS")
print("="*60)

# Variables of concern (per professor feedback)
concern_vars = ['Tax_Incentive', 'November_Appeal']
all_predictors = [col for col in reg_pooled.columns if col != 'gift_current_yes']

print("\n1. CORRELATION MATRIX (Variables of Concern)")
print("-" * 60)
corr_matrix = reg_pooled[concern_vars].corr()
print(corr_matrix)
print()

# Interpret correlation
tax_nov_corr = corr_matrix.loc['Tax_Incentive', 'November_Appeal']
if abs(tax_nov_corr) > 0.7:
    print(f"⚠️  HIGH CORRELATION ({tax_nov_corr:.3f}): Consider excluding one variable")
elif abs(tax_nov_corr) > 0.5:
    print(f"⚠️  MODERATE CORRELATION ({tax_nov_corr:.3f}): Monitor for multicollinearity")
else:
    print(f"✅ LOW CORRELATION ({tax_nov_corr:.3f}): Variables can be included together")
print()

# Full correlation matrix for all predictors
print("2. FULL CORRELATION MATRIX (All Predictors)")
print("-" * 60)
full_corr = reg_pooled[all_predictors].corr()
print(full_corr.round(3))
print()

# Variance Inflation Factor (VIF)
print("3. VARIANCE INFLATION FACTORS (VIF)")
print("-" * 60)
print("Rule of thumb: VIF > 10 indicates problematic multicollinearity")
print("             VIF > 5 suggests potential multicollinearity\n")

X_vif = reg_pooled[all_predictors]
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(vif_data.to_string(index=False))
print()

# Highlight problematic VIFs
high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"⚠️  WARNING: {len(high_vif)} variable(s) with VIF > 10:")
    for idx, row in high_vif.iterrows():
        print(f"   → {row['Variable']}: VIF = {row['VIF']:.2f}")
else:
    print("✅ All VIF values below 10 - No severe multicollinearity detected")
print()

# Visualization: Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(full_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - All Predictor Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("📊 Correlation heatmap saved as 'correlation_heatmap.png'\n")


# ============================================================
# FIX 4: LOGIT REGRESSION WITH ROBUST STANDARD ERRORS
# ============================================================

print("="*60)
print("LOGIT REGRESSION MODELS (WITH ROBUST STANDARD ERRORS)")
print("="*60)

def run_logit_with_robust_se(df, period_name, formula=None):
    """
    Run logit regression with HC1 robust standard errors.
    HC1 = Heteroskedasticity-Consistent standard errors (MacKinnon & White 1985)
    """
    print(f"\n{period_name} Logit Regression")
    print("-" * 60)
    
    if formula is None:
        # Base model
        formula = """gift_current_yes ~ gift_past_yes + z_income_state_proxy + 
                     z_age_state_proxy + z_edu_state_proxy + foreign_born_state_proxy + 
                     appeal.message.health + appeal.message.responsibility + 
                     appeal.suggest.amount + appeal.message.quality + Tax_Incentive"""
    
    # Fit model with ROBUST standard errors (HC1)
    model = smf.logit(formula, data=df).fit(cov_type='HC1', disp=False)
    
    print(model.summary())
    print(f"\nModel fit with ROBUST standard errors (HC1 - Heteroskedasticity-Consistent)")
    print(f"N = {model.nobs:.0f}")
    print(f"Log-Likelihood = {model.llf:.4f}")
    print(f"Pseudo R² = {model.prsquared:.4f}")
    print(f"AIC = {model.aic:.4f}")
    print()
    
    return model

# Run individual period models
logit_may23 = run_logit_with_robust_se(reg_may23, "May 2023")
logit_nov23 = run_logit_with_robust_se(reg_nov23, "November 2023")
logit_may24 = run_logit_with_robust_se(reg_may24, "May 2024")

# Pooled model with interaction terms
print("\nPooled Model (with Interaction Terms)")
print("-" * 60)
pooled_formula = """gift_current_yes ~ gift_past_yes + z_income_state_proxy + 
                    z_age_state_proxy + z_edu_state_proxy + foreign_born_state_proxy + 
                    Tax_Incentive + November_Appeal + 
                    appeal.message.health + appeal.message.responsibility + 
                    appeal.suggest.amount + appeal.message.quality + 
                    Tax_Incentive:appeal.message.quality"""

logit_pooled = smf.logit(pooled_formula, data=reg_pooled).fit(cov_type='HC1', disp=False)
print(logit_pooled.summary())
print(f"\nPooled model fit with ROBUST standard errors (HC1)")
print(f"N = {logit_pooled.nobs:.0f}")
print(f"Log-Likelihood = {logit_pooled.llf:.4f}")
print(f"Pseudo R² = {logit_pooled.prsquared:.4f}")
print(f"AIC = {logit_pooled.aic:.4f}")
print()


# ============================================================
# MARGINAL EFFECTS (POOLED MODEL)
# ============================================================

print("="*60)
print("AVERAGE MARGINAL EFFECTS (POOLED MODEL)")
print("="*60)

# Calculate marginal effects
marginal_effects = logit_pooled.get_margeff(at='overall', method='dydx')
print(marginal_effects.summary())
print()


# ============================================================
# EXPORT REGRESSION RESULTS
# ============================================================

print("="*60)
print("EXPORTING RESULTS")
print("="*60)

def extract_model_results(model, model_name):
    """Extract regression results into a clean DataFrame."""
    results_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        'z_value': model.tvalues.values,
        'p_value': model.pvalues.values,
        'CI_Lower': model.conf_int()[0].values,
        'CI_Upper': model.conf_int()[1].values
    })
    results_df['Model'] = model_name
    results_df['N'] = int(model.nobs)
    results_df['Log_Likelihood'] = model.llf
    results_df['Pseudo_R2'] = model.prsquared
    results_df['AIC'] = model.aic
    results_df['SE_Type'] = 'Robust (HC1)'
    
    return results_df

# Extract results from all models
results_may23 = extract_model_results(logit_may23, 'May_2023')
results_nov23 = extract_model_results(logit_nov23, 'November_2023')
results_may24 = extract_model_results(logit_may24, 'May_2024')
results_pooled = extract_model_results(logit_pooled, 'Pooled')

# Combine all results
all_results = pd.concat([results_may23, results_nov23, results_may24, results_pooled], 
                        ignore_index=True)

# Save to CSV
all_results.to_csv('logit_regression_results_ROBUST.csv', index=False)
print("✅ Regression results exported to 'logit_regression_results_ROBUST.csv'")

# Export VIF results
vif_data.to_csv('vif_multicollinearity_diagnostics.csv', index=False)
print("✅ VIF diagnostics exported to 'vif_multicollinearity_diagnostics.csv'")

# Export correlation matrix
full_corr.to_csv('correlation_matrix_all_predictors.csv')
print("✅ Correlation matrix exported to 'correlation_matrix_all_predictors.csv'")

print()


# ============================================================
# SUMMARY REPORT
# ============================================================

print("="*60)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*60)
print()
print("FIXES IMPLEMENTED:")
print("✅ Fix 1: Data validation against Answer Key (May 2023)")
print("✅ Fix 2: Renamed variables to state-level proxies")
print("✅ Fix 3: Corrected past giving definition (gift.first.date < date.sent)")
print("✅ Fix 4: Implemented robust standard errors (HC1)")
print("✅ Fix 5: Generated VIF and correlation diagnostics")
print()
print("KEY FINDINGS:")
print(f"→ Tax_Incentive × November_Appeal correlation: {tax_nov_corr:.3f}")
print(f"→ Highest VIF: {vif_data.iloc[0]['Variable']} = {vif_data.iloc[0]['VIF']:.2f}")
print(f"→ Pooled model Pseudo R²: {logit_pooled.prsquared:.4f}")
print()
print("OUTPUT FILES:")
print("📄 logit_regression_results_ROBUST.csv")
print("📄 vif_multicollinearity_diagnostics.csv")
print("📄 correlation_matrix_all_predictors.csv")
print("📊 correlation_heatmap.png")
print()
print("="*60)
print("Script execution complete!")
print("="*60)
