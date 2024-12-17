import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency, fisher_exact
pd.options.display.float_format = '{:,.4f}'.format

def check_normality(data):
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        raise ValueError("Data must be a list, NumPy array, or pandas Series.")
    data = np.array(data) 

    test_stat_normality, p_value_normality = stats.shapiro(data)
    return p_value_normality, p_value_normality >= 0.05

def check_variance_homogeneity(data):
    if len(data) < 2:
        return None, False  
    test_stat_var, p_value_var = stats.levene(*data)
    return p_value_var, p_value_var >= 0.05

test_descriptions = {
    "t_test_independent": "The Independent T-Test compares the means of two independent groups to determine if they are significantly different. This test assumes that the data are normally distributed and that the variances of the two groups are equal (homogeneity of variance). It is commonly used in experiments where two different groups are compared, such as treatment vs. control groups.",
    "dependent_ttest": "The Dependent (Paired) T-Test compares the means of two related groups to identify significant differences. This is often used when the same subjects are measured under two different conditions, such as before and after a treatment. The test assumes that the differences between the paired observations are normally distributed.",
    "repeated_measure_anova": "Repeated Measures ANOVA is used to determine if there are significant differences between measurements taken multiple times on the same group. This is particularly useful for tracking changes over time or across different conditions. It assumes normality and sphericity of the data.",
    "oneway_anova": "One-Way ANOVA tests whether the means of three or more independent groups are significantly different. It is an extension of the independent T-Test for more than two groups. The test assumes normality and homogeneity of variances. If significant, post-hoc tests can identify which groups differ.",
    "Wilcoxon_signed_rank": "The Wilcoxon Signed-Rank Test is a non-parametric alternative to the Dependent T-Test. It tests whether the median of differences between paired observations is zero. This test is suitable for ordinal data or when the assumption of normality is violated.",
    "Mann_Whitney_U_Test": "The Mann-Whitney U Test is a non-parametric test that compares two independent groups to assess whether their distributions differ. It is often used as an alternative to the Independent T-Test when data are not normally distributed or are ordinal.",
    "Friedman_Chi_Square": "The Friedman Test is a non-parametric test used to detect differences between multiple repeated measures on the same subjects. It is an alternative to Repeated Measures ANOVA when the assumptions of normality or sphericity are violated.",
    "Kruskal_Wallis": "The Kruskal-Wallis Test is a non-parametric alternative to One-Way ANOVA. It tests whether the distributions of three or more independent groups differ. This test is useful for ordinal data or when the assumption of normality is violated.",
    "McNemar_test": "The McNemar Test is used for matched categorical data to determine if there is a significant change between two conditions. It is commonly used in pre-test/post-test designs or when analyzing paired binary outcomes (e.g., success vs. failure).",
    "Chi_squared_test": "The Chi-Squared Test assesses whether there is a significant association between two categorical variables. It compares the observed frequencies in a contingency table to the frequencies expected under the null hypothesis of independence.",
    "Fisher_exact_test": "Fisher's Exact Test is used for small contingency tables to test the association between two categorical variables. Unlike the Chi-Squared Test, it is suitable for small sample sizes and provides an exact p-value.",
    "Cochran_Q_test": "The Cochran Q Test is used to test for differences in proportions across three or more matched groups. It is particularly useful for binary data when the same subjects are observed under multiple conditions.",
    "Marginal_Homogeneity_test": "The Marginal Homogeneity Test evaluates whether the marginal distributions of two matched categorical variables are the same. It is often used in cases where the data are paired, and the goal is to determine if there is a significant shift in responses."
}

# Streamlit app starts here

# Configure Streamlit page
st.set_page_config(page_title="Hypothesis Testing Application", page_icon="⚛", layout="wide")

# Sidebar header and logo
st.sidebar.image("TEDU_LOGO.png", use_container_width=True) # /Users/serdarhosver/Desktop/511 Project/TEDU_LOGO.png
st.sidebar.title("ADS 511: Statistical Inference Methods")
st.sidebar.write("Developed by: Serdar Hosver")

st.sidebar.title("Hypothesis Testing Map")
st.sidebar.image("Hipotez teti Map.drawio.png",use_container_width=True) # /Users/serdarhosver/Desktop/511 Project/Hipotez teti Map.drawio.png

# Sidebar to display all available tests with details
st.sidebar.header("List of Available Hypothesis Tests")

for test, description in test_descriptions.items():
    with st.sidebar.expander(f"ℹ️ {test}"):
        st.write(description)
     
# Title and Introduction
st.title("Hypothesis Testing Application")
st.markdown("This app allows you to perform various **hypothesis tests** with ease. Simply upload your data or enter it manually, and let the app guide you through hypothesis testing.")

# Step 1: Data Input
st.header("Step 1: Data Input")
data_choice = st.radio("How would you like to input your data?", ("Upload CSV", "Enter Manually"))


all_groups = []
if data_choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        columns = st.multiselect("Select columns for testing", options=data.columns)
        if columns:
            all_groups = [data[col].dropna().tolist() for col in columns]

elif data_choice == "Enter Manually":
    st.write("Enter your data as arrays. Type each array separately.")
    group_input = st.text_area("Enter arrays (e.g., [1,2,3]) one by one, separated by new lines.")
    if group_input:
        try:
            all_groups = [eval(line.strip()) for line in group_input.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Error in parsing input: {e}")

if not all_groups:
    st.warning("No valid data provided.")
    st.stop()

# Step 1.5: Data Type Selection
data_type = st.radio(
    "What is your data type?",
    options=["Select", "Numerical Data", "Categorical Data"],
    index=0
)

if data_type == "Select...":
    st.warning("Please select your data type to proceed.")
    st.stop()


# Step 2: Assumption Checks (if Numerical Data)
if data_type == "Numerical Data":
    st.header("Step 2: Assumption Check")
    st.write("Performing Normality and Variance Homogeneity Tests")

    results = []
    for i, group in enumerate(all_groups, start=1):
        try:
            p_normality, is_normal = check_normality(group)
            results.append((f"Group {i}", "Normality", p_normality, "Pass" if is_normal else "Fail"))
        except ValueError as e:
            st.error(f"Error with Group {i}: {e}")

    if len(all_groups) > 1:
        try:
            p_variance, is_homogeneous = check_variance_homogeneity(all_groups)
            results.append(("All Groups", "Variance Homogeneity", p_variance, "Pass" if is_homogeneous else "Fail"))
        except Exception as e:
            st.error(f"Error in Variance Homogeneity Test: {e}")

    results_df = pd.DataFrame(results, columns=["Group", "Test", "P-value", "Result"])
    st.table(results_df)

    if all(res[3] == "Pass" for res in results):
        st.info("Your data is parametric data")
        parametric = True
    else:
        st.info("Your data is non-parametric data")
        parametric = False

# Step 3: Hypothesis Testing
st.header("Step 3: Select and Perform a Hypothesis Test")

if data_type == "Numerical Data":
    if parametric:
        methods = [
            "t_test_independent", "dependent_ttest", "repeated_measure_anova", "oneway_anova"
        ]
    else:
        methods = [
            "Wilcoxon_signed_rank", "Mann_Whitney_U_Test", "Friedman_Chi_Square", "Kruskal_Wallis"
        ]
else:
    methods = [
        "McNemar_test", "Chi_squared_test", "Fisher_exact_test",
        "Cochran_Q_test", "Marginal_Homogeneity_test"
    ]

selected_method = st.selectbox("Choose a Hypothesis test to perform", methods)

if st.button("Run Test"):
    result_message = ""
    try:
        if selected_method in ["t_test_independent", "Mann_Whitney_U_Test", "Wilcoxon_signed_rank", "dependent_ttest"]:
            if len(all_groups) < 2:
                st.error("At least two groups are required for this test.")
            else:
                group1, group2 = all_groups[:2]
                if selected_method == "t_test_independent":
                    ttest, p_value = stats.ttest_ind(group1, group2)
                    result_message = f"T-test Independent: Test Statistic = {ttest:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Mann_Whitney_U_Test":
                    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                    result_message = f"Mann-Whitney U Test: Test Statistic = {u_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else:                    
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Wilcoxon_signed_rank":
                    w_stat, p_value = stats.wilcoxon(group1, group2)
                    result_message = f"Wilcoxon Signed-Rank Test: Test Statistic = {w_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "dependent_ttest":
                    t_stat, p_value = stats.ttest_rel(group1, group2)
                    result_message = f"Dependent T-test: Test Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")

        elif selected_method in ["oneway_anova", "Kruskal_Wallis", "Friedman_Chi_Square"]:
            if len(all_groups) < 3:
                st.error("At least three groups are required for this test.")
            else:
                group1, group2, group3 = all_groups[:3]
                if selected_method == "oneway_anova":
                    f_stat, p_value = stats.f_oneway(group1, group2, group3)
                    result_message = f"One-Way ANOVA: F Statistic = {f_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Kruskal_Wallis":
                    h_stat, p_value = stats.kruskal(group1, group2, group3)
                    result_message = f"Kruskal-Wallis Test: H Statistic = {h_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Friedman_Chi_Square":
                    chi_stat, p_value = stats.friedmanchisquare(group1, group2, group3)
                    result_message = f"Friedman Chi-Square Test: Chi-Square = {chi_stat:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")

        elif selected_method in ["McNemar_test", "Chi_squared_test", "Fisher_exact_test", "Cochran_Q_test", "Marginal_Homogeneity_test"]:
            if len(all_groups) < 2:
                st.error("Categorical tests require at least two groups.")
            else:
                group1, group2 = all_groups[:2]
                if selected_method == "McNemar_test":
                    table = pd.crosstab(group1, group2)
                    result, p_value = mcnemar(table, exact=False, correction=True)
                    result_message = f"McNemar Test: Statistic = {result.statistic:.4f}, P-value = {result.pvalue:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Chi_squared_test":
                    table = pd.crosstab(group1, group2)
                    chi2, p_value, _, _ = chi2_contingency(table)
                    result_message = f"Chi-Squared Test: Chi2 Statistic = {chi2:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
                elif selected_method == "Fisher_exact_test":
                    table = pd.crosstab(group1, group2)
                    odds_ratio, p_value = fisher_exact(table)
                    result_message = f"Fisher Exact Test: Odds Ratio = {odds_ratio:.4f}, P-value = {p_value:.4f}"
                    st.info(result_message)
                    if p_value < 0.05: 
                        st.success("Result: Reject null hypothesis") 
                    else: 
                        st.warning("Result: Fail to reject null hypothesis")
    except Exception as e:
                st.warning(f"You may have chosen the wrong hypothesis test. Please check Hypothesis Testing Map.")
                st.error(f"Error: {e}")
st.write("Thank you for using the Hypothesis Testing Application!")
# cd ~/Desktop/"511 Project"
# Run the app with: streamlit run Hypothesis_Testing_WebApp.py
# The app will open in your default browser
