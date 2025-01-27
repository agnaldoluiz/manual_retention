import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from lifelines import KaplanMeierFitter

# App Title
st.title("Retention Analytics Dashboard")

# Load the data
final_df = pd.read_feather("data_results/final_df.ftr")

# Streamlit widgets to choose filtering option
grouping_option = st.selectbox(
    "Select the grouping option:", ("All", "By Country", "By Business Category")
)

# Filter the data based on the selection
if grouping_option == "All":
    filtered_df = final_df.copy()
elif grouping_option == "By Country":
    selected_country = st.selectbox(
        "Select Country", final_df["customer_country"].unique()
    )
    filtered_df = final_df[final_df["customer_country"] == selected_country].copy()
elif grouping_option == "By Business Category":
    selected_category = st.selectbox(
        "Select Business Category",
        final_df["taxonomy_business_category_group"].unique(),
    )
    filtered_df = final_df[
        final_df["taxonomy_business_category_group"] == selected_category
    ].copy()

# Key Numbers
st.header("Key Metrics")
st.metric("Total Customers", filtered_df["customer_id"].nunique())
st.metric("Total Subscriptions", filtered_df["subscription_id"].nunique())
st.metric(
    "Average Subscription Duration (days)",
    f"{filtered_df['subscription_duration'].mean():.2f}",
)


st.header("Exploratory data analysis")
# Country-wise Distribution
st.subheader("Customer Distribution by Country")
country_counts = filtered_df["customer_country"].value_counts()
st.bar_chart(country_counts)

# Taxonomy Group Distribution
st.subheader("Taxonomy Business Category Groups")
taxonomy_counts = filtered_df["taxonomy_business_category_group"].value_counts()
st.bar_chart(taxonomy_counts)

# Raw Data
st.subheader("Detailed Table")
st.dataframe(filtered_df)

# Retention heatmap

st.header("Retention heatmap")


def plot_retention_heatmap(final_df):
    # One row for each quarter the customer was active
    all_quarters = final_df.apply(
        lambda row: pd.date_range(row["from_date"], row["to_date"], freq="Q"), axis=1
    )
    df_expanded = final_df.loc[final_df.index.repeat(all_quarters.str.len())]
    df_expanded["active_quarter"] = np.concatenate(all_quarters.to_numpy())

    # Assign cohorts based on the first active quarter
    df_expanded["cohort"] = df_expanded.groupby("subscription_id")[
        "active_quarter"
    ].transform("min")

    # Calculate cohort index (quarter since acquisition)
    df_expanded["cohort_index"] = (
        df_expanded["active_quarter"].dt.year - df_expanded["cohort"].dt.year
    ) * 4 + (
        df_expanded["active_quarter"].dt.quarter - df_expanded["cohort"].dt.quarter
    )

    # Group data by cohort and cohort_index to count unique customers
    cohort_data = (
        df_expanded.groupby(["cohort", "cohort_index"])["subscription_id"]
        .nunique()
        .unstack(1)
    )

    # Normalize retention as a percentage of the cohort size at month 0
    cohort_sizes = cohort_data.iloc[:, 0]  # Size of each cohort at time 0
    retention = cohort_data.divide(cohort_sizes, axis=0) * 100

    # Replace NaN values with 0
    retention = retention.fillna(0)

    # Plot the heatmap
    fig = plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        retention,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar_kws={"label": "Retention (%)"},
        linewidths=0.5,
    )

    plt.title("Cohort Retention Heatmap", fontsize=16)
    plt.xlabel("Periods after Signup (Quarters)", fontsize=12)
    plt.ylabel("Cohort (Acquisition Month - Subscription)", fontsize=12)

    # Adjust labels
    quarter_labels = [f"{i + 1}" for i in range(retention.shape[1])]
    ax.set_xticklabels(quarter_labels)

    cohort_labels = retention.index.to_series().apply(
        lambda x: f"{x.year} Q{x.quarter}" if hasattr(x, "quarter") else str(x)
    )
    ax.set_yticklabels(cohort_labels)

    plt.xticks(rotation=0)
    return fig


st.pyplot(plot_retention_heatmap(filtered_df))


# Churn table
st.header("Churn table")


def get_churn_table(final_df):
    # Get the start and end of each month in the dataset
    min_date = final_df["from_date"].min()
    max_date = final_df["to_date"].max()
    months = pd.date_range(min_date.replace(day=1), max_date, freq="MS").to_period("M")

    # Initialize the summary table dictionary
    summary_dict = {
        "Metric": [
            "Existing Subscriptions",
            "New Subscriptions",
            "Cancelled Subscriptions",
            "Total Subscriptions",
        ]
    }

    # Loop through each month and calculate metrics
    for month in months:
        # Convert period to datetime for filtering
        month_start = month.start_time
        month_end = month.end_time

        # Existing Subscriptions (active at the beginning of the month)
        existing_subscriptions = final_df[
            (final_df["from_date"] < month_start) & (final_df["to_date"] >= month_start)
        ]

        # New Subscriptions (started in the month)
        new_subscriptions = final_df[
            (final_df["from_date"] >= month_start)
            & (final_df["from_date"] <= month_end)
        ]

        # Cancelled Subscriptions (ended in the month)
        cancelled_subscriptions = final_df[
            (final_df["to_date"] >= month_start) & (final_df["to_date"] <= month_end)
        ]

        # Total Subscriptions (active at the end of the month)
        total_subscriptions = final_df[
            (final_df["from_date"] <= month_end) & (final_df["to_date"] > month_end)
        ]

        # Append the counts to the dictionary
        summary_dict[str(month)] = [
            len(existing_subscriptions),
            len(new_subscriptions),
            len(cancelled_subscriptions),
            len(total_subscriptions),
        ]

    # Convert dictionary to DataFrame
    summary_df_q = pd.DataFrame(summary_dict)

    # Set the Metric column as index
    summary_df_q.set_index("Metric", inplace=True)

    return summary_df_q


st.dataframe(get_churn_table(filtered_df))


# Churn plot

st.header("Churn plot")


# Get churn plot
def get_churn_plot(final_df):
    # Calculate churn rate for each month
    churn_rates = []

    min_date = final_df["from_date"].min()
    max_date = final_df["to_date"].max()
    months = pd.date_range(min_date.replace(day=1), max_date, freq="MS").to_period("M")

    for month in months:
        # Convert period to datetime for filtering
        month_start = month.start_time
        month_end = month.end_time

        # Total Subscriptions at the start of the month (active from the previous month)
        total_subscriptions_start = final_df[
            (final_df["from_date"] < month_start) & (final_df["to_date"] >= month_start)
        ]

        # Cancelled Subscriptions during the month
        cancelled_subscriptions = final_df[
            (final_df["to_date"] >= month_start) & (final_df["to_date"] <= month_end)
        ]

        # Calculate churn rate
        churn_rate = (
            len(cancelled_subscriptions) / len(total_subscriptions_start)
            if len(total_subscriptions_start) > 0
            else 0
        )
        churn_rates.append(churn_rate)

    # Plot churn rate
    fig = plt.figure(figsize=(14, 6))
    plt.plot(
        months[3:-2].astype(str),
        churn_rates[3:-2],
        marker="o",
        color="b",
        label="Churn Rate",
    )  # delete initial and final months for clarity
    plt.xlabel("Month")
    plt.ylabel("Churn Rate")
    plt.title("Monthly Churn Rate")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    return fig


st.pyplot(get_churn_plot(filtered_df))

# Kaplan-Meier chart
st.header("Kaplan-Meier chart")


def get_km(final_df, grouping_option):
    # Instantiate Kaplan-Meier Fitter
    kmf = KaplanMeierFitter()

    if grouping_option == "All":
        # Fit the Kaplan-Meier model
        kmf.fit(
            durations=final_df["subscription_duration"],  # Subscription durations
            event_observed=~final_df[
                "censored"
            ],  # Convert censored column to event observed
        )

        # Plot the survival curve
        fig = plt.figure(figsize=(16, 10))
        kmf.plot_survival_function(ci_show=True)  # Show confidence interval

    elif grouping_option == "By Country":
        for cohort, cohort_data in final_df.groupby("customer_country"):
            kmf.fit(
                durations=cohort_data["subscription_duration"],
                event_observed=~cohort_data["censored"],
                label=f"Cohort {cohort}",
            )
            # Plot the survival curve
            fig = plt.figure(figsize=(16, 10))
            kmf.plot_survival_function(
                ci_show=False
            )  # Disable confidence intervals for clarity

    elif grouping_option == "By Business Category":
        for cohort, cohort_data in final_df.groupby("taxonomy_business_category_group"):
            kmf.fit(
                durations=cohort_data["subscription_duration"],
                event_observed=~cohort_data["censored"],
                label=f"Cohort {cohort}",
            )
            # Plot the survival curve
            fig = plt.figure(figsize=(16, 10))
            kmf.plot_survival_function(
                ci_show=False
            )  # Disable confidence intervals for clarity

    # Aesthetic improvements
    plt.title("Kaplan-Meier Survival Curves", fontsize=16, pad=20)
    plt.xlabel("Subscription Duration (days)", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.legend(title="Cohorts", fontsize=10, title_fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


get_km(final_df, grouping_option)
