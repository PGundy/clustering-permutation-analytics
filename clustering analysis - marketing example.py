#%%
import os
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.express as px

# %%

### The below process enables autocomplete on 'ComboAnalysis'
src_path = (
    "/Users/philipgundy/projects/_python/Python-Functions/ComboAnalysis.py"
)
dst_path = os.getcwd()
shutil.copy(src_path, dst_path)
print("Copied")

from ComboAnalysis import ComboAnalysis

# %%
##############################
######### Load Data ##########
##############################

### Customer Personality Analysis -- Marketing Data
### Kaggle: "https://www.kaggle.com/imakash3011/customer-personality-analysis"
marketing = pd.read_csv(
    "data/Kaggle - Customer Personality Analysis/marketing_campaign.csv",
    sep="\t",
)
marketing.rename(inplace=True, columns={"ID": "cust_id"})


#### Attributes
##
### People
## ID: Customer's unique identifier
## Year_Birth: Customer's birth year
## Education: Customer's education level
## Marital_Status: Customer's marital status
## Income: Customer's yearly household income
## Kidhome: Number of children in customer's household
## Teenhome: Number of teenagers in customer's household
## Dt_Customer: Date of customer's enrollment with the company
## Recency: Number of days since customer's last purchase
## Complain: 1 if the customer complained in the last 2 years, 0 otherwise
##
### Products
## MntWines: Amount spent on wine in last 2 years
## MntFruits: Amount spent on fruits in last 2 years
## MntMeatProducts: Amount spent on meat in last 2 years
## MntFishProducts: Amount spent on fish in last 2 years
## MntSweetProducts: Amount spent on sweets in last 2 years
## MntGoldProds: Amount spent on gold in last 2 years
##
### Promotion
## NumDealsPurchases: Number of purchases made with a discount
## AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
## AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
## AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
## AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
## AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
## Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
##
### Place
## NumWebPurchases: Number of purchases made through the company’s website
## NumCatalogPurchases: Number of purchases made using a catalogue
## NumStorePurchases: Number of purchases made directly in stores
## NumWebVisitsMonth: Number of visits to company’s website in the last month

marketing.head()

marketing["Complain"] = marketing["Complain"] > 0


marketing["ChildrenTotal"] = marketing["Kidhome"] + marketing["Teenhome"]
marketing["AnyChildren"] = marketing["ChildrenTotal"] > 0

marketing["Z_Profit"] = marketing["Z_Revenue"] - marketing["Z_CostContact"]

marketing["AcceptAnyCmp"] = (
    marketing["AcceptedCmp1"]
    + marketing["AcceptedCmp2"]
    + marketing["AcceptedCmp3"]
    + marketing["AcceptedCmp4"]
    + marketing["AcceptedCmp5"]
    + marketing["Response"]
) > 0

marketing["MultipleDeals"] = marketing["NumDealsPurchases"] > 1


# %%

marketing_reporting_columns = [
    "Education",
    "Marital_Status",
    # "ChildrenTotal",
    "AnyChildren",
    "Complain",
    "MultipleDeals",
]


## Are the columns in the underkying data found within our select subset of columns?
check = all(
    item in list(marketing.columns) for item in marketing_reporting_columns
)
if check is False:
    sys.exit(
        "Columns supplied in marketing_reporting_columns are not found in list(marketing.columns)"
    )

for i_col in marketing_reporting_columns:

    if marketing.dtypes[i_col] == "object":
        marketing[i_col] = marketing[i_col].fillna("Unknown")

    marketing[i_col] = marketing[i_col].astype(str)

marketing.info()

# %%


marketing_permutations = alpha_numeric_ordered_permutation(
    marketing_reporting_columns, depth_max=3
)

marketing_permutations = list(filter(None, marketing_permutations))
marketing_permutations

# %%

df_calc = pd.DataFrame()

for i_perm_loop in enumerate(marketing_permutations):
    i_perm_step = i_perm_loop[0]
    i_perm = i_perm_loop[1]
    i_total_step_count = len(marketing_permutations)

    print(i_perm_step, "of", i_total_step_count, "--", i_perm)

    df_calc_i = marketing.groupby(i_perm).agg(
        Cohort_Pop=("cust_id", "count"),
        Cohort_Accepted=("AcceptAnyCmp", np.sum),
        Cumm_Success_Rate=("AcceptAnyCmp", np.mean),
        Total_Profit=("Z_Profit", np.sum),
        Avg_Recency_Days=("Recency", np.mean),
        Avg_NumWebVisitsMonth=("NumWebVisitsMonth", np.mean),
        Total_NumWebVisitsMonth=("NumWebVisitsMonth", np.sum),
        Avg_CostContact=("Z_CostContact", np.mean),
    )
    df_calc_i["Avg_Profit_Per_Web_Visit"] = (
        df_calc_i["Total_Profit"] / df_calc_i["Total_NumWebVisitsMonth"]
    )

    ## pull calculated columns before reseting the index
    calculated_col_names = list(df_calc_i.columns)
    df_calc_i = df_calc_i.reset_index()

    df_calc_i["depth"] = len(i_perm)  ## number of varialbes interacted

    ## Create pd.Series where each 'cell' is a list of the relevant values
    df_calc_i["grouped_vars"] = pd.Series([i_perm] * len(df_calc_i))
    df_calc_i["grouped_values"] = df_calc_i[i_perm].apply(
        lambda row: list(row.values.astype(str)), axis=1
    )

    ## Now we need to zip the two series from above into a single structured series
    for row in np.arange(0, len(df_calc_i), 1):
        if row == 0:
            #    df_calc_i["grouped_clean"] = ""
            df_calc_i["grouped_clean"] = pd.Series(dtype=str)

        df_calc_i["grouped_clean"][row] = [
            i + ": " + j + "__"
            for i, j in zip(
                df_calc_i["grouped_vars"][row],
                df_calc_i["grouped_values"][row],
            )
        ]

    ## Select & order our final columns
    df_calc_i = df_calc_i[
        list(["depth", "grouped_vars", "grouped_values", "grouped_clean",])
        + calculated_col_names
    ]

    if i_perm_step == 0:
        df_calc = df_calc_i.copy()
    else:
        df_calc = df_calc.append(df_calc_i)

df_calc = df_calc.reset_index(drop=True)
df_calc.sample(n=10)


##TODO: functionalize the below code!!-- list -> clean_string ???
df_calc["grouped_vars"] = df_calc["grouped_vars"].apply(
    lambda x: " ".join(map(str, x))
)
df_calc["grouped_vars"] = df_calc["grouped_vars"].str.replace(
    r"\s", ", ", regex=True
)
df_calc["grouped_vars"] = df_calc["grouped_vars"].str.replace(
    r"(.*)(\,\s)(.*)", "\\1 & \\3", regex=True
)


##TODO: functionalize the below code!!-- list -> clean_complex_string ???
df_calc["grouped_clean"] = df_calc["grouped_clean"].apply(
    lambda x: " ".join(map(str, x))
)
df_calc["grouped_clean"] = df_calc["grouped_clean"].str.replace(
    r"(.*)(\_{2}$)", "\\1", regex=True
)
df_calc["grouped_clean"] = df_calc["grouped_clean"].str.replace(
    r"(.*)(\_{2}\s)(.*)", "\\1, & \\3", regex=True
)
df_calc["grouped_clean"] = df_calc["grouped_clean"].str.replace(
    r"\_{2}\s", ", ", regex=True
)

df_calc.sample(n=10)

#%%
df_calc.info()

#%%
df_calc[(df_calc["Cohort_Pop"] > 50)].sort_values(
    "Cumm_Success_Rate", ascending=False
).head(n=10)

# %%
df_calc[
    (df_calc["Cohort_Pop"] > 50) & (df_calc["depth"] == 1)
].sort_values("Cumm_Success_Rate", ascending=False).head(n=10)

# %%
df_calc[
    (df_calc["Cohort_Pop"] > 50) & (df_calc["depth"] == 2)
].sort_values("Cumm_Success_Rate", ascending=False).head(n=10)

# %%
df_calc[(df_calc["Cohort_Pop"] > 75)].sort_values(
    "Cumm_Success_Rate", ascending=False
).head(n=10)

# %%
df_calc.info()

# %%
df_calc_viz = df_calc.copy()
df_calc_viz["depth"] = df_calc_viz["depth"].astype(str)
df_calc_viz["depth"] = df_calc_viz["depth"].astype(str)

fig = df_calc_viz[df_calc_viz["Cohort_Pop"] >= 30].pipe(
    lambda df: px.histogram(
        data_frame=df,
        x="Cohort_Pop",
        log_y=True,
        nbins=len(range(0, 2_300, 100)),
    )
)
fig.update_layout(bargap=0.2)
fig

# %%
df_calc_viz.info()

#%%
fig2a = (
    df_calc_viz[
        (
            (df_calc_viz["Cohort_Pop"] > 50)
            & df_calc_viz["depth"].isin(["1"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>Larger area reflects more buyers!"
            ),
            x="Avg_Profit_Per_Web_Visit",
            y="Cumm_Success_Rate",
            size="Total_Profit",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2a
#%%
fig2b = (
    df_calc_viz[
        (
            (df_calc_viz["Cohort_Pop"] > 50)
            & df_calc_viz["depth"].isin(["2"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>Top right are our most profitable groups!"
            ),
            x="Avg_Profit_Per_Web_Visit",
            y="Cumm_Success_Rate",
            size="Total_Profit",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2b

#%%
fig2c = (
    df_calc_viz[
        (
            (df_calc_viz["Cohort_Pop"] > 50)
            & df_calc_viz["depth"].isin(["3"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>Top right are our most profitable groups!"
            ),
            x="Avg_Profit_Per_Web_Visit",
            y="Cumm_Success_Rate",
            size="Total_Profit",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2c

#%%
fig3a = (
    df_calc_viz[
        (
            (df_calc_viz["Cohort_Pop"] > 50)
            & df_calc_viz["depth"].isin(["1"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>How often is each group purchasing? (Scaled by total_profit)"
            ),
            x="Avg_Recency_Days",
            y="Cumm_Success_Rate",
            size="Total_Profit",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig3a

# %%

fig3b = (
    df_calc_viz[
        (
            (df_calc_viz["Cohort_Pop"] > 50)
            & df_calc_viz["depth"].isin(["2"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>How often is each group purchasing? (Scaled by total_profit)"
            ),
            x="Avg_Recency_Days",
            y="Cumm_Success_Rate",
            size="Total_Profit",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig3b

# %%
