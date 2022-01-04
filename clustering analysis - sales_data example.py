#%%
import itertools
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# %%

sys.path.insert(0, "/Users/philipgundy/projects/_python//Python-Functions")

from alpha_numeric_ordered_permutations import (
    alpha_numeric_ordered_permutation,
)

# %%
##############################
######### Load Data ##########
##############################

### Individual Customer Sales Data
### Kaggle Link: "https://www.kaggle.com/mickey1968/individual-company-sales-data"

sales_data = pd.read_csv(
    "data/Kaggle - Individual Customer Sales Data/sales_data.csv"
)

sales_data["flag"] = np.where(sales_data["flag"] == "Y", True, False)

sales_data["fam_income"] = "Inc Lv:" + sales_data["fam_income"].apply(
    lambda x: f"{ord(x)%32:02}"  ## %32 changes uses to 0-26 for alphas
)

sales_data["age"] = sales_data["age"].str.replace(r"\_", " ", regex=True)

#### 40,000 rows of a customer's infomation, and includes the variables:
##
## flag: Whether the customer has bought the target product or not
## gender: Gender of the customer
## education: Education background of customer
## house_val: Value of the residence the customer lives in
## age: Age of the customer by group
## online: Whether the customer had online shopping experience or not
## customer_psy: Variable describing consumer psychology based on the area of residence
## marriage: Marriage status of the customer
## children: Whether the customer has children or not
## occupation: Career information of the customer
## mortgage: Housing Loan Information of customers
## house_owner: Whether the customer owns a house or not
## region: Information on the area in which the customer are located
## car_prob: The probability that the customer will buy a new car(1 means the maximum possible）
## fam_income: Family income Information of the customer(A means the lowest, and L means the highest)


sales_data.head()

# %%
sales_data_reporting_columns = [
    "gender",
    "education",
    "age",
    "online",
    "marriage",
    "region",
    "fam_income",
    "house_owner",
]

## Are the columns in the underkying data found within our select subset of columns?
check = all(
    item in list(sales_data.columns)
    for item in sales_data_reporting_columns
)
if check is False:
    sys.exit(
        "Columns supplied in sales_data_reporting_columns are not found in list(sales_data.columns)"
    )

for i_col in sales_data_reporting_columns:

    if sales_data.dtypes[i_col] == "object":
        sales_data[i_col] = sales_data[i_col].fillna("Unknown")

    sales_data[i_col] = sales_data[i_col].astype(str)

sales_data.info()


#%%

sales_data_permutations = alpha_numeric_ordered_permutation(
    sales_data_reporting_columns, depth_max=3
)

# sales_data_permutations = list(filter(None, sales_data_permutations))
sales_data_permutations

#%%

df_calc = pd.DataFrame()

for i_perm_loop in enumerate(sales_data_permutations):
    i_perm_step = i_perm_loop[0]
    i_perm = i_perm_loop[1]
    i_total_step_count = len(sales_data_permutations)

    print(i_perm_step, "of", i_total_step_count, "--", i_perm)

    df_calc_i = sales_data.groupby(i_perm).agg(
        Cohort_Pop=("flag", "count"),
        Count_of_Purchasers=("flag", np.sum),
        Rate_of_Purchase=("flag", np.mean),
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
        list(["depth", "grouped_vars", "grouped_values", "grouped_clean"])
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
df_calc[(df_calc["Cohort_Pop"] > 50)].sort_values(
    "Count_of_Purchasers", ascending=False
).head(n=10)

#%%

df_calc[
    (df_calc["Cohort_Pop"] > 50) & (df_calc["depth"] == 1)
].sort_values("Rate_of_Purchase", ascending=False).head(n=10)

#%%
df_calc[
    (df_calc["Cohort_Pop"] > 50) & (df_calc["depth"] == 2)
].sort_values("Rate_of_Purchase", ascending=False).head(n=10)

#%%
df_calc[(df_calc["Cohort_Pop"] > 75)].sort_values(
    "Rate_of_Purchase", ascending=False
).head(n=10)

# %%

df_calc.info()


# %%
df_calc_viz = df_calc.copy()
df_calc_viz["depth"] = df_calc_viz["depth"].astype(str)
df_calc_viz["depth"] = df_calc_viz["depth"].astype(str)
df_calc_viz["Majority_Purchase"] = df_calc_viz["Rate_of_Purchase"] > 0.50

fig = df_calc_viz[df_calc_viz["Cohort_Pop"] >= 50].pipe(
    # px.scatter(data=., x="Cohort_Pop", y="Rate_of_Purchase", color="depth")
    lambda df: px.histogram(
        data_frame=df,
        x="Cohort_Pop",
        log_y=True,
        nbins=len(range(0, 30_000, 1000)),
    )
)
fig.update_layout(bargap=0.2)
fig

# %%


fig2a = (
    df_calc_viz[
        (df_calc_viz["Cohort_Pop"] >= 500)
        & (df_calc_viz["depth"].isin(["1"]))
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Which groups are the best markting targets?"
                + "<br>Larger area reflects more buyers!"
            ),
            x="Cohort_Pop",
            y="Rate_of_Purchase",
            size="Count_of_Purchasers",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2a

# %%

fig2b = (
    df_calc_viz[
        (df_calc_viz["Cohort_Pop"] >= 2000)
        & (df_calc_viz["Rate_of_Purchase"] >= 0.5)
        & (df_calc_viz["depth"].isin(["1", "2"]))
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            x="Cohort_Pop",
            y="Rate_of_Purchase",
            title=(
                "Which groups are the best markting targets?"
                + "<br>Larger area reflects more buyers!"
            ),
            size="Count_of_Purchasers",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2b

# %%
