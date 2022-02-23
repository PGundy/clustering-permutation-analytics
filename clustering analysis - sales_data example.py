#%%
# import itertools
import os
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

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


CA = ComboAnalysis()
CA.inputData = sales_data
CA.inputColNames = sales_data_reporting_columns
CA.depthMax = 3
CA.checkInputs()

#%%
CA.createCombos()

#%%


def exampleFunction(data):
    return data.agg(
        Cohort_Pop=("flag", "count"),
        Count_of_Purchasers=("flag", np.sum),
        Rate_of_Purchase=("flag", np.mean),
    )


## Below is the computation step, can be lengthy
CA.runComboAnalysis(userFunc=exampleFunction)

ComboAnalysisData = CA.exportComboAnalysisData()
ComboAnalysisData.sample(5)

#%%

## Create a copy of the data for visualizations
ca_df_viz = ComboAnalysisData.copy()

## The nested lists have to be changed to strings
### This method does that and has the following arguments:
### inputColName - column name
### data - defaults to the class's interiorly stores data
### delimiter - seperated value. defaults to ' -- '

ca_df_viz["grouped_vars"] = CA.listToString(
    inputColName="grouped_vars", data=ca_df_viz
)

ca_df_viz["grouped_clean"] = CA.listToString(
    inputColName="grouped_clean", data=ca_df_viz
)

ca_df_viz.sample(5)

#%%

ca_df_viz["depth"] = ca_df_viz["depth"].astype(str)
ca_df_viz["Majority_Purchase"] = ca_df_viz["Rate_of_Purchase"] > 0.50

fig = ca_df_viz[ca_df_viz["Cohort_Pop"] >= 50].pipe(
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


#%%

fig2a = (
    ca_df_viz[
        (ca_df_viz["Cohort_Pop"] >= 500) & (ca_df_viz["depth"].isin(["1"]))
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
    ca_df_viz[
        (ca_df_viz["Cohort_Pop"] >= 2000)
        & (ca_df_viz["Rate_of_Purchase"] >= 0.5)
        & (ca_df_viz["depth"].isin(["1"]))
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

#%%

## here we run the same as above graph but with depth 1 & 2

fig2c = (
    ca_df_viz[
        (ca_df_viz["Cohort_Pop"] >= 2000)
        & (ca_df_viz["Rate_of_Purchase"] >= 0.5)
        & (ca_df_viz["depth"].isin(["1", "2"]))
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

fig2c

# %%
