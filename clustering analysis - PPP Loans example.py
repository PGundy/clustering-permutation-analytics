#%%
import itertools
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px

# %%


sys.path.insert(0, "/Users/philipgundy/projects/_python//Python-Functions")

from alpha_numeric_ordered_permutations import (
    alpha_numeric_ordered_permutation,
)

# %%

##############################
######### Load Data ##########
##############################

### PPP Loan Data & extras
### Kaggle: "https://www.kaggle.com/davincermak/payroll-protection-program-loanlevel-data"
ppp_loan = pd.read_csv("data/Kaggle - PPP Loan Data/ppp_loan_data.csv")
ppp_loan["veteran"] = np.where(
    ppp_loan["veteran"].isna(),
    "Non-Veteran -- Assumed",
    ppp_loan["veteran"],
)


#%%

ppp_loan_zip_code_info = pd.read_csv(
    "data/Kaggle - PPP Loan Data/zip_county_crosswalk.csv"
)
ppp_loan_zip_code_info.columns = ppp_loan_zip_code_info.columns.str.lower()

ppp_loan.info()

# 1 - RES_RATIO -- Ratio of residential addresses in the ZIP code to the number of residential addresses in the County.
# 2 - BUS_RATIO -- Ratio of business addresses in the ZIP code to the number of business addresses in the County.
# 3 - OTH_RATIO -- Ratio of other addresses in the ZIP code to the number of other addresses in the County.
# 4 - TOT_RATIO -- Ratio of all addresses in the ZIP code to the total number of all types of addresses in the entire County.

## ppploandata.cvs: In addition to the loan amount (in U.S. dollars), the data file contains loan-level demographic information on the loans originated under the PPP program. Geographic fields include state, city, U.S. Congressional District, and zip code. It also includes business ownership type and a detailed NAICS code indicating the borrower's type of business, originating bank, and some borrower characteristics.

## naics_6.csv: this file contains industry names to accompany the NAICS code in the data file

## zipcountycrosswalk.cvs: this file maps zip codes to U.S. counties for anyone interested in incorporating additional demographic, economic, or other data of interest.


#%%
ppp_loan__NAICS_6 = pd.read_csv("data/Kaggle - PPP Loan Data/naics_6.csv")
ppp_loan__NAICS_6.rename(
    inplace=True,
    columns={
        "industry_code": "naicscode",
        "industry_title": "industry_title_full",
    },
)

### Below is from the NAICS website about generalized industry codes
RAW_key_NAICS_codes = """11;Agriculture, Forestry, Fishing and Hunting\n21;Mining\n22;Utilities\n23;Construction\n31;Manufacturing\n32;Manufacturing\n33;Manufacturing\n42;Wholesale Trade\n44;Retail Trade\n45;Retail Trade\n48;Transportation and Warehousing\n49;Transportation and Warehousing\n51;Information\n52;Finance and Insurance\n53;Real Estate Rental and Leasing\n54;Professional, Scientific, and Technical Services\n55;Management of Companies and Enterprises\n56;Admin Support and Waste Management\n61;Educational Services\n62;Health Care and Social Assistance\n71;Arts, Entertainment, and Recreation\n72;Accommodation and Food Services\n81;Other Services (except Public Admin)\n92;Public Administration"""

key_NAICS_codes = pd.DataFrame(
    [x.split(";") for x in RAW_key_NAICS_codes.split("\n")]
)
key_NAICS_codes.rename(
    inplace=True, columns={0: "Industry_Code", 1: "Industry_Name"}
)

key_NAICS_codes["Industry_Code"] = key_NAICS_codes["Industry_Code"].astype(
    str
)


ppp_loan__NAICS_6["Industry_Code"] = ppp_loan__NAICS_6["naicscode"].apply(
    lambda x: str(x)[:2]
)

ppp_loan__NAICS_6 = ppp_loan__NAICS_6.merge(
    key_NAICS_codes, how="left", on="Industry_Code"
)

ppp_loan__NAICS_6.head()


#%%

ppp_loan["Industry_Code"] = ppp_loan["naicscode"].apply(
    lambda x: str(x)[:2]
)
ppp_loan = ppp_loan.merge(key_NAICS_codes, how="left", on="Industry_Code")

# %%

ppp_loan_reporting_columns = [
    "state",
    "businesstype",
    ##"raceethnicity", ## This has so many Unknowns it is useless
    ##"gender", ## This has so many Unknowns it is useless
    ##"veteran", ## This has so many Unknowns it is useless
    "Industry_Name",
    ##"lender",
]

## Are the columns in the underkying data found within our select subset of columns?
check = all(
    item in list(ppp_loan.columns) for item in ppp_loan_reporting_columns
)
if check is False:
    sys.exit(
        "Columns supplied in ppp_loan_reporting_columns are not found in list(ppp_loan.columns)"
    )

for i_col in ppp_loan_reporting_columns:

    if ppp_loan.dtypes[i_col] == "object":
        ppp_loan[i_col] = ppp_loan[i_col].fillna("Unknown")

    ppp_loan[i_col] = ppp_loan[i_col].astype(str)

ppp_loan.info()


#%%

ppp_loan_permutations = alpha_numeric_ordered_permutation(
    ppp_loan_reporting_columns, depth_max=3
)

ppp_loan_permutations

#%%

df_calc = pd.DataFrame()

for i_perm_loop in enumerate(ppp_loan_permutations):
    i_perm_step = i_perm_loop[0]
    i_perm = i_perm_loop[1]
    i_total_step_count = len(ppp_loan_permutations)

    print(i_perm_step, "of", i_total_step_count, "--", i_perm)

    df_calc_i = ppp_loan.groupby(i_perm).agg(
        loan_count=("loanamount", "count"),
        loan_total=("loanamount", np.sum),
        loan_avg=("loanamount", np.mean),
        # loan_median=("loanamount", np.median),
        jobs_reported_total=("jobsreported", np.sum),
        jobs_reported_avg=("jobsreported", np.mean),
        # jobs_reported_median=("jobsreported", np.median),
        lender_countD=("lender", pd.Series.nunique),
        congress_district_countD=(
            "congressionaldistrict",
            pd.Series.nunique,
        ),
    )

    df_calc_i["avg_cost_per_job"] = (
        df_calc_i["loan_total"] / df_calc_i["jobs_reported_total"]
    )

    ##df_calc_i["median_cost_per_job"] = (
    ##    df_calc_i["loan_median"] / df_calc_i["jobs_reported_avg"]
    ##)

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
df_calc[
    (df_calc["loan_count"] > 500) & (df_calc["depth"] == 1)
].sort_values("avg_cost_per_job", ascending=False).head(n=10)

#%%

df_calc[
    (df_calc["loan_count"] > 50) & (df_calc["depth"] == 2)
].sort_values("avg_cost_per_job", ascending=False).head(n=10)

#%%
df_calc[
    (df_calc["loan_count"] > 50) & (df_calc["depth"].isin([1, 2]))
].sort_values("avg_cost_per_job", ascending=False).head(n=10)

#%%
df_calc[(df_calc["loan_count"] > 75)].sort_values(
    "avg_cost_per_job", ascending=False
).head(n=10)

# %%

df_calc.info()


# %%
df_calc_viz = df_calc.copy()
df_calc_viz["depth"] = df_calc_viz["depth"].astype(str)
df_calc_viz["loan_total_log"] = np.log(df_calc_viz["loan_total"])


fig = df_calc_viz[df_calc_viz["loan_count"] >= 0].pipe(
    lambda df: px.histogram(
        data_frame=df,
        x="loan_avg",
        log_y=True,
        nbins=len(range(0, 4_050_000, 50_000)),
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
            (df_calc_viz["loan_count"] >= 1000)
            & df_calc_viz["depth"].isin(["1"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "How expensive was each job saved? With 1 Variable"
                + "<br>Scaled by distinct count of lenders within each group"
            ),
            x="loan_total",
            y="avg_cost_per_job",
            size="lender_countD",
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
            (df_calc_viz["loan_count"] >= 1000)
            & df_calc_viz["depth"].isin(["1", "2"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "How expensive was each job saved? With 2 Variable"
                + "<br>Scaled by distinct count of lenders within each group"
            ),
            x="loan_total",
            y="avg_cost_per_job",
            size="lender_countD",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2b

#%%
fig2b = (
    df_calc_viz[
        (
            (df_calc_viz["loan_count"] >= 1000)
            & df_calc_viz["depth"].isin(["1"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=("LINE1" + "<br>LINE2"),
            x="jobs_reported_avg",
            y="avg_cost_per_job",
            size="loan_total",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig2b

#%%
fig3a = (
    df_calc_viz[
        (
            (df_calc_viz["loan_count"] >= 500)
            & df_calc_viz["depth"].isin(["1"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Relationship b/w Cost of Saved Jobs & Number of Saved Jobs"
                + "<br>Scaled by total loan amount"
            ),
            x="jobs_reported_avg",
            y="avg_cost_per_job",
            size="loan_total",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig3a

#%%
fig3b = (
    df_calc_viz[
        (
            (df_calc_viz["loan_count"] >= 500)
            & df_calc_viz["depth"].isin(["2"])
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Relationship b/w Cost of Saved Jobs & Number of Saved Jobs"
                + "<br>Scaled by total loan amount"
            ),
            x="jobs_reported_avg",
            y="avg_cost_per_job",
            size="loan_total",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig3b

# %%
fig3c = (
    df_calc_viz[
        (
            (df_calc_viz["loan_count"] >= 500)
            & df_calc_viz["depth"].isin(["2", "3"])
            & (df_calc_viz.grouped_clean.str.contains("state: CA"))
        )
    ]
    .sort_values("depth", ascending=False)
    .pipe(
        lambda df: px.scatter(
            data_frame=df,
            title=(
                "Only CA - Relationship b/w Cost of Saved Jobs & Number of Saved Jobs"
                + "<br>Scaled by total loan amount"
            ),
            x="jobs_reported_avg",
            y="avg_cost_per_job",
            size="loan_total",
            color="grouped_vars",
            hover_name="grouped_clean",
        )
    )
)

fig3c

# %%
