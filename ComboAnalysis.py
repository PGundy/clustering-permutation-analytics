### Further notes & documentation to be added.
##
## Step by step examples in the 'if __name__ == "__main__":' section
##
### Basic Outline of code will look similar to the following:
##
## CA = ComboAnalysis()
## CA.inputColNames = df_key_vars
## CA.depthMax = 3
## CA.data = df
##
## comboAnalysisList = CA.generateCombos()
## def exampleFunction(data):
##      print(data) ## Should reflect your intended computation
##
## ComboAnalysisOutput = CA.ComputeCombos(
##         generatedCombos=CA.generateCombos(),
##         userFunc=exampleFunction
##     )
## ComboAnalysisOutput.sample(n=5)
##
## CA.getVarContents(data=ComboAnalysisOutput, varNames="color")
## CA.FilterListBroadly(data=ComboAnalysisOutput, columnToSearch="col_name", searchList="content")
##
##

#%%
import itertools

#%%
import numpy as np
import pandas as pd

#%%


class ComboAnalysis:
    inputColNames: list
    depthMax: int
    inputData: pd.DataFrame
    combinations: list
    dataCombos: pd.DataFrame

    def __init__(
        self,
        inputColNames=list(),
        depthMax=int,
        inputData=pd.DataFrame(None),
        combinations=list(),
        completedComboAnalysis=pd.DataFrame(None),
    ):
        self.inputColNames = inputColNames
        self.depthMax = depthMax
        self.inputData = inputData
        self.combinations = combinations
        self.completedComboAnalysis = completedComboAnalysis

    ## check the input values
    def checkInputs(self):
        print("Below are the currently defined inputs:")
        print(f"     inputColNames: {self.inputColNames}\n")
        print(f"     depthMax: {self.depthMax}\n\n")
        print(f"     inputData: {self.inputData.head()}\n\n")
        print(f"     combinations: {self.combinations}\n\n")
        print(
            f"     completedComboAnalysis: {self.completedComboAnalysis}"
        )

    ## get the list of alphanumerically sorted combinations
    def createCombos(self):

        if self.inputColNames is None:
            raise ValueError("inputColNames is set to None")
        if self.depthMax is None:
            raise ValueError("depthMax is set to None")

        ##define the interim objects
        combination_final = []

        for i in np.arange(0, self.depthMax, 1) + 1:
            combinations = list(
                itertools.combinations(self.inputColNames, int(i))
            )
            combinations = list(map(list, combinations))
            combinations.sort()
            ##print(
            ##    "combinations with element length",
            ##    i,
            ##    "have",
            ##    len(combinations),
            ##    "variations.",
            ##)

            ## Build up nested list of all combinations
            # combination_interim.append(combinations)
            combination_final = [*combination_final, combinations]

            if i == self.depthMax:
                ## unnest the combinations
                combination_final = list(
                    itertools.chain.from_iterable(combination_final)
                )
                print(
                    "In total there are",
                    len(combination_final),
                    "combinations with a depthMax max of",
                    self.depthMax,
                )

                self.combinations = list(filter(None, combination_final))

                return print(
                    f"The combinations are calculated using a depthMax of {self.depthMax} & the following variables:\n{self.inputColNames}\n\n\n"
                )

    ## print the combinations created from 'createCombos'
    def viewCombos(self):
        return self.combinations

    ## Iterate using userFunc & inputData using groupby(Combos)
    def runComboAnalysis(self, userFunc):
        result = []
        resultList = []
        dataCopyForLoop = self.inputData.copy()
        for step_i, grp_var_i in enumerate(self.combinations):
            print(
                "Step:",
                step_i,
                "of",
                len(self.combinations) - 1,
                " -- Grouped by:",
                grp_var_i,
            )

            ## Run the calculations for each step

            groupedData = dataCopyForLoop.groupby(grp_var_i)
            result = userFunc(groupedData)

            ## Create variables to create 'clean' columns
            result_col_names = list(result.columns)
            result.reset_index(inplace=True)

            ### Create the easy to read variables
            ## Count of interactions
            result["depth"] = len(grp_var_i)

            ## pd.Series of list(relevant variable names)
            result["grouped_vars"] = pd.Series([grp_var_i] * len(result))

            ## pd.Series of the corresponding variable's values
            result["grouped_values"] = result[grp_var_i].apply(
                lambda row: list(row.values.astype(str)), axis=1
            )

            ### Create the list zip("grouped_vars","grouped_values")
            for row in np.arange(0, len(result), 1):

                ## create empty pd.Series -- Then loop builds it
                if row == 0:
                    result["grouped_clean"] = pd.Series(dtype=str)
                ###
                ### KNOWN BUG WITH CHAINED INDEXING
                ###
                ### TODO: Is this solvable with pd.Series of lists?
                ###
                result["grouped_clean"][row] = [
                    i + ": " + j + ""
                    for i, j in zip(
                        result["grouped_vars"].iloc[row],
                        result["grouped_values"].iloc[row],
                    )
                ]

                ## Select & order our final columns
                result = result[
                    list(
                        [
                            "depth",
                            "grouped_vars",
                            "grouped_values",
                            "grouped_clean",
                        ]
                    )
                    + result_col_names
                ]

            ## Final step - build the aggregated object
            resultList.append(result)
            result = []
            resultDataFrame = pd.concat(resultList).reset_index(drop=True)
            self.completedComboAnalysis = resultDataFrame

    ## simple export of the data object
    def exportComboAnalysisData(self):
        return self.completedComboAnalysis

    ## Return all the viable variable names
    def getVarNames(self):
        temp = ComboAnalysisData[ComboAnalysisData["depth"].isin([1])][
            "grouped_vars"
        ]

        ## TODO change the below to return a df with cols: 'varNames' & 'unique_values'
        return temp.explode().value_counts()

    ## Return all rows for a variable to see the content of the variable
    def getVarContents(self, varNames):
        if isinstance(varNames, str):
            varNames = [varNames]
        temp = self.completedComboAnalysis[
            self.completedComboAnalysis["grouped_vars"].apply(
                set(varNames).issuperset
            )
        ]
        temp = temp[temp["depth"].isin([1])]
        return temp

    ## Filter to rows to any input variables
    def filterListBroadly(
        self, columnToSearch: str, searchList, data=None, depth_filter=None
    ):
        ## Are you searching the full analysis or a chosen subset?
        if data is None:
            searchData = self.completedComboAnalysis
        else:
            searchData = data

        ## required transformation for easier searching
        if isinstance(searchList, str):
            searchList = [searchList]

        ## return error if not searching a list
        if not isinstance(searchList, list):
            ValueError(
                "The object provided to search_list is not a list. Nor a string capable of being transformed into a list"
            )

        searchResult = []
        for searchTerm in searchList:
            searchResult.append(
                searchData[
                    [searchTerm in x for x in searchData[columnToSearch]]
                ]
            )
        searchResult = pd.concat(searchResult)

        ## TODO: improve depth_filter to take a list of ints
        if depth_filter is not None:
            searchResult = searchResult[
                searchResult["depth"].isin([depth_filter])
            ]
        searchResult.sort_values("depth", ascending=True, inplace=True)
        ## TODO: Is it better to reset the index or not?
        ##searchResult.reset_index(inplace=True, drop=True)
        return searchResult.copy()

    ## Filter rows in a more strict manner. Only complete matches
    def filterListStrictly(
        self, columnToSearch: str, searchList, data=None, depth_filter=None
    ):
        ## Are you searching the full analysis or a chosen subset?
        if data is None:
            searchData = self.completedComboAnalysis
        else:
            searchData = data

        ## required transformation for easier searching
        if isinstance(searchList, str):
            searchList = [searchList]

        ## return error if not searching a list
        if not isinstance(searchList, list):
            ValueError(
                "The object provided to search_list is not a list. Nor a string capable of being transformed into a list"
            )

        searchResult = []
        searchResult = self.completedComboAnalysis[
            self.completedComboAnalysis[columnToSearch].apply(
                set(searchList).issuperset
            )
        ]

        if depth_filter is not None:
            searchResult = searchResult[
                searchResult["depth"].isin([depth_filter])
            ]

        searchResult.sort_values("depth", ascending=True, inplace=True)
        ## TODO: Is it better to reset the index or not?
        ##searchResult.reset_index(inplace=True, drop=True)
        return searchResult.copy()

    def listToString(self, inputColName, data=None, delimiter=None):
        delimiter: str

        if delimiter is None:
            delimiter = " -- "

        if data is None:
            data = self.completedComboAnalysis

        return data[inputColName].apply(
            lambda x: delimiter.join(map(str, x))
        )


# %%

if __name__ == "__main__":

    import seaborn as sns

    df = sns.load_dataset("diamonds")
    df_key_vars = list(df.select_dtypes(include=["category"]).columns)

    CA = ComboAnalysis()
    CA.inputColNames = df_key_vars
    CA.depthMax = 3
    CA.inputData = df
    CA.createCombos()
    CA.viewCombos()

    ###################################
    # Below are examples evaluating the comboAnalysis
    ###################################

    ## create the user defined summary function
    ### Key things
    ### (1) 'data' is the only argument
    ### (2) this function must work for the entire dataset (or sample)
    def exampleFunction(data):
        return data.agg(
            count=pd.NamedAgg("price", "count"),
            size=pd.NamedAgg("price", "size"),
            avg_price=pd.NamedAgg("price", np.mean),
            med_price=pd.NamedAgg("price", np.median),
        )

    CA.runComboAnalysis(userFunc=exampleFunction)
    ComboAnalysisData = CA.exportComboAnalysisData()

    CA.getVarNames()

    ## Example of how to view contents of any particular variable
    CA.getVarContents(varNames="color")

    ## Return all rows where 'color' appears - 54 rows
    CA.filterListBroadly(
        columnToSearch="grouped_clean", searchList="color: H",
    )

    ## Return all rows where ONLY 'color' appears - only 1 row
    CA.filterListStrictly(
        columnToSearch="grouped_clean", searchList="color: H",
    )

    ## Return all rows with these values
    CA.filterListBroadly(
        columnToSearch="grouped_clean",
        searchList=["color: H", "color: D", "cut: Very Good"],
    )

    ## Return only rows with these values -- Note the interactions
    CA.filterListStrictly(
        columnToSearch="grouped_clean",
        searchList=["color: H", "color: D", "cut: Very Good"],
    )

    ## Example of multistep filtering
    ### Step 1 Limit to any row with 'cut'
    filter_df_step1 = CA.filterListBroadly(
        columnToSearch="grouped_clean", searchList="cut: Ideal",
    )

    ### Step 2 limit to all rows of 'color:H' from rows with 'cut'
    filter_df_step2 = CA.filterListBroadly(
        data=filter_df_step1,
        columnToSearch="grouped_clean",
        searchList="color: H",
    )

    filter_df_step2.head()


## Example of converting lists to delimited string
# CA.listToString(ComboAnalysisData["grouped_clean"].tail(5))
# CA.listToString(
#    ComboAnalysisData["grouped_clean"].tail(5), delimiter="; "
# )

#%%
