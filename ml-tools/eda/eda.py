from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from loguru import logger
from matplotlib.ticker import PercentFormatter
from pandas_profiling import ProfileReport
from scipy.stats import pearsonr


def dedup(df: pd.DataFrame, drop=False, subset=None):
    """Logs the number of duplicate rows in a dataframe and optionally drops them.

    Args:
        df (pd.DataFrame): A pandas dataframe.
        drop (bool, optional): When True, drops the duplicate rows from the dataset. Defaults to False.
        subset (list, optional): A list of column names to consider when looking for duplicates. Defaults to None.
    """
    duplicate_count = len(df) - len(df.drop_duplicates(subset=subset))
    if duplicate_count > 0:
        logger.warning(f"{duplicate_count} duplicate rows found\nTo examine them run: df[df.duplicated(keep=False)]\nTo drop them run: df.drop_duplicates(inplace=True)")
        if drop == True:
            df.drop_duplicates(subset=subset, inplace=True)
            logger.warning(f"Dropped {duplicate_count} duplicate rows")
            

def data_overview(df: pd.DataFrame):
    """Prints an overview of a dataframe including shape, df types, missing values, percentage missing values, and unique values.
    
    datetime and datetime64[ns] columns are excluded from the overview.

    Args:
        df (pd.DataFrame): A pandas dataframe.
    """

    df = df.copy()
    df = df.select_dtypes(exclude=["datetime", "datetime64[ns]"])
    # Shape of df
    print(f"Dataframe has {df.shape[0]} rows and {df.shape[1]} columns")
    print("\n")

    # Data Types and Missing Values
    print(f"Data Types, Missing Values Per Column, and Unique Values Per Column:\n")
    df_overview = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes,
        "# Missing Values": df.isnull().sum(),
        "% Missing Values": df.isnull().mean() * 100,
        "Unique Values": df.nunique()
    }).sort_values(by=["Data Type", "% Missing Values"]).reset_index(drop=True)
    display(df_overview)
    print("\n")

    # Data Types Counts
    print(f"Data Types Counts:\n")
    display(df.dtypes.value_counts())
    
    
def numeric_summary(data: pd.DataFrame):
    """Prints a summary of numeric columns in a dataframe using the describe method, and plots the distribution of numeric columns using histograms.

    Args:
        data (pd.DataFrame): A pandas dataframe.
    """

    df = data.copy()
    # Summary Statistics
    print(f"Numeric Variables Summary Statistics\n")
    display(df.describe())

    # Distributions of Numeric Columns
    print(f"Distributions of Numeric Columns:\n")
    df.hist(figsize=(12, 8), bins=50)
    plt.show()


def categorical_summary(data: pd.DataFrame):
    """Plots the distribution of categorical columns in a dataframe, and prints the 5 most common values for columns with 20 or more unique values.

    Args:
        data (pd.DataFrame): A pandas dataframe.
    """
    
    df = data.copy()
    # Count plots of categorical columns
    for col_name in df.select_dtypes(exclude="number").columns:
        nunique = df[col_name].nunique()
        desc_order = df[col_name].value_counts().index
        if nunique > 5 and nunique < 20:
            sns.countplot(df=df, y=col_name, order=desc_order)
            plt.title(f"# of Observations in {col_name}")
            plt.xlabel("# of Observations")
            plt.ylabel(col_name)
            plt.show()
        elif nunique <= 5:
            sns.countplot(df=df, x=col_name, order=desc_order)
            plt.title(f"# of Observations in {col_name}")
            plt.ylabel("# of Observations")
            plt.xlabel(col_name)
            plt.show()
        else:
            print(f"🚨 {col_name} column had 20 or more unique values, so distribution skipped. Consider binning this variable 🚨")
            print(f"Here are the 5 most common values:\n{df[col_name].value_counts().head(5)}")
            

def generate_profile_report(train_data: pd.DataFrame, path=None) -> None:
    """Generates a pandas_profiling EDA report and displays it in the notebook.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
    """
    
    # Create pandas_profiling EDA Report
    profile = ProfileReport(train_data, title="Data Profiling Report")
    # Display pandas_profiling EDA Report
    profile.to_notebook_iframe()
    
    # Optionally save pandas_profiling EDA Report to HTML
    if path:
        profile.to_file(path)
        
def identify_outliers(train_data: pd.DataFrame) -> pd.DataFrame:
    """Identifies potential outliers using the IQR method and prints the results.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
    """
    quant1 = []
    quant3 = []
    int_quartile_range = []
    outlier_counts = []
    for col_name in train_data.select_dtypes("number").columns:
        q1 = train_data[col_name].quantile(0.25)
        q3 = train_data[col_name].quantile(0.75)
        iqr = q3 - q1
        outlier_count = len(train_data[col_name][(train_data[col_name] < (q1 - 1.5 * iqr)) | (train_data[col_name] > (q3 + 1.5 * iqr))])
        outlier_counts.append(outlier_count)
        quant1.append(q1)
        quant3.append(q3)
        int_quartile_range.append(iqr)
    outlier_df = pd.DataFrame({
        "column": train_data.select_dtypes("number").columns,
        "q1": quant1,
        "q3": quant3,
        "iqr": int_quartile_range,
        "outlier_count": outlier_counts
    }).sort_values(by="outlier_count", ascending=False).reset_index(drop=True)
    print(f"\nNumber of potential outliers per column (IQR method)\n")
    print(f"To examine outliers run: train_data[((train_data[col_name] < (q1 - 1.5 * iqr)) | (train_data[col_name] > (q3 + 1.5 * iqr)))]")
    display(outlier_df)
    return outlier_df
        
        
def corr_sig(df: pd.DataFrame) -> np.array:
    """Returns a matrix of p-values for all pairwise correlations between columns in a dataframe.

    Args:
        df (pd.DataFrame): A pandas dataframe.

    Returns:
        np.array: A matrix of p-values for all pairwise correlations between columns in a dataframe.
    """
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def pearson_r_heatmap(train_data: pd.DataFrame) -> None:
    """Plots a heatmap of pearson r values for all pairwise correlations between columns in a dataframe.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
    """
    p_values = corr_sig(train_data.select_dtypes('number'))
    mask = np.invert(np.tril(p_values<0.05))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 15))

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(train_data.corr(method="pearson", numeric_only=True),
                        mask=mask,
                        annot=True,
                        fmt='.2f',
                        square=True)
    g.set(title="* A Blank Square Means the Correlation was not Statisically Significant at p-value of 0.05 *")
    f.set_tight_layout(True)
    plt.show()
    

def pairgrid_plots(train_data: pd.DataFrame, category_limit=6) -> None:
    """Plots a pairgrid of scatterplots and histograms for all numeric columns, and a pairgrid of scatterplots and histograms for all numeric columns with hue for each categorical variable.
    
    If there are over `category_limit` categories for a given categorical variable, the pairgrid with hue will not be plotted for that variable.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
    """
    # No hue
    g = sns.PairGrid(train_data, corner=True)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    plt.show()

    # With hue for each categorical variable
    for col_name in train_data.select_dtypes(exclude="number").columns:
        if train_data[col_name].nunique() < category_limit:
            g = sns.PairGrid(train_data, corner=True, hue=col_name)
            g.map_diag(sns.histplot)
            g.map_offdiag(sns.scatterplot)
            g.add_legend()
            plt.show()
        else:
            logger.warning(f"🚨 {col_name} column had {category_limit} or more unique values, so pairgrid with hue skipped 🚨")
            
            
def regression_boxplots(train_data: pd.DataFrame, target: str) -> None:
    """Plots a boxplot of the distribution of the target variable by each categorical variable.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
        target (str): The name of the target variable for the regression.
    """
    
    for col_name in train_data.select_dtypes(exclude="number").columns:
        # Medians in descending order
        desc_order = train_data.groupby(by=[col_name])[target].median().sort_values(ascending=False).index
        g = sns.boxplot(data=train_data, y=col_name, x=target, order=desc_order)
        g.set(xlabel=target, ylabel=col_name)
        g.set_title(f"{target} Distribution by {col_name}")
        plt.show()
        
def regression_two_way_interaction_boxplots(train_data: pd.DataFrame, target: str) -> None:
    """For each unique 2 way combination of categorical variables, plots a boxplot of the distribution of the target variable by each categorical variable.
    
    This is a good way to identify impacts of interactions between categorical variables and the target variable.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
        target (str): The name of the target variable for the regression.
    """
    L = train_data.select_dtypes(exclude="number").columns
    # All unqiue 2 way combinations
    unique_2_way_combos = [comb for comb in combinations(L, 2)]
    for i, j in unique_2_way_combos:
        g = sns.catplot(data=train_data, y=i, x=target, col=j, kind="box", col_wrap=3)
        g.set_axis_labels(f"{target}", i)
        g.set_titles("{col_var} = {col_name}")
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(f"{target} Distribution by {i} and {j}")
        plt.show()
        
def classification_boxplots(train_data: pd.DataFrame, target: str) -> None:
    """Plots a boxplot of the distribution of the target variable by each categorical variable.

    Args:
        train_data (pd.DataFrame): A pandas dataframe containing the training data.
        target (str): The name of the target variable for the classification.
    """
    
    for col_name in train_data.select_dtypes(include="number").columns:
        # Medians in descending order
        desc_order = train_data.groupby(by=[target])[col_name].median().sort_values(ascending=False).index
        g = sns.boxplot(data=train_data, y=target, x=col_name, order=desc_order)
        g.set(xlabel=col_name, ylabel=target)
        g.set_title(f"{col_name} Distribution by {target}")
        plt.show()


def annotate_count_plot(g: sns.FacetGrid, normalize=False):
    """Annotates the bars in a countplot with the count or percentage of observations in each bar.

    Args:
        g (sns.FacetGrid): A seaborn FacetGrid object.
        normalize (bool, optional): _description_. Defaults to False.
    """
    # Annotate the bars with the count at the top of each bar
    # iterate through axes
    if normalize == True:
        # Annotate the bars with the count at the top of each bar
        # iterate through axes
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{round(100 * v.get_height(),2)} %' for v in c]
                ax.bar_label(c, labels=labels, label_type='edge')
            ax.margins(y=0.2)
    else:
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{v.get_height()}' for v in c]
                ax.bar_label(c, labels=labels, label_type='edge')
            ax.margins(y=0.2)
            
def classification_count_plots(train_data: pd.DataFrame, target: str) -> None:
    # Compare each of the following categorical variables relationship to Survived
    small_categoricals = []
    large_categoricals = []
    for col_name in train_data.drop(columns=[target]).select_dtypes(exclude="number").columns:
        nunique = train_data[col_name].nunique()
        if nunique <= 15:
            small_categoricals.append(col_name)
        else:
            large_categoricals.append(col_name)
            
    for category in small_categoricals:
        g = sns.catplot(train_data=train_data, x=target, hue=category, kind="count")
        g.set_ylabels("Number of Observations")
        g.fig.suptitle(f"{target} by {category}")
        annotate_count_plot(g)
        plt.show()
        g = sns.catplot(train_data=train_data, x=category, y=target, hue=category, kind="bar", ci=False)
        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        g.set_ylabels("% of Observations")
        g.fig.suptitle(f"% {target} by {category}")
        annotate_count_plot(g, kind="percent")
        plt.show()
    
    logger.warning(f"🚨 The following columns had 15 or more unique values, so count plots were skipped: {large_categoricals} 🚨")
    
    
