import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np
import pandas as pd

def plot_boxplots(df, features, save_folder="Milestone 1/boxplot_images", figsize=(8, 10), palette="coolwarm"):
    """
    Plots boxplots for the specified numerical features in a DataFrame and saves each plot to a specified subfolder.
    Displays the distribution and identifies potential outliers for each feature.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        features (list): A list of column names (features) to plot as boxplots.
        save_folder (str, optional): The folder to save the plots. Supports nested folders. Default is "Milestone 1/boxplot_images".
        figsize (tuple, optional): The size of the plot. Default is (8,10).
        palette (str, optional): The color palette for the boxplots. Default is "coolwarm".

    Returns:
        None: Saves each boxplot to the specified folder.
    """
    # Create nested folder path if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Plot and save boxplots
    for feature in features:
        plt.figure(figsize=figsize)
        sns.boxplot(y=df[feature], width=0.4, palette=palette)
        plt.grid(alpha=0.3)
        plt.title(feature, fontsize=12, fontweight="bold")
        plt.xticks([])

        # Save plot
        plot_path = os.path.join(save_folder, f"{feature}_boxplot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    print(f"Boxplots saved in '{save_folder}'")


def plot_histograms(data, features, colors=None, save_folder="Milestone 1/histogram_images", figsize=(10, 6)):
    """
    Plots histograms with KDE for the specified features in the dataset and saves each plot to a specified subfolder.
    Displays the distribution for each feature and optionally assigns custom or random colors.
    
    Args:
        data (pd.DataFrame): The dataset containing the data.
        features (list): A list of feature names (strings) to plot histograms for.
        colors (list, optional): A list of colors for each feature. If None, random colors will be generated.
        save_folder (str, optional): The folder to save the histogram plots. Default is "Milestone 1/histogram_images".
        figsize (tuple, optional): The size of each plot. Default is (10, 6).
        
    Returns:
        None: Saves each histogram as a PNG file in the specified folder.
    """
    # Standardize column names to lowercase for case-insensitive matching
    data.columns = map(str.lower, data.columns)

    # Convert the feature names to lowercase for consistent comparison
    features_lower = [feature.lower() for feature in features]

    # Generate random colors if none are provided
    if colors is None:
        colors = [
            "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(len(features_lower))
        ]

    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Loop through each feature and plot its histogram with KDE
    for feature in features_lower:
        if feature in data.columns:
            plt.figure(figsize=figsize)
            color = colors[features_lower.index(feature)]  # Select the color for the current feature
            
            # Plot the histogram with KDE
            sns.histplot(data[feature], bins=10, kde=True, color=color)
            plt.title(feature.capitalize(), fontsize=12, fontweight="bold")  # Set the title
            plt.xticks(rotation=30)  # Rotate x-axis labels
            plt.yticks(rotation=30)  # Rotate y-axis labels

            # Save the plot as a PNG file
            plot_path = os.path.join(save_folder, f"{feature}_histogram.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to avoid overlap with the next one
        else:
            print(f"Warning: Column '{feature}' is missing from the dataset. Skipping this feature.")

    print(f"Histograms saved in '{save_folder}'")


def plot_pairplots(data, features, hue=None, save_folder="pairplot_images", graph_Name="pairplot"):
    """
    Plots pair plots for the specified features in the dataset. 
    Each plot is saved as a separate image file.
    
    Args:
        data (pd.DataFrame): The dataset containing the data.
        features (list): A list of feature names (strings) to include in the pair plot.
        hue (str, optional): The name of the column to use for color encoding. Default is None.
        save_folder (str, optional): The directory to save the pair plot images. Default is "pairplot_images".
        graph_Name (str, optional): The name of the saved graph file. Default is "pairplot".
        
    Returns:
        None: The pair plot is saved as a PNG file in the specified folder.
    """
    # Standardize column names to lowercase for case-insensitive matching
    data.columns = map(str.lower, data.columns)

    # Convert the feature names to lowercase for consistent comparison
    features_lower = [feature.lower() for feature in features]

    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Filter the dataset to include only the specified features
    data_filtered = data[features_lower]

    # Create the pair plot using seaborn
    pair_plot = sns.pairplot(data_filtered, hue=hue)

    # Set the title of the plot
    pair_plot.fig.suptitle("Pair Plot of Selected Features", fontsize=16, fontweight="bold")
    
    # Adjust layout to make room for the title
    pair_plot.fig.tight_layout()
    pair_plot.fig.subplots_adjust(top=0.95)  # Adjust the top space for title

    # Save the plot as a PNG file
    plot_path = os.path.join(save_folder, f"{graph_Name}.png")
    pair_plot.savefig(plot_path)
    plt.close()  # Close the plot to avoid overlap with the next one

    print(f"Pair plot has been saved to the '{save_folder}' folder.")

def plot_heatmap(data, features, save_folder="heatmap_images", graph_Name="correlation_heatmap", cmap="coolwarm", figsize=(10, 8)):
    """
    Plots a heatmap of the correlation matrix for the selected features in the dataset.
    The heatmap is saved as a separate image file.
    
    Args:
        data (pd.DataFrame): The dataset containing the data.
        features (list): A list of feature names to include in the correlation matrix.
        save_folder (str, optional): The directory to save the heatmap image. Default is "heatmap_images".
        graph_Name (str, optional): The name of the output graph file. Default is "correlation_heatmap".
        cmap (str, optional): The color map to use for the heatmap. Default is "coolwarm".
        figsize (tuple, optional): The size of the figure. Default is (10, 8).
        
    Returns:
        None: The heatmap is saved as a PNG file in the specified folder.
    """
    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Filter the data to only include the specified features
    data_filtered = data[features]

    # Calculate the correlation matrix
    corr_matrix = data_filtered.corr()

    # Create the heatmap
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", cbar=True, 
                          square=True, linewidths=0.5, linecolor='black', vmin=-1, vmax=1)

    # Set title and adjust layout
    plt.title("Correlation Heatmap", fontsize=16, fontweight="bold")
    
    # Save the plot as a PNG file
    plot_path = os.path.join(save_folder, f"{graph_Name}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid overlap with the next one

    print(f"Correlation heatmap has been saved to the '{save_folder}' folder.")

def plot_model_performance(y_true, y_pred, task_type='regression', save_folder="model_performance"):
    """
    Creates comprehensive model performance visualizations.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - task_type: Type of task ('regression' or 'classification')
    - save_folder: Directory to save the plots
    """
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(save_folder, exist_ok=True)
    
    if task_type == 'regression':
        # Residual plot
        plt.figure(figsize=(10, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig(os.path.join(save_folder, 'residual_plot.png'))
        plt.close()
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.savefig(os.path.join(save_folder, 'actual_vs_predicted.png'))
        plt.close()
        
    elif task_type == 'classification':
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))
        plt.close()
        
        # ROC curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_folder, 'roc_curve.png'))
            plt.close()

def plot_feature_importance(model, feature_names, save_folder="feature_importance"):
    """
    Plots feature importance for tree-based models.
    
    Parameters:
    - model: Trained model
    - feature_names: List of feature names
    - save_folder: Directory to save the plot
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature importance or coefficients")
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'feature_importance.png'))
    plt.close()

def plot_correlation_heatmap(df, save_folder="correlation", figsize=(12, 10)):
    """
    Creates an enhanced correlation heatmap with annotations.
    
    Parameters:
    - df: DataFrame
    - save_folder: Directory to save the plot
    - figsize: Figure size
    """
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt='.2f')
    
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'correlation_heatmap.png'))
    plt.close()

def plot_time_series(data, date_column, value_column, save_folder="time_series"):
    """
    Creates time series plots with trend and seasonality.
    
    Parameters:
    - data: DataFrame with time series data
    - date_column: Name of the date column
    - value_column: Name of the value column
    - save_folder: Directory to save the plots
    """
    # Convert date column to datetime if not already
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Set date as index
    data = data.set_index(date_column)
    
    # Time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(data[value_column])
    plt.title(f'Time Series of {value_column}')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'time_series.png'))
    plt.close()
    
    # Monthly average plot
    monthly_avg = data[value_column].resample('M').mean()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg)
    plt.title(f'Monthly Average of {value_column}')
    plt.xlabel('Date')
    plt.ylabel(f'Average {value_column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'monthly_average.png'))
    plt.close()
    
    # Seasonal decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(data[value_column], period=12)
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'seasonal_decomposition.png'))
    plt.close()
