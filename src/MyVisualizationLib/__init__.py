import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

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


def plot_pairplots(data, features, hue=None, save_folder="pairplot_images",graph_Name="pairplot"):
    """
    Plots pair plots for the specified features in the dataset. 
    Each plot is saved as a separate image file.
    
    Args:
        data (pd.DataFrame): The dataset containing the data.
        features (list): A list of feature names (strings) to include in the pair plot.
        hue (str, optional): The name of the column to use for color encoding. Default is None.
        save_folder (str, optional): The directory to save the pair plot images. Default is "pairplot_images".
        
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



def plot_heatmap(data, save_folder="heatmap_images", graph_Name="correlation_heatmap", cmap="coolwarm", figsize=(10, 8)):
    """
    Plots a heatmap of the correlation matrix for the dataset.
    The heatmap is saved as a separate image file.
    
    Args:
        data (pd.DataFrame): The dataset containing the data.
        save_folder (str, optional): The directory to save the heatmap image. Default is "heatmap_images".
        graph_Name (str, optional): The name of the output graph file. Default is "correlation_heatmap".
        cmap (str, optional): The color map to use for the heatmap. Default is "coolwarm".
        figsize (tuple, optional): The size of the figure. Default is (10, 8).
        
    Returns:
        None: The heatmap is saved as a PNG file in the specified folder.
    """
    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Calculate the correlation matrix
    corr_matrix = data.corr()

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
