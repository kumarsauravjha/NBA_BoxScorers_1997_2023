#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
url = 'https://drive.google.com/uc?export=download&id=1--R70gS8K-ye1LW3kht6_8FM9JmSZXuw'
url2 = 'https://drive.google.com/uc?export=download&id=1o00DWxJAYTU8djknN04qS0eWAXUxH8OX'


df_player = pd.read_csv(url)
df_teams = pd.read_csv(url2)

# %%
print(df_player.head().to_string())
# %%
print(df_player.shape)
# %%
############### DATA CLEANING ##############
'''Missing Values'''
# Check for missing values in each column
missing_values = df_player.isnull().sum()
print(missing_values[missing_values > 0])

#no missing value
#%%
# Convert 'date' from object to datetime
df_player['date'] = pd.to_datetime(df_player['date'])

# Convert categorical columns to 'category' data type
categorical_cols = ['type', 'team', 'home', 'away', 'win']
for col in categorical_cols:
    df_player[col] = df_player[col].astype('category')

# Example: Convert 'PTS' to integers if they are floats and not missing any value
# df_player['PTS'] = df_player['PTS'].astype(int)

#%%
'''Checking duplicates'''

print('Duplicate records \n',df_player[df_player.duplicated()])
#no duplicate
#%%
'''Verifying data types'''
print(df_player.dtypes)

#%%
# Checking for outliers in 'PTS'
# Generating subplots for the four features: PTS, MIN, REB, STL
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plotting each feature as a boxplot in a subplot
sns.boxplot(x=df_player['PTS'], ax=axes[0, 0])
axes[0, 0].set_title('Point Outliers Check')

sns.boxplot(x=df_player['MIN'], ax=axes[0, 1])
axes[0, 1].set_title('Minutes Outliers Check')

sns.boxplot(x=df_player['REB'], ax=axes[1, 0])
axes[1, 0].set_title('Rebounds Outliers Check')

sns.boxplot(x=df_player['STL'], ax=axes[1, 1])
axes[1, 1].set_title('Steals Outliers Check')

# Display the plot
plt.tight_layout()
plt.show()

#%%
# Date and other possible inconsistencies
if any(df_player['date'] > pd.to_datetime('2023-12-31')):
    print("Data contains future dates which need to be corrected.")

if df_player['home'].equals(df_player['team']):
    df_player.drop(['home', 'away'], axis=1, inplace=True)
    print("Dropped redundant columns: 'home', 'away'")

print("Updated data shape:", df_player.shape)

df_player.reset_index(drop=True, inplace=True)


#%%
# # Example for date components
# df_player['year'] = df_player['date'].dt.year
# df_player['month'] = df_player['date'].dt.month
# df_player['day'] = df_player['date'].dt.day

#%%
# Select only the numerical features from the DataFrame
numerical_features = df_player.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()

# Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))  # Increase figure size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 10})  # Adjust font size and color bar
plt.xticks(rotation=45, ha='right')  # Rotate x labels
plt.yticks(rotation=0)  # Ensure y labels are horizontal for better readability
plt.title('Heatmap of Numerical Features', fontsize=16)  # Increase title font size
plt.tight_layout()  # Fit the layout
plt.show()


#%%
#PCA
# # Selecting only the necessary features
# features = ['MIN', 'PTS', 'AST', 'FGA','3PA','FTA','FG%', '3P%', 'FT%', 'REB', 'STL', 'BLK', 'TOV', 'PF', '+/-']
# df_features = df_player[features]
#
# # Standardize the features
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df_features)
#
# # Singular Value Decomposition on the original data
# U, S, V = np.linalg.svd(df_scaled, full_matrices=False)
# condition_number_original = S[0] / S[-1]
#
# # Apply PCA
# pca = PCA(n_components=0.95)
# principalComponents = pca.fit_transform(df_scaled)
#
# # The number of components that hold 90% of the variance
# n_components = pca.n_components_
#
# # The explained variance ratio of each principal component
# explained_variance = pca.explained_variance_ratio_
#
# # SVD on the reduced data (from the covariance matrix of the reduced data)
# # Since PCA is essentially SVD on the covariance matrix, we can use the singular values from PCA
# S_reduced = pca.singular_values_
# condition_number_reduced = S_reduced[0] / S_reduced[-1]
#
# # Print results
# print("Number of components:", n_components)
# print("Explained variance ratio:", explained_variance)
# print("Original Singular Values:", S)
# print("Condition Number of Original Data:", condition_number_original)
# print("Reduced Singular Values:", S_reduced)
# print("Condition Number of Reduced Data:", condition_number_reduced)

#%%
#Number of features to be removed
# from sklearn.decomposition import PCA
#
# # Assuming 'df_features' is the DataFrame with the standardized features already computed
# pca = PCA()
# pca.fit(df_features)  # 'df_features' should be your standardized features
#
# # Determine the number of components needed for 95% explained variance
# explained_variance_ratio_cumsum = pca.explained_variance_ratio_.cumsum()
# num_components_95 = (explained_variance_ratio_cumsum < 0.95).sum() + 1
#
# # Calculate the explained variance ratio for the original feature space
# explained_variance_ratio_original = pca.explained_variance_ratio_
#
# # Fit PCA with the selected number of components
# pca_reduced = PCA(n_components=num_components_95)
# pca_reduced.fit(df_features)
#
# # Calculate the explained variance ratio for the reduced feature space
# explained_variance_ratio_reduced = pca_reduced.explained_variance_ratio_
#
# # Display the number of features to be removed and the explained variance ratios
# num_features_removed = df_features.shape[1] - num_components_95
# print("Number of features to be removed:", num_features_removed)
# print("Explained variance ratio of original feature space:", explained_variance_ratio_original)
# print("Explained variance ratio of reduced feature space:", explained_variance_ratio_reduced)

#%%
#PLOT THE CURVE
# Calculate cumulative explained variance
# # cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
#
# # Find the index where cumulative explained variance first exceeds 95%
# index_95_variance = next(i for i, cum_var in enumerate(explained_variance_ratio_cumsum ) if cum_var > 0.95)
#
# # Plot the cumulative explained variance
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance_ratio_cumsum ) + 1), explained_variance_ratio_cumsum  * 100, marker='o')
#
# # Draw dashed lines for 95% explained variance
# plt.axhline(y=95, color='black', linestyle='--')
# plt.axvline(x=index_95_variance + 1, color='red', linestyle='--')
#
# # Annotate the optimum number of features
# plt.annotate(f'Optimum number of features: {index_95_variance + 1}',
#              xy=(index_95_variance + 1, explained_variance_ratio_cumsum[index_95_variance] * 100),
#              xytext=(index_95_variance + 2, explained_variance_ratio_cumsum[index_95_variance] * 100 + 5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'))
#
# # Set labels and title
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance (%)')
# plt.title('Cumulative Explained Variance versus Number of Components')
#
# # Show the plot
# plt.grid(True)
# plt.show()

#%%
#HEATMAP OF THE REDUCED FEATURE SPACE
#
# features_reduced = pca_reduced.transform(df_scaled)
#
# # Calculate the correlation coefficient matrix for the reduced feature space
# correlation_matrix_reduced = np.corrcoef(features_reduced.T)  # Transpose for features as columns
#
# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix_reduced, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Coefficient Matrix for Reduced Feature Space')
# plt.show()
# #%%
# sns.histplot(df_player['PTS'], kde=True)
# plt.title('Distribution of Points')
# plt.show()

#%%
# Assuming df_player is your original dataframe with features and target variables.

# Selecting only the necessary features
features = ['MIN', 'PTS','FGA','FG%', '3PA','3P%','FTA', 'FT%','REB','AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']
df_features = df_player[features]

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Singular Value Decomposition on the original data to obtain original singular values
U, S_original, V = np.linalg.svd(df_scaled, full_matrices=False)
condition_number_original = S_original[0] / S_original[-1]

# Apply PCA
pca = PCA()
pca.fit(df_scaled)

S_reduced = pca.singular_values_
condition_number_reduced = S_reduced[0] / S_reduced[-1]

# Determine the number of components needed for 95% explained variance
explained_variance_ratio_cumsum = pca.explained_variance_ratio_.cumsum()
num_components_95 = np.argmax(explained_variance_ratio_cumsum >= 0.95) + 1

# Calculate the explained variance ratio for the original feature space
explained_variance_ratio_original = pca.explained_variance_ratio_

# Use the PCA fit to transform the scaled data to the reduced dimensionality
features_reduced = pca.transform(df_scaled)[:, :num_components_95]

# Calculate the correlation coefficient matrix for the reduced feature space
correlation_matrix_reduced = np.corrcoef(features_reduced.T)  # Transpose for features as columns

# Plotting the Cumulative Explained Variance to find the number of components to keep
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum * 100, marker='o')
plt.axhline(y=95, color='black', linestyle='--')
plt.axvline(x=num_components_95, color='red', linestyle='--')
plt.annotate(f'95% variance explained by {num_components_95} components',
             xy=(num_components_95, 95),
             xytext=(num_components_95 - 5, 90),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Heatmap of the Correlation Coefficient Matrix for Reduced Feature Space
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_reduced, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Coefficient Matrix for Reduced Feature Space')
plt.show()

# Histogram of Points for additional EDA
sns.histplot(df_player['PTS'], kde=True)
plt.title('Distribution of Points')
plt.show()

# Print the results
print("Original Singular Values:", S_original)
print("Condition Number of Original Data:", condition_number_original)
print("Reduced Singular Values:", S_reduced[:num_components_95])  # Only show the singular values for the retained components
print("Condition Number of Reduced Data:", condition_number_reduced)
print("Number of components to retain 95% variance:", num_components_95)
print("Number of features removed:", len(features) - num_components_95)


#%%
#NORMALITY TEST

from scipy.stats import kstest, norm

# Normality test using Kolmogorov-Smirnov test
# We use the 'norm' to compare against a normal distribution
stat, p = kstest(df_player['PTS'], 'norm')

# Interpret
alpha = 0.05
if p > alpha:
    normality = 'normal distribution (fail to reject H0)'
else:
    normality = 'not normal distribution (reject H0)'

print(f'Statistics={stat:.3f}, p={p:.3f}')
print(f'The data follows a {normality}')

#%%
#LINE-PLOT1
average_points_per_season = df_player.groupby('season')['PTS'].mean().reset_index()
plt.plot(average_points_per_season['season'], average_points_per_season['PTS'])
plt.title('Average Points per Game Over Seasons')
plt.xlabel('Season')
plt.ylabel('Average Points')
plt.grid(True)
plt.show()

#%%
#a fresh start, using the PDF of project report
# team_performance = df_player.groupby(['team', 'season'])['PTS'].mean().reset_index()
#
# # Now we will create a line plot for each team's average points across seasons.
# # We will only plot a few teams for clarity, but you can choose which teams to plot.
#
# # Let's choose a subset of teams to plot for clarity
# teams_to_plot = ['CHI', 'LAL', 'BOS', 'MIA', 'GSW']  # Example team abbreviations
# team_performance_subset = team_performance[team_performance['team'].isin(teams_to_plot)]
#
# # Creating the line plot
# plt.figure(figsize=(14, 7))
# sns.lineplot(data=team_performance_subset, x="season", y="PTS", hue="team", marker="o")
#
# # Customizing the plot
# plt.title('Average Points Scored by Team Across Seasons', fontsize=16, color='blue', fontname='serif')
# plt.xlabel('Season', fontsize=14, color='darkred', fontname='serif')
# plt.ylabel('Average Points Scored', fontsize=14, color='darkred', fontname='serif')
# plt.xticks(rotation=45)
# plt.legend(title='Team')
# plt.grid(True)
# plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
# plt.show()

#%%
#LINE-PLOT2
three_point_trends = df_player.groupby('season')['3PM'].mean().reset_index()

# Create the line plot for the league trend
plt.figure(figsize=(14, 7))
sns.lineplot(data=three_point_trends, x='season', y='3PM', marker='o')

# Customize the plot
plt.title('League-Wide Average of Three-Point Shots Made Over Seasons', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Average 3-Point Shots Made', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

#%%
#BAR PLOT GROUPED
performance_metrics = df_player.groupby('type')[['PTS', 'AST', 'REB']].mean().reset_index()

# 'Melt' the DataFrame to go from wide to long format, which is better for seaborn's barplot.
performance_melted = pd.melt(performance_metrics, id_vars='type', value_vars=['PTS', 'AST', 'REB'],
                             var_name='Metric', value_name='Average')

# Now, create the grouped bar plot with seaborn.
plt.figure(figsize=(10, 6))
sns.barplot(x='type', y='Average', hue='Metric', data=performance_melted)

# Customize the plot with titles and labels.
plt.title('Average Player Performance Metrics by Game Type')
plt.xlabel('Game Type')
plt.ylabel('Average Value')
plt.legend(title='Metric')
plt.grid(True)
plt.tight_layout()  # This will ensure the plot fits well in the figure space.
plt.show()

#%%
#BAR PLOT STACKED
# Grouping data by team and win/loss outcome and summing up the counts
team_wins_losses = df_player.groupby(['team', 'win']).size().unstack(fill_value=0)

# Plotting the stacked bar plot
team_wins_losses.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Team Wins vs. Losses')
plt.xlabel('Team')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Outcome', loc='upper right', labels=['Loss', 'Win'])
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
# Grouping data by team and win/loss outcome and summing up the counts
team_game_results = df_player.groupby(['team', 'win'])['gameid'].nunique().unstack(fill_value=0)
team_game_results.columns = ['Losses', 'Wins']

# Plotting the grouped bar plot
ax = team_game_results.plot(kind='bar', figsize=(14, 7), color=['red', 'green'])

# Adding annotations
for i, (index, row) in enumerate(team_game_results.iterrows()):
    ax.text(i - 0.15, row['Wins'] + 0.1, row['Wins'], color='black', rotation=90)
    ax.text(i + 0.05, row['Losses'] + 0.1, row['Losses'], color='black', rotation=90)

# Add titles and labels
plt.title('Total Wins and Losses per Team Across All Seasons')
plt.xlabel('Team')
plt.ylabel('Total Games')
plt.legend(title='Outcome')
plt.tight_layout()  # Adjust plot to fit into the figure area neatly
plt.show()

#%%
#COUNT PLOT
team_wins = df_player[df_player['win'] == 1].groupby(['team', 'season']).size()

# Determine the seasons where teams have won 50 or more games
team_50_wins = team_wins[team_wins >= 50].reset_index(name='Win Count')

# Count how many times each team has won 50 or more games across all seasons
team_50_wins_frequency = team_50_wins['team'].value_counts().reset_index(name='Frequency')
team_50_wins_frequency.rename(columns={'index': 'Team'}, inplace=True)

# Now create the count plot
plt.figure(figsize=(14, 7))
sns.barplot(data=team_50_wins_frequency, x='Team', y='Frequency')
plt.title('Number of Seasons with 50+ Wins per Team')
plt.xlabel('Team')
plt.ylabel('Frequency of 50+ Win Seasons')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%%
#COUNT PLOT with annotations
# Now create the count plot with annotations showing the count above each bar
plt.figure(figsize=(14, 7))
barplot = sns.barplot(data=team_50_wins_frequency, x='Team', y='Frequency', palette='viridis')

# Annotate each bar with the value of its height
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.0f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 9),
                     textcoords = 'offset points')

plt.title('Number of Seasons with 50+ Wins per Team')
plt.xlabel('Team')
plt.ylabel('Frequency of 50+ Win Seasons')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%%
#PIE-CHART
# # Calculate the total wins at home and away
# home_wins = df_player[(df_player['win'] == 1) & (df_player['team'] == df_player['home'])].shape[0]
# away_wins = df_player[(df_player['win'] == 1) & (df_player['team'] == df_player['away'])].shape[0]
#
# # Prepare data for the pie chart
# win_data = [home_wins, away_wins]
# win_labels = ['Home Wins', 'Away Wins']
# colors = ['lightblue', 'lightgreen']  # Colors for the pie sections
#
# # Create the pie chart
# plt.figure(figsize=(8, 8))
# plt.pie(win_data, labels=win_labels, autopct='%1.1f%%', startangle=140, colors=colors)
# plt.title('Proportion of Home vs. Away Game Wins')
# plt.show()
# #%%
# corr = df_player.select_dtypes(include=['int64', 'float64']).corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()
#
# #%%
home_wins = df_player[(df_player['win'] == 1) & (df_player['team'] == df_player['home'])].shape[0]
away_wins = df_player[(df_player['win'] == 1) & (df_player['team'] != df_player['home'])].shape[0]

# Prepare data for the pie chart
labels = ['Home Wins', 'Away Wins']
sizes = [home_wins, away_wins]
colors = ['#007acc', '#66cc33']
explode = (0.1, 0)  # only "explode" the 1st slice (i.e. 'Home Wins')

# Plot
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Proportion of Home vs. Away Wins')
plt.show()

#%%
#DIST PLOT
# Creating the distribution plot for minutes played
plt.figure(figsize=(12, 6))
sns.histplot(df_player['MIN'], bins=30, kde=True, color='skyblue')

# Adding titles and labels
plt.title('Distribution of Minutes Played by Players')
plt.xlabel('Minutes Played')
plt.ylabel('Density')

plt.grid(True)
plt.tight_layout()  # Adjust layout to fit all components
plt.show()

'''The histogram with a Kernel Density Estimate (KDE) overlay that you've provided shows the distribution of minutes played by players. The x-axis represents the minutes played, ranging from 0 to 60, which likely corresponds to the length of an NBA game. The y-axis indicates the density or count of player-game instances for each bin of minutes played.

Here's a simple interpretation:

- The most common range of minutes played by players is around the 10 to 30-minute mark, as indicated by the tallest bars.
- There is a significant peak around 20 minutes, suggesting a large number of players are playing around half the game.
- The distribution decreases as the minutes increase beyond 30, showing that fewer players tend to play for longer durations in a game.
- The KDE line follows the shape of the histogram, showing a smooth estimate of the distribution. It confirms that the distribution of minutes played is skewed towards lower values, with fewer instances of very high minutes played.
- There's a noticeable drop-off in density as minutes played approach 60, indicating that it's rare for players to play the entire game without rest.

This visualization helps understand the typical rotation and resting patterns in games, highlighting that most players are rotated out regularly and not utilized for the full duration of the game.'''

'''The y-axis values, which are in the tens of thousands, represent the frequency (or count) of the data points falling within each bin of the histogram. In the context of your NBA dataset:

- Each bar's height corresponds to the number of player-game instances where players played for the number of minutes indicated on the x-axis.
- A value of 10,000 on the y-axis means there are 10,000 instances in the dataset where players played for that range of minutes.
- The high values suggest that the dataset likely contains multiple entries for each player across many games and seasons, which is why the counts are so high.

So if a bar reaches up to 20,000 on the y-axis, it means that in 20,000 player-games, players played the number of minutes corresponding to the range of that particular bar on the x-axis.'''

#%%
#PAIR PLOT
# Selecting the columns of interest for the pair plot
# cols = ['FGM', 'FGA', '3PM', '3PA']
#
# # Creating the pair plot
# sns.pairplot(df_player[cols], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=3)
#
# # Adding a title to the pair plot. Since pairplot doesn't support titles directly, we use plt.subplots_adjust and plt.suptitle
# plt.subplots_adjust(top=0.9)
# plt.suptitle('Pair Plot of Shooting Efficiency and Tendencies', height=16)
#
# plt.show()

#%%
columns_of_interest = ['FGM', 'FGA', '3PM', '3PA']

# Creating the pair plot with the specified columns
sns.pairplot(df_player[columns_of_interest], diag_kind='kde', plot_kws={'alpha': 0.2, 's': 20, 'edgecolor': 'k'})

# Adjust layout for a better fit and add a main title
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle('Shooting Efficiency and Tendencies', fontsize=16)

# Show the pair plot
plt.show()
'''The pair plot you've shared displays the relationships between the numbers of field goals made (FGM), field goal attempts (FGA), three-point shots made (3PM), and three-point attempts (3PA). Here's a simple interpretation:

- The diagonal plots show the distribution for each of these variables. The shape of these distributions indicates how commonly different values occur; for example, a peak suggests a common value.
- The off-diagonal plots are scatter plots that show the relationship between pairs of these variables. A pattern in these plots, such as points clustering along a line, suggests a correlation between the variables.
- The scatter plots between FGM and FGA, as well as between 3PM and 3PA, likely show positive correlations: as attempts increase, so do the made shots.
- The plots combining FGA and 3PM or 3PA could indicate players' shooting preferences or how often players who attempt many field goals also shoot three-pointers.

Overall, this visualization helps in understanding the shooting behavior of players and how often their attempts translate into actual points, as well as their inclination towards taking three-point shots.'''
#%%
#HEATMAP with cbar
# Calculating the correlation matrix
correlation_matrix = df_player[['PTS', 'AST', 'REB', 'STL', 'BLK']].corr()

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

# Adding titles and labels
plt.title('Correlation Heatmap of Key Statistics', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)  # Keep the y-axis labels horizontal for readability
plt.show()

#%%
#HISTOGRAM plot with KDE
# Creating the histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df_player['PTS'], kde=True, bins=30, color='blue')

# Adding titles and labels
plt.title('Histogram of Player Scoring Distribution with KDE', fontsize=16)
plt.xlabel('Points Scored', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()
'''The histogram with a Kernel Density Estimate (KDE) you've provided shows the distribution of points scored by players. Here's a straightforward interpretation:

- The highest bars are clustered at the lower end of the points scored axis, indicating that it's most common for players to score a small number of points.
- The frequency of players scoring higher points progressively decreases, as seen by the decreasing height of the bars as we move right on the x-axis.
- Very high point scores, such as above 30, are quite rare—noticeable by the few and low-height bars in that range.
- The KDE curve follows the shape of the histogram and shows a peak around the lower end, confirming that lower scoring is the most frequent.
- There's a long tail extending towards the higher points scored, which suggests that while it's uncommon, there are instances where players score many points.'''

#%%
#QQ-PLOT
# Q-Q plot for the 'PTS' column
plt.figure(figsize=(10, 6))
stats.probplot(df_player['PTS'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Normality of Player Points')
plt.ylabel('Ordered Values')
plt.xlabel('Theoretical Quantiles')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
stats.probplot(df_player['MIN'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Normality of Player Minutes')
plt.ylabel('Ordered Values')
plt.xlabel('Theoretical Quantiles')
plt.grid(True)
plt.show()
'''The Q-Q plot you've provided is used to determine if a set of data follows a certain distribution—in this case, a normal distribution. The blue points represent your data's quantiles, and the red line represents the expected quantiles if the data were normally distributed.

Here's the interpretation:

- **Straight Line Match:** If the points lie on the red line, your data matches the expected normal distribution.
- **Deviation from Line:** If the points deviate from the red line, this indicates that the data does not follow a normal distribution.
- **S-shaped Curve:** In your plot, the blue points form an S-shaped curve rather than following the red line. This suggests that the distribution of player minutes is not normal.
  - The left tail (lower end) of the plot curves upwards, indicating that there are more low values in your data than you would expect in a normal distribution.
  - The right tail (higher end) curves downwards, indicating that there are more high values than would be expected in a normal distribution.
  
In simple terms, your players' minutes are not distributed normally. There are more instances of players playing very few and very many minutes than would be expected if the minutes were normally distributed. This could be due to players who rarely play (perhaps due to being on the bench) and star players who play almost the entire game, respectively.'''
#%%
#KDE PLOT
plt.figure(figsize=(10, 6))
sns.kdeplot(df_player['PTS'], fill=True, alpha=0.6, linewidth=3, color='slateblue')

# Adding titles and labels
plt.title('KDE of Player Points', fontsize=16)
plt.xlabel('Points Scored', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True)
plt.show()

#%%
#REG PLOT
plt.figure(figsize=(12, 6))
sns.regplot(x='PTS', y='AST', data=df_player, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})

# Adding titles and labels
plt.title('Regression Plot of Points Scored vs. Assists', fontsize=16)
plt.xlabel('Points Scored', fontsize=14)
plt.ylabel('Assists', fontsize=14)
plt.show()

'''The regplot you’ve provided displays the relationship between points scored and assists. The scatter points represent individual player-game instances, and the red line shows the trend or regression line, which indicates the general direction of the relationship between the two variables.

Here's a simple interpretation:

- **Positive Trend:** The upward slope of the regression line suggests a positive relationship between points scored and assists, meaning that as points increase, assists tend to increase as well.
- **Data Spread:** The scatter points seem to form a band that widens as points increase, which may indicate greater variability in assists among high-scoring games.
- **Concentration of Data Points:** There is a concentration of data points at the lower end of the points axis, indicating that many player-game instances involve scoring fewer points and correspondingly fewer assists.
- **Outliers:** There are instances at the higher end of points scored with a wide range of assists, suggesting that high scorers can have very different assist numbers.

This visualization helps to understand the typical pattern in player statistics: players who score more also tend to have more assists, but the variability in assists increases with higher points scored.'''

#%%
#BOXEN PLOT
plt.figure(figsize=(14, 7))
sns.boxenplot(x='team', y='PTS', data=df_player)

# Adding titles and labels
plt.title('Boxen Plot of Player Points Distribution by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Points Scored', fontsize=14)
plt.xticks(rotation=90)  # Rotate the x labels for better visibility
plt.show()

#%%
#AREA PLOT
df_player['date'] = pd.to_datetime(df_player['date'])

# Aggregate the total points by team over time
team_points_over_time = df_player.groupby(['date', 'team'])['PTS'].sum().unstack().fillna(0)

# Plotting the area plot for one team as an example
team = 'LAL'  # Los Angeles Lakers as an example, replace with your team abbreviation
team_points_over_time[team].cumsum().plot.area(figsize=(12, 6), alpha=0.5)

# Adding titles and labels
plt.title(f'Cumulative Points Scored by {team} Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Points Scored', fontsize=14)
plt.grid(True)
plt.show()

#%%
#VIOLIN PLOT
# Creating the violin plot for minutes played by team
plt.figure(figsize=(14, 7))
sns.violinplot(x='team', y='MIN', data=df_player, inner='quartile')

# Adding titles and labels
plt.title('Violin Plot of Minutes Played by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Minutes Played', fontsize=14)
plt.xticks(rotation=90)  # Rotate the x labels for better visibility
plt.grid(True)
plt.show()

#%%
#JOINT PLOT with KDE and scatter representation
# Sample a subset of the data if the dataset is very large to speed up plotting
sampled_data = df_player.sample(frac=0.1, random_state=1) if len(df_player) > 10000 else df_player

# Creating the joint plot with KDE and scatter representations
joint_plot = sns.jointplot(x='PTS', y='AST', data=sampled_data, kind='scatter', color='m', alpha=0.6)
joint_plot.plot_joint(sns.kdeplot, color="k", levels=5)

# Set the title of the plot
joint_plot.fig.suptitle('Joint Distribution of Points and Assists', fontsize=16)
joint_plot.fig.subplots_adjust(top=0.95)  # adjust the Figure in jointplot to provide space for the title


# Show the plot
plt.show()

#%%
#RUG PLOT
# Creating the rug plot for points scored
plt.figure(figsize=(12, 4))  # Long and narrow figure size for a rug plot
sns.rugplot(x='PTS', data=df_player, height=0.5)

# Adding titles and labels
plt.title('Rug Plot of Player Scoring Distribution', fontsize=16)
plt.xlabel('Points Scored', fontsize=14)
plt.grid(True)
plt.show()

'''Based on your description of the rug plot:

The plot shows a set of vertical lines (or "rugs") along the points scored axis, which corresponds to individual instances of player scoring in the dataset. Each rug represents a data point.

Here’s a simple interpretation:

- **Density of Rugs:** The denser clusters of rugs indicate that more scores fall within those point ranges. It looks like there's a higher concentration of scores in the lower point range, which is common in basketball scoring as not all players score high in each game.
- **Spacing Between Rugs:** Areas with wider spaces between rugs indicate fewer scores within that range. The decrease in density as the points increase suggests that high-scoring games are less common.
- **Extremes:** Rugs that appear farther out on the scale, toward higher point values, represent the exceptional high-scoring performances. These are more spread out, indicating such high scoring games are relatively rare.

The rug plot gives a visual sense of where most of the data lies and how it's spread out, which in this case is predominantly toward the lower end of the scoring range.'''


#%%
#3D PLOT and CONTOUR PLOT

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_player['PTS'], df_player['AST'], df_player['REB'], c='b', marker='o')

# Labeling the axes
ax.set_xlabel('Points Scored')
ax.set_ylabel('Assists')
ax.set_zlabel('Rebounds')

plt.title('3D Scatter Plot of PTS, AST, and REB')
plt.show()

#%%
#CLUSTER MAP
plt.figure(figsize=(14, 7))
team_stats = df_player.groupby('team')[['PTS', 'AST', 'REB']].mean()

# Creating the cluster map
sns.clustermap(team_stats, standard_scale=1)

plt.show()

'''The image you've shared appears to be a cluster map (or clustered heatmap). This type of visualization displays data values in a matrix format, where the color intensity represents the magnitude of the statistic, and the rows and columns are ordered based on a hierarchical clustering algorithm. Here's how to interpret the cluster map:

- **Color Gradient:** The colors on the heatmap range from dark to light, corresponding to the value of the statistic, with darker colors typically indicating higher values and lighter colors indicating lower values. The color bar (legend) on the left side shows the scale of values.
  
- **Rows:** The rows of the heatmap represent different NBA teams, which seem to be clustered based on similarity in the statistics shown. Teams that are close to each other on the dendrogram (tree-like structure to the left) have similar statistical profiles.
  
- **Columns:** The columns represent specific basketball statistics, such as assists (AST), points (PTS), and rebounds (REB). The clustering at the top indicates which statistics are more similar across all teams.

- **Clusters:** The dendrograms (the branching structures) on the sides of the heatmap show how teams (on the right) and statistics (on the top) are grouped into clusters. The length of the branches in the dendrogram reflects how different the clusters are from one another, with longer branches indicating greater differences.

- **Interpretation:** Without direct access to the plot, specific interpretations are limited, but in general, teams that are grouped together in rows have similar performance metrics, and metrics that are grouped together in columns behave similarly across teams. For example, if two teams are close together on the row dendrogram, they might have similar points, assists, and rebounds profiles. If two statistics are close on the column dendrogram, they may have a related pattern across the teams.

This heatmap can provide insights into which teams have similar styles or effectiveness in gameplay based on the metrics plotted.'''

#%%
#HEXBIN PLOT
plt.figure(figsize=(10, 6))
plt.hexbin(df_player['PTS'], df_player['AST'], gridsize=50, cmap='Blues', edgecolors='gray')
plt.colorbar(label='Density')
plt.title('Hexbin Plot of Player Points vs. Assists', fontsize=16)
plt.xlabel('Points Scored', fontsize=14)
plt.ylabel('Assists', fontsize=14)
plt.xlim(0, max(df_player['PTS']))  # Adjust x-axis limits if necessary
plt.ylim(0, max(df_player['AST']))  # Adjust y-axis limits if necessary
plt.show()

'''From your description of the hexbin plot, it appears to show a concentration of values in the lower end for both points scored and assists, indicated by the darker colored hexagons. The hexagons become sparser and lighter as the values increase, showing that higher points and assists are less common.

In simple terms:

- The majority of player-game instances involve scoring few points and making few assists, as shown by the denser, darker hexagons at the bottom left of the plot.
- High-scoring games or games with many assists are rare, as indicated by the fewer, lighter hexagons in the upper-right portion of the plot.
- The plot reveals that instances of very high points scored or assists are outliers in this dataset.

The added grid of hexagons with outlines helps distinguish between areas of different densities, and the color bar on the right provides a reference for the density levels represented by the colors. The density is highest where the hexagons are darkest and decreases as the hexagons become lighter.'''

#%%
#STRIP PLOT
# Creating the strip plot for points scored by team
plt.figure(figsize=(14, 7))
sns.stripplot(x='team', y='PTS', data=df_player, jitter=True, dodge=True)

# Adding titles and labels
plt.title('Strip Plot of Player Points by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Points Scored', fontsize=14)
plt.xticks(rotation=90)  # Rotate the x labels for better visibility
plt.show()

'''The strip plot you’re describing shows individual player points by team. Each dot represents a player’s points scored in a game, aligned with the team they were playing for.

Here’s a straightforward interpretation:

- **Clusters of Points:** The vertical lines of dots for each team indicate the number of points scored by players in different games. Where you see a lot of dots clustered around a certain point value, it means many players scored that number of points while playing for that team.
  
- **Variability:** Teams with dots spread out over a wider range of the y-axis indicate greater variability in the number of points scored by players. A tight cluster would suggest more consistency in scoring.
  
- **Outliers:** Any dots that appear far away from the main clusters could be considered outliers, representing unusually high or low scoring performances compared to the rest.

The plot provides a quick way to compare scoring between teams and to see the distribution and range of player points within each team.'''

#%%
#SWARM PLOT

# Creating the swarm plot for points scored by team
# plt.figure(figsize=(14, 7))
# sns.swarmplot(x='team', y='PTS', data=df_player)
#
# # Adding titles and labels
# plt.title('Swarm Plot of Player Points by Team', fontsize=16)
# plt.xlabel('Team', fontsize=14)
# plt.ylabel('Points Scored', fontsize=14)
# plt.xticks(rotation=90)  # Rotate the x labels for better visibility
# plt.show()
plt.figure(figsize=(16, 10))  # Increase the figure size
sample_data = df_player.sample(n=1000)  # Adjust n to fit your needs
swarm_plot = sns.swarmplot(x='team', y='PTS', data=sample_data)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Optionally, reduce the font size of the labels if they're still too large
swarm_plot.set_xticklabels(swarm_plot.get_xticklabels(), fontsize=8)

plt.title('Swarm Plot of Player Points by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Points Scored', fontsize=14)

plt.tight_layout()  # Adjust layout to fit all components
plt.show()

#########################################################################
#%%
#SUBPLOTS
#1ST ONE

# First, we'll calculate the average of the stats per season.
seasonal_stats = df_player.groupby('season')[['PTS', 'AST', 'REB', 'STL']].mean().reset_index()

# Now, let's create subplots for each statistic.
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plotting average points per game over seasons
sns.lineplot(x='season', y='PTS', data=seasonal_stats, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Average Points per Game Over Seasons')
axs[0, 0].set_xlabel('Season')
axs[0, 0].set_ylabel('Points')

# Plotting average assists per game over seasons
sns.lineplot(x='season', y='AST', data=seasonal_stats, ax=axs[0, 1], color='green')
axs[0, 1].set_title('Average Assists per Game Over Seasons')
axs[0, 1].set_xlabel('Season')
axs[0, 1].set_ylabel('Assists')

# Plotting average rebounds per game over seasons
sns.lineplot(x='season', y='REB', data=seasonal_stats, ax=axs[1, 0], color='red')
axs[1, 0].set_title('Average Rebounds per Game Over Seasons')
axs[1, 0].set_xlabel('Season')
axs[1, 0].set_ylabel('Rebounds')

# Plotting average steals per game over seasons
sns.lineplot(x='season', y='STL', data=seasonal_stats, ax=axs[1, 1], color='purple')
axs[1, 1].set_title('Average Steals per Game Over Seasons')
axs[1, 1].set_xlabel('Season')
axs[1, 1].set_ylabel('Steals')

# Adjust the subplots layout for a better fit
plt.tight_layout()
plt.show()

#%%
# Ensure that 'season' is a categorical type for proper ordering in the plots
df_player['season'] = df_player['season'].astype('category')

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Average points per game over seasons
sns.lineplot(data=df_player, x='season', y='PTS', ax=axs[0, 0], estimator='mean')
axs[0, 0].set_title('Average Points per Game Over Seasons')
axs[0, 0].set_ylabel('Average Points')

# Average assists per game over seasons
sns.lineplot(data=df_player, x='season', y='AST', ax=axs[0, 1], estimator='mean')
axs[0, 1].set_title('Average Assists per Game Over Seasons')
axs[0, 1].set_ylabel('Average Assists')

# Average rebounds per game over seasons
sns.lineplot(data=df_player, x='season', y='REB', ax=axs[1, 0], estimator='mean')
axs[1, 0].set_title('Average Rebounds per Game Over Seasons')
axs[1, 0].set_ylabel('Average Rebounds')

# Average steals per game over seasons
sns.lineplot(data=df_player, x='season', y='STL', ax=axs[1, 1], estimator='mean')
axs[1, 1].set_title('Average Steals per Game Over Seasons')
axs[1, 1].set_ylabel('Average Steals')

# Rotate the x-axis labels for readability
for ax in axs.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

# Adjust the layout
plt.tight_layout()
plt.show()

#%%
#2ND ONE
# Setting the size for our plot
plt.figure(figsize=(15, 10))

# 2x2 grid, first plot: Points Scored in Wins vs. Losses
plt.subplot(2, 2, 1)  # (rows, columns, panel number)
sns.boxplot(x='win', y='PTS', data=df_player)
plt.title('Points Scored in Wins vs. Losses')
plt.xlabel('Game Outcome')
plt.ylabel('Points Scored')

# Second plot: Assists in Wins vs. Losses
plt.subplot(2, 2, 2)  # (rows, columns, panel number)
sns.boxplot(x='win', y='AST', data=df_player)
plt.title('Assists in Wins vs. Losses')
plt.xlabel('Game Outcome')
plt.ylabel('Assists')

# Third plot: Rebounds in Wins vs. Losses
plt.subplot(2, 2, 3)  # (rows, columns, panel number)
sns.boxplot(x='win', y='REB', data=df_player)
plt.title('Rebounds in Wins vs. Losses')
plt.xlabel('Game Outcome')
plt.ylabel('Rebounds')

# Fourth plot: Turnovers in Wins vs. Losses
plt.subplot(2, 2, 4)  # (rows, columns, panel number)
sns.boxplot(x='win', y='TOV', data=df_player)
plt.title('Turnovers in Wins vs. Losses')
plt.xlabel('Game Outcome')
plt.ylabel('Turnovers')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

'''The image shows a 2x2 grid of box plots illustrating the distribution of four different basketball statistics—points scored, assists, rebounds, and turnovers—comparing their values in the context of game outcomes, wins versus losses.

- **Points Scored in Wins vs. Losses (Top Left):**
  The box plot suggests that more points are scored during wins (1) than losses (0), as indicated by the higher median and greater distribution in the 'win' category. The presence of outliers on both sides indicates that there are some games with exceptionally high or low points scored, regardless of the outcome.

- **Assists in Wins vs. Losses (Top Right):**
  Similar to points scored, there are typically more assists in games that are won than in games that are lost. The median is higher in wins, and there's a wider spread of data points, showing a more varied distribution of assists in victories.

- **Rebounds in Wins vs. Losses (Bottom Left):**
  Rebounds also show a trend of being higher in games that are won. The median value for rebounds is higher for wins, and the range of rebounds is wider, reflecting a greater variation in the number of rebounds players get during winning games.

- **Turnovers in Wins vs. Losses (Bottom Right):**
  In contrast to the other statistics, turnovers do not show a significant difference between wins and losses. The medians appear close, suggesting that turnovers may not fluctuate as dramatically between wins and losses compared to other stats like points, assists, and rebounds.

The overall interpretation is that winning games tend to have higher points, assists, and rebounds, suggesting these factors contribute positively to winning. Turnovers don't show a clear pattern, indicating that the relationship between turnovers and winning may be less direct or influenced by other factors not displayed here.'''


#%%
# Points by team
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_player, x='team', y='PTS')
plt.title('Points Distribution by Team')
plt.xticks(rotation=90)
plt.show()

#%%
# Top 10 players by points
plt.figure(figsize=(10, 6))
top_players_by_points = df_player.groupby('player')['PTS'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_players_by_points.values, y=top_players_by_points.index)
plt.title('Top 10 Players by Average Points')
plt.xlabel('Average Points')
plt.grid(True)
plt.show()

#%%
# Win/Loss comparison
plt.figure(figsize=(14, 7))
sns.barplot(data=df_player, x='win', y='PTS', estimator=sum)
plt.title('Total Points in Wins vs. Losses')
plt.show()

#%%
#team analysis
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_teams, x='team', y='PTS')
plt.title('Points Distribution by Team - 2nd dataset')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#%%
#now using plotly we'll plot the graphs
# Plotting the average points per game over the seasons
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_player, x="season", y="PTS", estimator='mean')
plt.title('Average Points per Game Over Seasons')
plt.xticks(rotation=45)
plt.ylabel('Average Points')
plt.xlabel('Season')
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

#%%
sns.lineplot(data=df_player, x="season", y="MIN", estimator='mean')
plt.title('Average Minutes Played Over Seasons')
plt.xticks(rotation=45)
plt.xlabel('Season')
plt.ylabel('Average Minutes')
plt.grid()
plt.tight_layout()
plt.show()

#%%
sns.boxplot(x="type", y="PTS", data=df_player)
plt.title('Player Points Distribution by Game Type')
plt.xlabel('Game Type')
plt.ylabel('Points Scored')
plt.grid()
plt.tight_layout()
plt.show()

'''Based on the box plot you've provided, which shows the distribution of player points by game type, here's an analysis of the plot:

1. **Game Types:** There are three types of games depicted—playin, playoff, and regular. These likely represent different stages or types of matches in the NBA season.

2. **Median Points Scored:** The median points scored (indicated by the line in the middle of each box) appears to be fairly consistent across all game types, with playoff games having a slightly higher median than regular and playin games.

3. **Spread of Points Scored (Interquartile Range):** The interquartile range (IQR), represented by the height of each box, seems to be similar for playin and regular games, suggesting a similar variability in points scored during these games. Playoff games have a slightly larger IQR, indicating greater variability in how many points players score during these games.

4. **Outliers:** There are many outliers in each category, particularly in the playoff and regular games, indicated by the points above the upper whiskers. These outliers represent games where players scored significantly more points than usual. The presence of more outliers in playoff games suggests that there are more instances of high-scoring performances, which could be due to the high stakes of playoff matches prompting stronger performances from players.

5. **Range of Points Scored:** The overall range of points (from the minimum to the maximum, including outliers) is largest in playoff games, followed by regular games, and then playin games. This range is indicated by the vertical lines extending from the boxes (whiskers) and the outliers. Playoff games show some of the highest points scored, with outliers reaching the highest values among all game types.

6. **Potential Skewness:** The distribution for playoff games might be slightly skewed towards higher values, given the cluster of outliers above the upper whisker, and the median being closer to the bottom of the box.

7. **Comparison of Playin and Regular Games:** The playin games have the smallest spread and fewer outliers compared to regular games, which might indicate that these games are typically lower scoring or more consistent in scoring.

**Conclusions and Further Analysis:**

- **High-Stakes Impact:** Players might be scoring more points during playoff games, possibly due to the high stakes or because the best-performing teams (with potentially better scorers) are the ones that make it to the playoffs.
- **Consistency:** Regular and playin games show a more consistent scoring pattern, which could be attributed to a wider range of teams playing, including both high and low-scoring teams.
- **Further Analysis:** To understand these trends better, one could look at the context of the games, the players involved, and the team dynamics during the different types of games. Additionally, comparing these distributions with other performance metrics like assists, rebounds, and player efficiency could yield a more comprehensive understanding of player performances.

The analysis suggests that while the typical performance in terms of points scored does not vary widely across game types, the playoffs tend to have more variability and instances of high scoring, which could be indicative of more competitive play or a focus on offensive strategies.'''

#%%
# Assuming 'gameid' is a column that uniquely identifies each game
games_by_season = df_player.groupby('season')['gameid'].nunique()

# Now plot the number of unique games per season
games_by_season.plot(kind='bar', title='Number of Games by Season')
plt.xlabel('Season')
plt.ylabel('Unique Game Count')
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 11))
# Grouping data by season and counting unique game IDs
games_by_season = df_player.groupby('season')['gameid'].nunique()

# Now plot the number of unique games per season
ax = games_by_season.plot(kind='bar', title='Number of Games by Season')
plt.xlabel('Season')
plt.ylabel('Unique Game Count')
plt.xticks(rotation=90)
plt.grid()

# Adding text on top of each bar
for i, count in enumerate(games_by_season):
    ax.text(i, count + 20, str(count), ha='center', rotation=90)

# Adjusting the position of the title
# ax.set_title('Number of Games by Season', pad=6, y=1.05)  # Increase the pad and adjust y position


plt.tight_layout()
plt.show()

#%%
# Grouping data by season and summing up the number of 3-pointers made
three_pointers_by_season = df_player.groupby('season')['3PM'].sum()

# Plotting the total number of 3-pointers made every season
plt.figure(figsize=(10, 6))
plt.plot(three_pointers_by_season.index, three_pointers_by_season.values, marker='o', linestyle='-')
plt.title('Total Number of 3-Pointers Made Every Season')
plt.xlabel('Season')
plt.ylabel('Total 3-Pointers Made')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# Grouping data by season and finding the player with the most points for each season
player_most_points_by_season = df_player.groupby('season').apply(lambda x: x.loc[x['PTS'].idxmax()])

# Plotting the player with the most points for each season
plt.figure(figsize=(10, 6))
bars = plt.bar(player_most_points_by_season.index, player_most_points_by_season['PTS'], color='skyblue')

# Adding player names as legend
for i, bar in enumerate(bars):
    season = player_most_points_by_season.index[i]
    player_name = player_most_points_by_season.loc[season, 'player']
    bar.set_label(player_name)

plt.title('Player with Most Points for Each Season')
plt.xlabel('Season')
plt.ylabel('Points')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#%%
'''T-TEST'''
import numpy as np
from scipy.stats import ttest_ind

# Create arrays of wins for home and away games
home_wins = np.array([1 if (win == 1 and team == home) else 0 for win, team, home in zip(df_player['win'], df_player['team'], df_player['home'])])
away_wins = np.array([1 if (win == 1 and team != home) else 0 for win, team, home in zip(df_player['win'], df_player['team'], df_player['home'])])

# Perform t-test
t_stat, p_val = ttest_ind(home_wins, away_wins)

print(f'T-test Statistic: {t_stat}')
print(f'P-value: {p_val}')

# Interpretation of the result
alpha = 0.05
if p_val < alpha:
    print("We reject the null hypothesis - there is a significant difference between home and away wins.")
else:
    print("We fail to reject the null hypothesis - there is no significant difference between home and away wins.")

#%%
# Calculate Points Per Game for each team
team_ppg = df_teams.groupby('team')['PTS'].mean().sort_values(ascending=False)

# Get the top 10 teams with the highest average points per game
top_10_teams_ppg = team_ppg.tail(10)

# Create a horizontal bar plot using seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(x=top_10_teams_ppg.values, y=top_10_teams_ppg.index, orient='h')

# Set plot title and axis labels
plt.title('Top 10 NBA Teams with Highest Average Points Per Game')
plt.xlabel('Average Points Per Game')
plt.ylabel('Team')

# Show the plot
plt.show()


#%%
# Calculate the average for the relevant columns and then sort the values to get the top 5
top_avg_points = df_player.groupby('player')['PTS'].mean().sort_values(ascending=False).head(5)
top_avg_assists = df_player.groupby('player')['AST'].mean().sort_values(ascending=False).head(5)
top_avg_rebounds = df_player.groupby('player')['REB'].mean().sort_values(ascending=False).head(5)
top_avg_minutes = df_player.groupby('player')['MIN'].mean().sort_values(ascending=False).head(5)

# Creating subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plotting the top 5 players for each statistic
top_avg_points.plot(kind='bar', ax=axes[0,0], color='blue')
axes[0,0].set_title('Top 5 Players by Avg Points')
axes[0,0].set_ylabel('Average Points')

top_avg_assists.plot(kind='bar', ax=axes[0,1], color='green')
axes[0,1].set_title('Top 5 Players by Avg Assists')
axes[0,1].set_ylabel('Average Assists')

top_avg_rebounds.plot(kind='bar', ax=axes[1,0], color='red')
axes[1,0].set_title('Top 5 Players by Avg Rebounds')
axes[1,0].set_ylabel('Average Rebounds')

top_avg_minutes.plot(kind='bar', ax=axes[1,1], color='purple')
axes[1,1].set_title('Top 5 Players by Avg Minutes')
axes[1,1].set_ylabel('Average Minutes')

# Improving layout and aesthetics
plt.suptitle("Legends on the Game")
plt.tight_layout()
plt.show()

#%%
from tabulate import tabulate
# Calculate Points Per Game
team_ppg = df_teams.groupby('team')['PTS'].sum() / df_teams.groupby('team')['gameid'].nunique()

# Calculate Total Number of Wins
team_wins = df_teams[df_teams['win'] == 1].groupby('team').size()

# Calculate Playoff Qualifications
team_playoff_years = df_teams[df_teams['type'] == 'playoff'].groupby('team')['season'].nunique()

# Calculate Total Number of Games Played by the Team
team_games = df_teams.groupby('team')['gameid'].nunique()

# Create a new DataFrame with all metrics
team_metrics = pd.DataFrame({
    'PPG': team_ppg,
    'Total_Wins': team_wins,
    'Playoff_Qualifications': team_playoff_years,
    'Total_Games': team_games  # Added metric
}).reset_index()

# Fill NaN values with 0 for teams that never won or didn't qualify for playoffs
team_metrics['Total_Wins'].fillna(0, inplace=True)
team_metrics['Playoff_Qualifications'].fillna(0, inplace=True)

# Print the DataFrame to verifytable = tabulate(team_metrics, headers='keys', tablefmt='psql')
table = tabulate(team_metrics, headers='keys', tablefmt='psql')

# Print the table
print(table)





