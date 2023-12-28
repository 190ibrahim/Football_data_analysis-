import pandas as pd
from sqlalchemy import create_engine
import pandasql as psql
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import zipfile


directory = './Data.zip'

data = {}
# Open the ZIP file and extract its contents
with zipfile.ZipFile(directory, 'r') as zip_ref:
    zip_ref.extractall('./Data')
    # Loop through the extracted files
    for filename in zip_ref.namelist():
        full_path = os.path.join('./Data', filename)
        if filename.endswith('.csv'):
            data[filename[:-4]] = pd.read_csv(full_path)



for key in data:
    print('**'+key+'**')
    print('INFO')
    print('\n-----------------------')
    print(data[str(key)].info())
    print('\n-----------------------')
    print('DESCRIBE')
    print(data[str(key)].describe())
    print('\n-----------------------\n')

# print(data['players'].info())
# date_of_birth                           30255 non-null object
# sub_position                            30130 non-null object
# foot                                    27913 non-null object
data['players']['date_of_birth'] = pd.to_datetime(data['players']['date_of_birth'], format = '%Y-%m-%d')
#  'sub_position' and 'foot', have a limited number of unique
#  values that repeat often. They can offer advantages in terms of memory and performance.
data['players']['sub_position'] = data['players']['sub_position'].astype('category')
data['players']['foot'] = data['players']['foot'].astype('category')
# print(data['players'].info())


# print(data['competitions']['sub_type'].unique())
# print(data['competitions']['type'].unique())
data['competitions']['sub_type'] = data['competitions']['sub_type'].astype('category')
data['competitions']['type'] = data['competitions']['type'].astype('category')
# data['competitions'].info()


data['games']['date'] = pd.to_datetime(data['games']['date'], format = '%Y-%m-%d')
data['games']['competition_type'] = data['games']['competition_type'].astype('category')


data['game_events']['type'] = data['game_events']['type'].astype('category')


data['club_games']['is_win'].unique()
data['club_games']['hosting'].unique()
data['club_games']['hosting'] = data['club_games']['hosting'].astype('category')
data['club_games']['is_win'] = data['club_games']['is_win'].astype('bool')
# print(pd.DataFrame(data['club_games'][['hosting','is_win']]).head())

data['player_valuations']['datetime'] = pd.to_datetime(data['player_valuations']['datetime'], format = '%Y-%m-%d %H:%M:%S')
data['player_valuations']['date'] = pd.to_datetime(data['player_valuations']['date'], format = '%Y-%m-%d')
data['player_valuations']['dateweek'] = pd.to_datetime(data['player_valuations']['dateweek'], format = '%Y-%m-%d')

data['appearances']['date'] = pd.to_datetime(data['appearances']['date'],format = '%Y-%m-%d')



def check_duplicated(data):
    for key in data:
        df = data[str(key)]
        print('**'+key+'**')
        print('\n-----------------------')
        print(df[df.duplicated()])
        print('\n-----------------------')
def check_null(data):
        for key in data:
            df = data[str(key)]
            print(key)
            print('\n--------------')
            print(df.isnull().sum())
            print('\n--------------')


check_duplicated(data)
check_null(data)



# handling missing values by dropping columns that have a substantial amount of missing data. 
# This action might be taken because these columns might not be crucial for the analysis you're planning to perform.
# Dropping columns with a significant number of missing values can sometimes simplify the dataset and make further analysis more manageable.
# print(data['players'].isnull().sum())
# first_name                               1965
# country_of_birth                         2689
# city_of_birth                            2203
# market_value_in_eur                     10919
# contract_expiration_date                11467
# agent_name                              15361
data['players'] = data['players'].drop(['first_name', 'last_name','country_of_birth','city_of_birth','market_value_in_eur','contract_expiration_date','agent_name'], axis=1)
# print(data['players'].head())


# Creating a subset 'br' by querying the 'players' DataFrame for specific columns 
# where certain columns have null values. The subset is ordered by 'highest_market_value_in_eur'.
players = data['players']
# br = psql.sqldf("""
#                 select name, foot, height_in_cm, highest_market_value_in_eur, date_of_birth, sub_position 
#                 from players 
#                 where foot is Null or height_in_cm is Null or highest_market_value_in_eur is Null 
#                         or date_of_birth is Null or sub_position is Null   
#                 order by highest_market_value_in_eur desc
#                 """)
pd.set_option('display.max_rows', 50, 'display.max_columns', None, "display.width", 1000)
# print(pd.DataFrame(br))
# Filtering the original 'players' DataFrame to remove rows where specific columns have null values.
data['players'] = data['players'][data['players']['foot'].isnull() == False]
data['players'] = data['players'][data['players']['height_in_cm'].isnull() == False]
data['players'] = data['players'][data['players']['highest_market_value_in_eur'].isnull() == False]
data['players'] = data['players'][data['players']['date_of_birth'].isnull() == False]
data['players'] = data['players'][data['players']['sub_position'].isnull() == False]
# print(data['players'].isnull().sum())

# Calculate the age of each player
data['players'] = data['players'][data['players']['date_of_birth'].notnull()]
now = pd.to_datetime('now')
data['players']['date_of_birth'] = pd.to_datetime(data['players']['date_of_birth'])
data['players']['age'] = now.year - data['players']['date_of_birth'].dt.year
data['players']['age'] = data['players']['age'].fillna(0).astype(int)

# print(data['competitions'].isnull().sum())
competitions = data['competitions']
# br = psql.sqldf("""
#                 select  name, country_name, domestic_league_code 
#                 from competitions 
#                 where country_name is Null or domestic_league_code is Null
#                 """)
# print(pd.DataFrame(br))


# print(data['games'].isnull().sum())
data['games'] = data['games'].drop(['home_club_manager_name', 'away_club_manager_name','attendance','referee','home_club_name','away_club_name'], axis=1)
# print(data['games'].isnull().sum())


game_events = data['game_events']
# br = psql.sqldf("""
#                 select *
#                 from game_events
#                 where type = 'Goals'
#                 """)
# print(pd.DataFrame(br))
# br = psql.sqldf("""
#                 select *
#                 from game_events
#                 where type = 'Substitutions'
#                 """)
# print(pd.DataFrame(br))
game_events = data['game_events']
# br = psql.sqldf("""
#                 select *
#                 from game_events
#                 where description is not Null and player_in_id is not Null 
#                 """)
# print("yeees", pd.DataFrame(br))
game_events = data['game_events']
# br = psql.sqldf("""
#                 select *
#                 from game_events
#                 where description is Null and player_in_id is Null
#                 """)
# print(pd.DataFrame(br))
game_events = data['game_events']
# br = psql.sqldf("""
#                 select *
#                 from game_events
#                 where description is Null and player_in_id is Null and type = 'Goals'
#                 """)
# print(pd.DataFrame(br))
# Empty DataFrame
# Columns: [game_event_id, date, game_id, minute, type, club_id, player_id, description, player_in_id, player_assist_id]
# Index: []

# Dropping rows from 'game_events' DataFrame where both 'description' and 'player_in_id' columns are null.
data['game_events'] = data['game_events'].dropna(how = 'all', subset = ['description','player_in_id'])
# print(data['game_events'].isnull().sum())



# print(data['club_games'].isnull().sum()) #opponent_manager_name    1348, own_manager_name         1348
data['club_games'] = data['club_games'].drop(['own_manager_name','opponent_manager_name'],axis = 1)
# print(data['club_games'].head())

# print(data['appearances'].isnull().sum())# player_name               324
data['appearances'] = data['appearances'].drop(['player_name'],axis = 1)
# print(data['appearances'].isnull().sum())



# print(data['clubs'].isnull().sum()) #total_market_value         426, coach_name                 426
data['clubs'] = data['clubs'].drop(['total_market_value','coach_name'],axis=1)
clubs = data['clubs']
# br = psql.sqldf("""
#                 select name, average_age, foreigners_number, foreigners_percentage
#                 from clubs
#                 where average_age is Null or foreigners_percentage is Null
#                 """)
# print(pd.DataFrame(br))
# Filling null values in the 'average_age' column of 'clubs' DataFrame with the mean value of 'average_age' column
data['clubs']['average_age'].fillna(data['clubs']['average_age'].mean(), inplace=True)
# Filling null values in the 'foreigners_percentage' column of 'clubs' DataFrame with 0
data['clubs']['foreigners_percentage'].fillna(0, inplace=True)
# print(data['clubs'].isnull().sum())

# check_null(data)


# Cleaning and standardizing the format of the 'round' column in the 'games' DataFrame.
# The result is a standardized format for the 'round' column where all the rounds now follow a consistent pattern ('XX. Matchday').
# print(data['games']['round'].unique())
data['games']['round'] = [f"{int(match.split('.')[0].strip()):02}. Matchday" if match.split('.')[0].strip().isdigit() else match for match in data['games']['round']]
# print(data['games']['round'].unique())





# # Database connection
# engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/football")

# # Exporting dataframes to the database
# # Iterate through the data dictionary and add DataFrames to the PostgreSQL database
# for key, df in data.items():
#     # Replace 'table_name' with the name you want for each table in the database
#     table_name = key
#     schema_name = 'a_ibrahim'
#     # Add each DataFrame to the PostgreSQL database
#     df.to_sql(name=table_name, con=engine, schema='a_ibrahim', if_exists='replace', index=False)

pd.set_option('display.max_rows', None)

# for key in data:
#     print('**'+key+'**')
#     print('INFO')
#     print('\n-----------------------')
#     print(data[str(key)].columns)
#     print('\n-----------------------')


players = data['players']
appearances = data['appearances']
clubs = data['clubs']
games = data['games']
player_valuations = data['player_valuations']
club_games = data['club_games']
game_lineups = data['game_lineups']
competitions = data['competitions']
game_events = data['game_events']

def GOAT_scored_against(players, appearances, clubs, games):
    """Returns a data frame with the total goals scored by Cristiano Ronaldo and Lionel Messi against each club they faced.

    Args:
        playersDF
        appearancesDF
        clubsDF
        gamesDF
    """
    br = psql.sqldf("""
                    SELECT name, against_club_name, sum(goals) AS total_goals
        FROM (
            SELECT g.game_id, DATE(g.date) AS date, g.season, 
                goat.goat_club, c1.name AS goat_club_name, 
                CASE WHEN g.home_club_id = goat.goat_club THEN g.away_club_id ELSE g.home_club_id END AS against_club,
                c2.name AS against_club_name, 
                goat.name, 
                goat.goals
            FROM games g
            JOIN (
                SELECT a.game_id, p.name, a.player_club_id AS goat_club, a.goals
                FROM players p
                JOIN appearances a ON p.player_id = a.player_id
                WHERE p.name IN ('Cristiano Ronaldo', 'Lionel Messi') AND a.goals != 0
            ) goat ON g.game_id = goat.game_id
            JOIN clubs c1 ON c1.club_id = goat.goat_club
            JOIN clubs c2 ON (c2.club_id = g.home_club_id OR c2.club_id = g.away_club_id)
            WHERE c2.club_id != goat.goat_club
            ORDER BY g.season, goat.name
        )
        GROUP BY name, against_club_name
        ORDER BY name, total_goals DESC
                    """)
    goat = pd.DataFrame(br)
    print(goat)
    return goat

def draw_GOAT_scored_against():

    goat = GOAT_scored_against(players, appearances, clubs, games)

    # Load the images as masks
    cristiano_mask = np.array(Image.open("../Final_Project/Images/C.jpg").resize((800, 800)))
    messi_mask = np.array(Image.open("../Final_Project/Images/M.jpg").resize((800, 800)))

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))

    # Subplot 1: Cristiano Ronaldo
    cristiano_df = goat[goat['name'] == 'Cristiano Ronaldo']
    cristiano_text = {row['against_club_name']: row['total_goals'] for _, row in cristiano_df.iterrows()}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#D41711', '#DF8D8B'])
    wordcloud = WordCloud(background_color='white', colormap=cmap, max_words=200, mask=cristiano_mask).generate_from_frequencies(cristiano_text)
    axs[0].imshow(wordcloud, interpolation='bilinear')
    axs[0].axis('off')
    #axs[0].set_title("Cristiano Ronaldo - Goals Against Clubs", fontsize=16)

    # Subplot 2: Lionel Messi
    messi_df = goat[goat['name'] == 'Lionel Messi']
    messi_text = {row['against_club_name']: row['total_goals'] for _, row in messi_df.iterrows()}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#1CC1ED', '#9FDEDB'])
    wordcloud = WordCloud(background_color='white', colormap=cmap, max_words=200, mask=messi_mask).generate_from_frequencies(messi_text)
    axs[1].imshow(wordcloud, interpolation='bilinear')
    axs[1].axis('off')
    #axs[1].set_title("Lionel Messi - Goals Against Clubs", fontsize=16)


    fig.text(0.5, 1, "Cristiano Ronaldo & Lionel Messi Goals Against Clubs", ha='center', va='center', fontsize=20, fontweight='bold', fontfamily='serif' )
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure with subplots
    plt.show()


def best_scorers_by_sub_position(game_events, players, games):
    """Returns:
        pd.DataFrame: A data frame with three columns: name, sub_position, and max(goals).
        The name column contains the name of the player who scored the most goals in each sub_position.
        The sub_position column contains the sub_position of the player, such as Centre-Forward, Right Winger,
        Left-Back, etc. The max(goals) column contains the maximum number of goals scored by the player in that sub_position.
        The data frame is ordered by max(goals) in descending order.
    """
        
    br = psql.sqldf("""
                    SELECT name,sub_position, max(goals) 
                    
                    FROM
                    
                    (SELECT name, sub_position,
                    count(name) as goals
                    FROM
                    (SELECT g.season, g.game_id, p.player_id, p.name, p.sub_position
                    FROM game_events ge
                    JOIN games g
                    ON ge.game_id = g.game_id
                    JOIN players p
                    ON ge.player_id = p.player_id
                    WHERE ge.type = 'Goals'
                    ) gg
                    
                    GROUP BY name
                    ORDER BY goals desc, sub_position)
                    
                    GROUP BY sub_position
                    ORDER BY goals desc
                    """)

    best_scorers_by_sub_position = pd.DataFrame(br)
    print(best_scorers_by_sub_position)
    return best_scorers_by_sub_position
    
def plot_age_distribution(players):
    sns.color_palette("dark")
    plt.title("Age of all players")
    sns.histplot(x='age',data=players, binwidth=1,color='orange')
    plt.show()

    print("Mean player age : ", players['age'].mean())
    print("Median player age : ", players['age'].median())

    High_value_players = players[players['highest_market_value_in_eur']>75000000]
    plt.title("Age of High value players (+£75m market Value)")
    sns.color_palette("muted")
    sns.histplot(x='age',data=High_value_players, binwidth=1,color='blue')
    plt.show()

    print("Mean player age (High value) : ", High_value_players['age'].mean())
    print("Median player age (High value) : ", High_value_players['age'].median())

    Low_value_players = players[players['highest_market_value_in_eur']<25000000]
    plt.title("Age of Low value players (<£25m market Value)")
    sns.color_palette("muted")
    sns.histplot(x='age',data=High_value_players, binwidth=1,color='blue')
    plt.show()

    print("Mean player age (Low value) : ", Low_value_players['age'].mean())
    print("Median player age (Low value) : ", Low_value_players['age'].median())

def plot_player_position_distribution(players):
    plt.pie(players['position'].value_counts(), labels=players['position'].unique(), autopct='%1.1f%%', startangle=140)
    plt.title('Player Position Distribution')
    plt.axis('equal')
    plt.show()

def plot_player_position_by_age(players):
   plt.figure(figsize=(12, 6))
   for age in range(15, 41):  # Loop through each age from 15 to 40
        players_at_age = players[players['age'] == age]
        position_counts = players_at_age['position'].value_counts()
        position_counts.plot(kind='bar', stacked=True, label=f'Age {age}')

   plt.title('Position Distribution for Each Age (Ages 15-40)')
   plt.xlabel('Position')
   plt.ylabel('Number of Players')
   plt.xticks(rotation=45)
   plt.legend(title='Age', bbox_to_anchor=(1.05, 1), loc='upper left')
   plt.tight_layout()
   plt.show()


def plot_top_30_youngster_teams(players, appearances, games, clubs):

    # Merge the dataframes
    merged_df = appearances.merge(games, on='game_id').merge(players, on='player_id').merge(clubs, left_on='player_club_id', right_on='club_id', how='inner')

    # Calculate age
    merged_df['age'] = (merged_df['date_x'] - merged_df['date_of_birth']).dt.days // 365

    # Filter for players aged 23 or less
    u23_df = merged_df[merged_df['age'] <= 23]

    # Group by season, domestic_competition_id, and club name, and calculate the number of players and total minutes played
    grouped_df = u23_df.groupby(['season', 'domestic_competition_id', 'name_y']).agg(
        u23_players=('player_id', 'nunique'),
        u23_min_play=('minutes_played', 'sum')
    ).reset_index()

    # Filter for the desired domestic_competition_ids
    filtered_df = grouped_df[grouped_df['domestic_competition_id'].isin(['GB1', 'ES1', 'IT1', 'L1', 'FR1', 'PO1', 'NL1'])]

    # Group by club and calculate the average number of players and total minutes played
    youngster = filtered_df.groupby(['name_y', 'domestic_competition_id']).agg(
        avg_num_of_u23_players=('u23_players', 'mean'),
        sum_u23_min_play=('u23_min_play', 'sum')
    ).sort_values(by=['sum_u23_min_play', 'avg_num_of_u23_players'], ascending=False).head(30)

    # Reset the index to make 'club' and 'domestic_competition_id' columns again
    youngster.reset_index(inplace=True)

    # Extract the relevant columns from the DataFrame
    x = youngster['avg_num_of_u23_players']
    y = youngster['sum_u23_min_play']
    club = youngster['name_y']
    league = youngster['domestic_competition_id']

    competition_dict = {
        'NL1': 'Eredivisie',
        'FR1': 'Ligue 1',
        'PO1': 'Primeira Liga',
        'IT1': 'Serie A',
        'GB1': 'Premier League',
        'ES1': 'La Liga',
        'L1': 'Bundesliga'
    }

    league = league.map(competition_dict)



    # Map leagues to colors
    colors = {
        'Premier League': 'red',
        'La Liga': 'blue',
        'Serie A': 'green',
        'Bundesliga': 'orange',
        'Ligue 1': 'purple',
        'Primeira Liga': 'cyan',
        'Eredivisie': 'yellow'
    }

    # Create a new column in the DataFrame to store league colors
    youngster['league_color'] = youngster['domestic_competition_id'].map(competition_dict).map(colors)

    # Scatter plot with colored points
    plt.figure(figsize=(12, 8))

    for i, row in youngster.iterrows():
        plt.scatter(row['avg_num_of_u23_players'], row['sum_u23_min_play'], s=150, alpha=0.7, label=row['domestic_competition_id'], c=row['league_color'], edgecolors='black')

    # Add labels and title
    plt.title('Top 30 Youngster Teams in European Leagues', fontsize=16)
    plt.xlabel('Average Number of U23 Players', fontsize=12)
    plt.ylabel('Total U23 Minutes Played', fontsize=12)
    plt.grid(True)

    # Create a legend with league names
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=league, markerfacecolor=color, markersize=10) for league, color in colors.items()]
    plt.legend(handles=legend_elements, title='Leagues', fontsize=10, title_fontsize='12')

    # Show club names with adjusted spacing
    for i, txt in enumerate(club):
        if i % 2 == 0:  # Alternate placing text above and below the point for better spacing
            plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(5, 8), ha='center', fontsize=8)
        else:
            plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(-5, -12), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('../Final_Project/plots/Top 30 Youngster Teams2.png', bbox_inches='tight')
    plt.show()
def plot_winning_precentages_by_year(games,clubs):
    """
    The horizontal bar chart visualizes these percentages over the seasons, allowing you to quickly identify trends,
      such as an increase or decrease in home advantage or the frequency of draws over time.
      It’s a useful way to present the data for comparative analysis across different seasons. 
    """

    # Merge the dataframes
    merged_df = games.merge(clubs, left_on='home_club_id', right_on='club_id', how='inner')

    # Create new columns for home win, draw, and away win
    merged_df['home_is_win'] = (merged_df['home_club_goals'] > merged_df['away_club_goals']).astype(int)
    merged_df['draw'] = (merged_df['home_club_goals'] == merged_df['away_club_goals']).astype(int)
    merged_df['away_is_win'] = (merged_df['home_club_goals'] < merged_df['away_club_goals']).astype(int)

    # Group by season and calculate the percentages
    grouped_df = merged_df.groupby('season').agg(
        home_winning_percentages=('home_is_win', 'mean'),
        draw_percentages=('draw', 'mean'),
        away_winning_percentages=('away_is_win', 'mean'),
        avg_home_club_goals=('home_club_goals', 'mean'),
        avg_away_club_goals=('away_club_goals', 'mean')
    )

    # Convert the percentages to the correct scale
    grouped_df[['home_winning_percentages', 'draw_percentages', 'away_winning_percentages']] *= 100

    # Round the percentages to 2 decimal places
    home_away = grouped_df.round(2).reset_index()

    fig, ax = plt.subplots(1,1, figsize = (17, 8))

    fig.text(0.13, 0.90, 'Home, Draw & Away Winning Percentages by Year', fontsize=15, fontweight='bold', fontfamily='serif')   
    #fig.text(0.13, 0.89, 'Percent Stacked Bar Chart', fontsize=12,fontfamily='serif') 


    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks(home_away['season'][::-1])
    ax.set_yticklabels(home_away['season'][::-1], fontfamily='serif', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    handles = []
    num_columns = 3
    colors = ['cornflowerblue', 'yellowgreen', 'lightcoral']
    labels = ['HOME', 'DRAW', 'AWAY']

    for j in home_away['season']:
        
        ax.barh(j, home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100, color = 'cornflowerblue', alpha = 0.8, label = 'HOME')
        ax.barh(j, home_away['draw_percentages'][(home_away.season) == j].values[0] / 100, left=home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100, color = 'yellowgreen', alpha = 0.8, label = 'DRAW')
        ax.barh(j, home_away['away_winning_percentages'][(home_away.season) == j].values[0] / 100, left=(home_away['draw_percentages'][(home_away.season) == j].values[0] + home_away['home_winning_percentages'][(home_away.season) == j].values[0]) / 100, color = 'lightcoral', alpha = 0.8, label = 'AWAY')
        
        
        handles.append(ax.barh(j, home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100, color = 'cornflowerblue', alpha = 0.8, label = 'HOME'))
        handles.append(ax.barh(j, home_away['draw_percentages'][(home_away.season) == j].values[0] / 100, left=home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100, color = 'yellowgreen', alpha = 0.8, label = 'DRAW'))
        handles.append(ax.barh(j, home_away['away_winning_percentages'][(home_away.season) == j].values[0] / 100, left=(home_away['draw_percentages'][(home_away.season) == j].values[0] + home_away['home_winning_percentages'][(home_away.season) == j].values[0]) / 100, color = 'lightcoral', alpha = 0.8, label = 'AWAY'))
        
        
        
        ax.annotate(f"{home_away['home_winning_percentages'][(home_away.season) == j].values[0]}%",
                    xy = ((home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100) / 2 , j ),
                    va = 'center', ha='center',fontsize=12, fontweight='light', 
                    fontfamily='serif',
                    color='white'
                )
        ax.annotate(f"{home_away['draw_percentages'][(home_away.season) == j].values[0]}%",
                    xy = ((home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100) + ((home_away['draw_percentages'][(home_away.season) == j].values[0] / 100) / 2) , j ),
                    va = 'center', ha='center',fontsize=12, fontweight='light', 
                    fontfamily='serif',
                    color='white'
                    )
        ax.annotate(f"{home_away['away_winning_percentages'][(home_away.season) == j].values[0]}%",
                    xy = ((home_away['home_winning_percentages'][(home_away.season) == j].values[0] / 100) + (home_away['draw_percentages'][(home_away.season) == j].values[0] / 100)  + ((home_away['away_winning_percentages'][(home_away.season) == j].values[0] / 100) / 2) , j ),
                    va = 'center', ha='center',fontsize=12, fontweight='light', 
                    fontfamily='serif',
                    color='white'
                    )

        


    legend = ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.83, 1.08),
                    ncol=num_columns, fancybox=True, shadow=True, fontsize=12)
    plt.savefig('../Final_Project/plots/Home, Draw & Away Winning Percentages by Year.png', bbox_inches='tight')    
    plt.show()

def plot_goal_from_which_position(game_events, players, games):
    """
    Each subplot will likely represent the goal-scoring trends for a specific position over the seasons.
    This visualization can help in quickly identifying which positions are most influential in scoring goals and how this has changed over time
    Positional Scoring Trends: You can analyze which positions are contributing most to the team’s scoring across seasons. For example,
      if Centre-Forwards are scoring less over time,
      it might indicate a shift in team strategy or the need for stronger players in that position.
Defensive Contributions: Goals by defensive positions like Centre-Backs or Left-Backs can highlight the effectiveness of set-pieces or the offensive capabilities of defenders.
Midfield Dynamics: The number of goals by midfielders, especially Attacking Midfielders and Wingers, can show how much the midfield contributes to the attack.
Seasonal Comparison: By comparing seasons, you can identify if there’s a consistent pattern in scoring across different positions or if there are anomalies
 that could be due to various factors like changes in coaching staff, player transfers, or injuries.
    """
    # Merge the dataframes
    merged_df = game_events.merge(games, on='game_id').merge(players, on='player_id')

    # Filter for goal events
    goals_df = merged_df[merged_df['type'] == 'Goals']

    # Create a new dataframe with the count of goals by sub_position and season
    goal_counts = goals_df.groupby(['season', 'sub_position']).size().reset_index(name='count')

    # Pivot the dataframe to have seasons as index and sub_positions as columns
    pivot_df = goal_counts.pivot(index='season', columns='sub_position', values='count').fillna(0)

    # Create a dictionary to map sub_positions to their respective goal columns
    position_to_goal = {
        'Centre-Forward': 'Centre_Forward_goal',
        'Centre-Back': 'Centre_Back_goal',
        'Goalkeeper': 'Goalkeeper_goal',
        'Left-Back': 'Left_Back_goal',
        'Attacking Midfield': 'Attacking_Midfield_goal',
        'Left Midfield': 'Left_Midfield_goal',
        'Right Winger': 'Right_Winger_goal',
        'Central Midfield': 'Central_Midfield_goal',
        'Right-Back': 'Right_Back_goal',
        'Left Winger': 'Left_Winger_goal',
        'Defensive Midfield': 'Defensive_Midfield_goal',
        'Right Midfield': 'Right_Midfield_goal',
        'Second Striker': 'Second_Striker_goal'
    }

    # Rename the columns in pivot_df according to position_to_goal
    pivot_df.rename(columns=position_to_goal, inplace=True)

    # Reset the index to make 'season' a column again
    g_position = pivot_df.reset_index()



    attack_position = g_position[['season', 'Centre_Forward_goal','Second_Striker_goal','Right_Winger_goal','Left_Winger_goal']]
    midfield_position = g_position[['season', 'Attacking_Midfield_goal','Left_Midfield_goal','Central_Midfield_goal','Defensive_Midfield_goal','Right_Midfield_goal']]
    defender_position = g_position[['season', 'Right_Back_goal','Left_Back_goal','Centre_Back_goal']]

    fig= plt.figure(figsize = (24,20))


    ax0 = plt.subplot2grid((20,28),(0,0), colspan = 18)


    ax1 = plt.subplot2grid((20,28),(1,1), colspan = 6, rowspan = 4) 
    ax2 = plt.subplot2grid((20,28),(1,7), colspan = 6, rowspan = 4) 
    ax3 = plt.subplot2grid((20,28),(1,13), colspan = 6, rowspan = 4)
    ax4 = plt.subplot2grid((20,28),(1,19), colspan = 6, rowspan = 4)


    ax5 = plt.subplot2grid((20,28),(5,1), colspan = 6, rowspan = 4)
    ax6 = plt.subplot2grid((20,28),(5,7), colspan = 6, rowspan = 4)
    ax7 = plt.subplot2grid((20,28),(5,13), colspan = 6, rowspan = 4)
    ax8 = plt.subplot2grid((20,28),(5,19), colspan = 6, rowspan = 4)


    ax9 = plt.subplot2grid((20,28),(9,1), colspan = 6, rowspan = 4)
    ax10 = plt.subplot2grid((20,28),(9,7), colspan = 6, rowspan = 4)
    ax11 = plt.subplot2grid((20,28),(9,13), colspan = 6, rowspan = 4)
    ax12= plt.subplot2grid((20,28),(9,19), colspan = 6, rowspan = 4)



    ax01 = plt.subplot2grid((20,28),(1,25), colspan = 2, rowspan = 4)
    ax02 = plt.subplot2grid((20,28),(5,25), colspan = 2, rowspan = 4)
    ax03 = plt.subplot2grid((20,28),(9,25), colspan = 2, rowspan = 4)


    colors = ['lightcoral', 'yellowgreen', 'cornflowerblue', 'mediumpurple']

    for i in (ax01, ax02, ax03):
        i.spines['top'].set_visible(False)
        i.spines['left'].set_visible(False)
        i.spines['right'].set_visible(False)
        i.spines['bottom'].set_visible(False)
        i.set_yticks([])
        i.set_xticks([])

        
    for i in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12):
        i.spines['right'].set_visible(False)
        i.spines['top'].set_visible(False)
        
        if i == ax1:
            ax1.scatter(attack_position['season'], attack_position['Centre_Forward_goal'], color=colors[0], marker = 'o')
            x = pd.to_numeric(attack_position['season'], errors='coerce')
            y = pd.to_numeric(attack_position['Centre_Forward_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax1.plot(x, p(x), color=colors[3], linestyle='--')
            ax1.set_title('Centre Forward', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[0])
            
        if i == ax2:
            ax2.scatter(attack_position['season'], attack_position['Second_Striker_goal'], color=colors[0], marker = 'o')
            x = pd.to_numeric(attack_position['season'], errors='coerce')
            y = pd.to_numeric(attack_position['Second_Striker_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax2.plot(x, p(x), color=colors[3], linestyle='--')
            ax2.set_title('Second Striker', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[0])
        
        if i == ax3:
            ax3.scatter(attack_position['season'], attack_position['Right_Winger_goal'], color=colors[0], marker = 'o')
            x = pd.to_numeric(attack_position['season'], errors='coerce')
            y = pd.to_numeric(attack_position['Right_Winger_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax3.plot(x, p(x), color=colors[3], linestyle='--')
            ax3.set_title('Right Winger', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[0])
            
        if i == ax4:
            ax4.scatter(attack_position['season'], attack_position['Left_Winger_goal'], color=colors[0], marker = 'o')
            x = pd.to_numeric(attack_position['season'], errors='coerce')
            y = pd.to_numeric(attack_position['Left_Winger_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax4.plot(x, p(x), color=colors[3], linestyle='--')
            ax4.set_title('Left Winger', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[0])

        if i == ax5:
            ax5.scatter(midfield_position['season'], midfield_position['Attacking_Midfield_goal'], color=colors[1], marker = 'o')
            x = pd.to_numeric(midfield_position['season'], errors='coerce')
            y = pd.to_numeric(midfield_position['Attacking_Midfield_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax5.plot(x, p(x), color=colors[3], linestyle='--')
            ax5.set_title('Attacking Midfielder', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[1])

            
        if i == ax6:
            ax6.scatter(midfield_position['season'], midfield_position['Left_Midfield_goal'], color=colors[1], marker = 'o')
            x = pd.to_numeric(midfield_position['season'], errors='coerce')
            y = pd.to_numeric(midfield_position['Left_Midfield_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax6.plot(x, p(x), color=colors[3], linestyle='--')
            ax6.set_title('Left Midfielder', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[1])
        
        if i == ax7:
            ax7.scatter(midfield_position['season'], midfield_position['Right_Midfield_goal'], color=colors[1], marker = 'o')
            x = pd.to_numeric(midfield_position['season'], errors='coerce')
            y = pd.to_numeric(midfield_position['Right_Midfield_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax7.plot(x, p(x), color=colors[3], linestyle='--')
            ax7.set_title('Left Midfielder', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[1])
            
        if i == ax8:
            ax8.scatter(midfield_position['season'], midfield_position['Central_Midfield_goal'], color=colors[1], marker = 'o')
            x = pd.to_numeric(midfield_position['season'], errors='coerce')
            y = pd.to_numeric(midfield_position['Central_Midfield_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax8.plot(x, p(x), color=colors[3], linestyle='--')
            ax8.set_title('Central Midfielder', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[1])
        
        if i == ax9:
            ax9.scatter(midfield_position['season'], midfield_position['Defensive_Midfield_goal'], color=colors[1], marker = 'o')
            x = pd.to_numeric(midfield_position['season'], errors='coerce')
            y = pd.to_numeric(midfield_position['Defensive_Midfield_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax9.plot(x, p(x), color=colors[3], linestyle='--')
            ax9.set_title('Defensive Midfielder', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[1])
        
        if i == ax10:
            ax10.scatter(defender_position['season'], defender_position['Right_Back_goal'], color=colors[2], marker = 'o')
            x = pd.to_numeric(defender_position['season'], errors='coerce')
            y = pd.to_numeric(defender_position['Right_Back_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax10.plot(x, p(x), color=colors[3], linestyle='--')
            ax10.set_title('Right Back', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[2])
            
        if i == ax11:
            ax11.scatter(defender_position['season'], defender_position['Left_Back_goal'], color=colors[2], marker = 'o')
            x = pd.to_numeric(defender_position['season'], errors='coerce')
            y = pd.to_numeric(defender_position['Left_Back_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax11.plot(x, p(x), color=colors[3], linestyle='--')
            ax11.set_title('Left Back', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[2])
        
        if i == ax12:
            ax12.scatter(defender_position['season'], defender_position['Centre_Back_goal'], color=colors[2], marker = 'o')
            x = pd.to_numeric(defender_position['season'], errors='coerce')
            y = pd.to_numeric(defender_position['Centre_Back_goal'], errors='coerce')
            coefficients = np.polyfit(x, y, deg=1)
            p = np.poly1d(coefficients)
            ax12.plot(x, p(x), color=colors[3], linestyle='--')
            ax12.set_title('Center Back', fontsize=14 , fontfamily='serif', fontweight='bold', color=colors[2])
        
        


    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.spines['top'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.text(0.76, 0.5, "Goal from every position", ha='center', va='center', fontsize=18, fontweight='bold', fontfamily='serif' )




    fig.text(0, 0.62, 'Aggregrated Goals', va='center', rotation='vertical', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.46, 0.3, 'Season', ha='center', fontsize=15, fontweight='bold', fontfamily='serif')

    plt.tight_layout()
    
    plt.savefig('../Final_Project/plots/Goal from every position.png', bbox_inches='tight') 
    plt.show()

def plot_top_10_players_by_yellow_cards(players,appearances):
    # Extracting relevant columns from 'appearances'
    player_appearances = appearances[['player_id', 'yellow_cards', 'red_cards', 'goals', 'assists', 'date']]
    # Extract the year from 'date'
    player_appearances['year'] = player_appearances['date'].dt.year

    # Filter for the year 2022
    player_appearances = player_appearances[player_appearances['year'] == 2022]    # Grouping the data by player_id and summing up the stats
    player_stats = player_appearances.drop('date', axis=1).groupby('player_id').sum().reset_index()

    # Merging with 'players' DataFrame to get player names
    merged_stats = pd.merge(player_stats, players[['player_id', 'name']], on='player_id')

    # Selecting the top 10 players with the most goals
    top_players = merged_stats.nlargest(10, 'red_cards')

    # Sorting the top players based on goals
    top_players = top_players.sort_values(by='goals', ascending=True)

    plt.figure(figsize=(10, 6))

    # Creating horizontal bar plots for each stat
    plt.barh(range(len(top_players)), top_players['yellow_cards'], color='yellow', alpha=0.7, label='Yellow Cards')
    plt.barh(range(len(top_players)), top_players['red_cards'], left=top_players['yellow_cards'], color='red', alpha=0.7, label='Red Cards')
    plt.barh(range(len(top_players)), top_players['goals'], left=top_players['yellow_cards']+top_players['red_cards'], color='blue', alpha=0.7, label='Goals')
    plt.barh(range(len(top_players)), top_players['assists'], left=top_players['yellow_cards']+top_players['red_cards']+top_players['goals'], color='green', alpha=0.7, label='Assists')

    # Adding player names on the left side of the bars
    plt.yticks(range(len(top_players)), top_players['name'])
    plt.xlabel('Count')
    plt.title('Top 10 Players in 2022: Goals, Yellow Cards, Red Cards, Assists')
    plt.legend()
    plt.tight_layout()

    plt.show()
def plot_most_valuable_player_by_year(players,player_valuations):

    # Convert the 'date' column to datetime
    player_valuations['date'] = pd.to_datetime(player_valuations['date'])

    # Extract year from the 'date' column
    player_valuations['year'] = player_valuations['date'].dt.year

    # Filter out records before 2005
    player_valuations = player_valuations[player_valuations['year'] > 2004]

    # Find the max market value for each year
    max_market_values = player_valuations.groupby('year')['market_value_in_eur'].max().reset_index()

    # Merge to get the names of players with the max market value for each year
    max_players = player_valuations.merge(max_market_values, on=['year', 'market_value_in_eur'])

    # Merge with the 'players' DataFrame to get player names
    max_players = max_players.merge(players, on='player_id')

    # Select the relevant columns and remove duplicates
    max_players = max_players[['year', 'name', 'market_value_in_eur']].drop_duplicates()

    # Group by year and concatenate player names
    player_value = max_players.groupby('year').agg({
        'name': lambda x: ', '.join(x),
        'market_value_in_eur': 'first'
    }).reset_index()

    plt.figure(figsize=(12, 7))

    # Creating a color palette for bars
    colors = sns.color_palette('viridis', len(player_value))

    # Plotting using Seaborn's barplot to enable using the color palette
    ax = sns.barplot(x='year', y='market_value_in_eur', data=player_value, palette=colors)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Market Value (in EUR)', fontsize=12)
    ax.set_title('Most Valuable Player by Year', fontsize=14)
    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability

    # Adding annotations on top of the bars
    for i, val in enumerate(player_value['market_value_in_eur']):
        ax.text(i, val + 500000, player_value['name'][i], ha='center', va='bottom', rotation=45, fontsize=8, color='black')

    # Adding a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_Player_Density_top_5_leages(competitions,clubs ):
    # Filter competitions to the desired competition IDs
    filtered_competitions = competitions[competitions['competition_id'].isin(['GB1', 'ES1', 'L1', 'IT1', 'FR1'])]

    # Merge clubs with the filtered competitions
    clubs_with_competitions = clubs.merge(filtered_competitions, left_on='domestic_competition_id', right_on='competition_id', how='inner')

    # Merge players with the clubs_with_competitions to get the current club's competition info for each player
    players_with_competitions = players.merge(clubs_with_competitions, left_on='current_club_id', right_on='club_id', how='inner')

    # Group by country_of_citizenship and competition_id and count the number of players
    nationality_counts = players_with_competitions.groupby(['country_of_citizenship', 'competition_id', 'name_y']).size().reset_index(name='count')

    # Sort the results by competition_name and count in descending order
    sorted_nationality = nationality_counts.sort_values(by=['competition_id', 'count'], ascending=[True, False])

    # Get the top 30 countries for each competition
    top_countries = sorted_nationality.groupby('competition_id').head(30)
    palette = sns.color_palette("colorblind")

    # Plotting for each league
    for league_id in top_countries['competition_id'].unique():
        league_data = top_countries[top_countries['competition_id'] == league_id]
        plt.figure(figsize=(12, 8))
        
        # Find the country with the most players
        max_players_country = league_data['country_of_citizenship'][league_data['count'].idxmax()]
        
        # Assign colors based on the country with the most players
        colors = [palette[1] if country == max_players_country else palette[0] for country in league_data['country_of_citizenship']]
        
        # Create the bar plot
        bars = plt.bar(league_data['country_of_citizenship'], league_data['count'], color=colors)
        
        # Annotate each bar with the number of players
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')
        
        # # Highlight the country with the most players by adding a marker
        # max_players_bar = bars[league_data['country_of_citizenship'].tolist().index(max_players_country)]
        # plt.scatter(max_players_bar.get_x() + max_players_bar.get_width()/2, max_players_bar.get_height(), color='red', zorder=5)
        
        plt.xlabel('Country')
        plt.ylabel('Number of Players')
        plt.title(f'Player Density in {league_data["name_y"].iloc[0]} by Country (Top 30)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        # Save the figure
        plt.savefig(f'Player_Density_{league_data["name_y"].iloc[0]}.png', format='png', dpi=300)
        plt.show()



def get_league_result(league, season):
    games = data['games']
    clubs = data['clubs']
    competitions = data['competitions']
    
    # Check if the league is a domestic league
    if competitions.loc[competitions['competition_id'] == league, 'type'].values[0] == 'domestic_league':
        # Filter games for the given league and season
        league_games = games[(games['competition_id'] == league) & (games['season'] == season)]
        
        # Create a DataFrame for home games
        home_games = league_games[['home_club_id', 'home_club_goals', 'away_club_goals']].copy()
        home_games.rename(columns={'home_club_id': 'club_id', 'home_club_goals': 'goals_for', 'away_club_goals': 'goals_against'}, inplace=True)
        home_games['result'] = home_games.apply(lambda row: 'win' if row['goals_for'] > row['goals_against'] else ('draw' if row['goals_for'] == row['goals_against'] else 'lose'), axis=1)
        
        # Create a DataFrame for away games
        away_games = league_games[['away_club_id', 'away_club_goals', 'home_club_goals']].copy()
        away_games.rename(columns={'away_club_id': 'club_id', 'away_club_goals': 'goals_for', 'home_club_goals': 'goals_against'}, inplace=True)
        away_games['result'] = away_games.apply(lambda row: 'win' if row['goals_for'] > row['goals_against'] else ('draw' if row['goals_for'] == row['goals_against'] else 'lose'), axis=1)
        
        # Combine home and away games
        all_games = pd.concat([home_games, away_games])
        
        # Calculate wins, draws, loses, goals for, goals against, goal difference, and points
        all_games['points'] = all_games['result'].map({'win': 3, 'draw': 1, 'lose': 0})
        all_games['goal_difference'] = all_games['goals_for'] - all_games['goals_against']
        
        # Group by club_id and calculate the aggregates
        league_table = all_games.groupby('club_id').agg(
            wins=('result', lambda x: (x == 'win').sum()),
            draws=('result', lambda x: (x == 'draw').sum()),
            loses=('result', lambda x: (x == 'lose').sum()),
            goals_for=('goals_for', sum),
            goals_against=('goals_against', sum),
            goal_difference=('goal_difference', sum),
            points=('points', sum)
        ).reset_index()
        
        # Merge with clubs to get the club names
        league_table = league_table.merge(clubs[['club_id', 'name']], on='club_id')
        
        # Sort the table by points, goal difference, and goals for
        league_table.sort_values(by=['points', 'goal_difference', 'goals_for'], ascending=[False, False, False], inplace=True)
        
        # Select the relevant columns and rename them
        league_table = league_table[['name', 'wins', 'draws', 'loses', 'goals_for', 'goals_against', 'goal_difference', 'points']]
        league_table.rename(columns={'name': 'team'}, inplace=True)
        
        return league_table
    else:
        print(f"{league} is not a domestic league")


# league = 'IT1'
# season = 2022
# result = get_league_result(league, season)

# print(result)

def get_UCL_result(season, data):
    games = data['games']
    clubs = data['clubs']

    # Filter games for the given season and competition 'CL'
    ucl_games = games[(games['season'] == season) & (games['competition_id'] == 'CL')]

    # Create a DataFrame for home and away games
    home_games = ucl_games[['round', 'home_club_id', 'home_club_goals', 'away_club_goals']].copy()
    home_games.rename(columns={'home_club_id': 'team_id', 'home_club_goals': 'goals_for', 'away_club_goals': 'goals_against'}, inplace=True)

    away_games = ucl_games[['round', 'away_club_id', 'away_club_goals', 'home_club_goals']].copy()
    away_games.rename(columns={'away_club_id': 'team_id', 'away_club_goals': 'goals_for', 'home_club_goals': 'goals_against'}, inplace=True)

    # Combine home and away games
    all_ucl_games = pd.concat([home_games, away_games])

    # Calculate result (win/draw/lose) for each game
    all_ucl_games['result'] = np.where(all_ucl_games['goals_for'] > all_ucl_games['goals_against'],
                                       'win', np.where(all_ucl_games['goals_for'] == all_ucl_games['goals_against'],
                                                       'draw', 'lose'))

    # Determine round_id based on round description
    round_mapping = {
        'Final': 4,
        'Semi-Finals': 3,
        'Quarter-Finals': 2,
        'Last 16': 1,
        'Group Stage': 0
    }

    all_ucl_games['round_id'] = all_ucl_games['round'].map(round_mapping)
    
    # Group by team_id and round_id and perform aggregations
    grouped_ucl = all_ucl_games.groupby(['team_id', 'round_id', 'round']).agg(
        is_pass=('result', lambda x: (x == 'win').any())  # Determine if team has at least one win in the round
    ).reset_index()

    # Join with clubs to get the team names
    grouped_ucl = grouped_ucl.merge(clubs[['club_id', 'name']], left_on='team_id', right_on='club_id')

    # Mapping round_id to descriptive round names
    round_mapping = {
        4: 'Final',
        3: 'Semi-Final',
        2: 'Quarter-Final',
        1: 'Last 16',
        0: 'Group Stage'
    }
    grouped_ucl['round'] = grouped_ucl['round_id'].map(round_mapping)

    # Convert 'is_pass' from boolean to integer for sorting
    grouped_ucl['is_pass'] = grouped_ucl['is_pass'].astype(int)

    # Sort the DataFrame by 'round_id' and 'is_pass'
    grouped_ucl.sort_values(by=['round_id', 'is_pass'], ascending=[False, False], inplace=True)

    # Rename columns and reorder
    grouped_ucl.rename(columns={'name': 'team_name'}, inplace=True)
    grouped_ucl = grouped_ucl[['round', 'team_name', 'round_id', 'is_pass']]

    return grouped_ucl

# Example usage:
# ucl_results_df = get_UCL_result(2019, data) # 'data' should be a dictionary containing 'games' and 'clubs' DataFrames
# print(ucl_results_df)
get_UCL_result(2022, data)