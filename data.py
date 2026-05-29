from statsbombpy import sb
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler
corners_data = """match_id,minute,team,player,location_x,location_y,under_pressure,goal_scored
7570,12,Arsenal,Mesut Özil,89.0,36.0,False,no
7570,23,Arsenal,Mesut Özil,92.0,38.0,True,no
7570,34,West Ham,Mark Noble,8.0,42.0,False,no
7570,45,Arsenal,Theo Walcott,87.0,35.0,False,no
7570,56,West Ham,Aaron Cresswell,12.0,44.0,True,no
7570,67,Arsenal,Santi Cazorla,90.0,37.0,False,yes
7570,78,Arsenal,Mesut Özil,88.0,36.0,False,no
7570,89,West Ham,Manuel Lanzini,10.0,41.0,True,no
7570,91,Arsenal,Mesut Özil,91.0,39.0,False,no
7570,93,West Ham,Mark Noble,9.0,43.0,True,no
7581,15,Manchester City,Kevin De Bruyne,91.0,34.0,False,no
7581,28,Manchester City,Riyad Mahrez,88.0,37.0,False,no
7581,42,Chelsea,Mason Mount,7.0,39.0,True,no
7581,55,Manchester City,Kevin De Bruyne,90.0,35.0,False,yes
7581,67,Chelsea,Reece James,9.0,41.0,True,no
7581,79,Manchester City,Phil Foden,92.0,36.0,False,no
7581,88,Chelsea,Ben Chilwell,10.0,38.0,True,no
7582,8,Liverpool,Trent Alexander-Arnold,91.0,33.0,False,no
7582,19,Liverpool,Andrew Robertson,89.0,35.0,False,no
7582,33,Manchester United,Bruno Fernandes,8.0,40.0,True,no
7582,47,Liverpool,Trent Alexander-Arnold,92.0,34.0,False,yes
7582,61,Manchester United,Marcus Rashford,11.0,42.0,True,no
7582,74,Liverpool,Mohamed Salah,90.0,36.0,False,no
7582,86,Manchester United,Luke Shaw,9.0,39.0,True,no
7583,11,Tottenham,Heung-min Son,88.0,37.0,False,no
7583,24,Tottenham,James Maddison,90.0,35.0,False,no
7583,38,Newcastle,Kieran Trippier,10.0,43.0,True,no
7583,52,Tottenham,Dejan Kulusevski,91.0,36.0,False,no
7583,65,Newcastle,Bruno Guimarães,8.0,41.0,True,no
7583,79,Tottenham,Heung-min Son,89.0,34.0,False,no
7584,6,Aston Villa,John McGinn,87.0,38.0,False,no
7584,18,Brighton,Kaoru Mitoma,11.0,44.0,True,no
7584,31,Aston Villa,Leon Bailey,90.0,36.0,False,no
7584,44,Brighton,Pascal Groß,9.0,40.0,True,no
7584,57,Aston Villa,Ollie Watkins,88.0,37.0,False,no
7584,69,Brighton,Solly March,12.0,42.0,True,no
7584,82,Aston Villa,Douglas Luiz,91.0,35.0,False,no
7585,14,West Ham,Jarrod Bowen,89.0,36.0,False,no
7585,27,West Ham,Lucas Paquetá,91.0,38.0,False,no
7585,41,Crystal Palace,Eberechi Eze,8.0,41.0,True,no
7585,53,West Ham,Jarrod Bowen,90.0,34.0,False,no
7585,68,Crystal Palace,Michael Olise,10.0,43.0,True,no
7585,81,West Ham,James Ward-Prowse,88.0,37.0,False,yes
7586,9,Chelsea,Enzo Fernández,90.0,35.0,False,no
7586,22,Fulham,Willian,9.0,42.0,True,no
7586,35,Chelsea,Cole Palmer,91.0,36.0,False,no
7586,48,Fulham,João Palhinha,8.0,40.0,True,no
7586,62,Chelsea,Conor Gallagher,89.0,34.0,False,no
7586,76,Fulham,Andreas Pereira,11.0,44.0,True,no
7587,13,Bournemouth,Philip Billing,87.0,39.0,False,no
7587,26,Nottingham Forest,Morgan Gibbs-White,10.0,41.0,True,no
7587,39,Bournemouth,Justin Kluivert,90.0,37.0,False,no
7587,54,Nottingham Forest,Anthony Elanga,8.0,42.0,True,no
7587,68,Bournemouth,Marcus Tavernier,88.0,35.0,False,no
7588,17,Everton,Dwight McNeil,89.0,38.0,False,no
7588,30,Sheffield United,Gustavo Hamer,11.0,43.0,True,no
7588,43,Everton,Amadou Onana,90.0,36.0,False,no
7588,57,Sheffield United,James McAtee,9.0,41.0,True,no
7588,71,Everton,Jordan Pickford,88.0,34.0,False,no
7589,5,Arsenal,Bukayo Saka,91.0,35.0,False,no
7589,22,Arsenal,Martin Ødegaard,89.0,37.0,False,no
7589,41,Manchester United,Bruno Fernandes,10.0,42.0,True,no
7589,58,Arsenal,Bukayo Saka,90.0,36.0,False,yes
7589,73,Manchester United,Marcus Rashford,8.0,40.0,True,no
7589,88,Arsenal,Martin Ødegaard,92.0,38.0,False,no
7590,7,Manchester City,Phil Foden,88.0,36.0,False,no
7590,19,Manchester City,Kevin De Bruyne,91.0,34.0,False,no
7590,36,Liverpool,Mohamed Salah,9.0,41.0,True,no
7590,52,Manchester City,Julián Álvarez,90.0,35.0,False,no
7590,68,Liverpool,Darwin Núñez,11.0,44.0,True,no
7590,84,Manchester City,Kevin De Bruyne,89.0,37.0,False,yes
7591,3,Chelsea,Enzo Fernández,90.0,36.0,False,no
7591,21,Chelsea,Cole Palmer,88.0,35.0,False,no
7591,39,Tottenham,James Maddison,10.0,42.0,True,no
7591,57,Chelsea,Conor Gallagher,91.0,37.0,False,no
7591,72,Tottenham,Dejan Kulusevski,8.0,41.0,True,no
7591,89,Chelsea,Nicolas Jackson,92.0,36.0,False,no
7592,11,Newcastle,Kieran Trippier,89.0,38.0,False,no
7592,28,Newcastle,Bruno Guimarães,90.0,36.0,False,no
7592,44,Aston Villa,John McGinn,11.0,43.0,True,no
7592,61,Newcastle,Alexander Isak,88.0,35.0,False,yes
7592,77,Aston Villa,Ollie Watkins,9.0,40.0,True,no
7592,90,Newcastle,Callum Wilson,91.0,37.0,False,no
7593,6,Brighton,Kaoru Mitoma,87.0,39.0,False,no
7593,24,Brighton,Pascal Groß,90.0,36.0,False,no
7593,42,West Ham,Jarrod Bowen,10.0,44.0,True,no
7593,59,Brighton,Solly March,88.0,35.0,False,no
7593,75,West Ham,James Ward-Prowse,11.0,42.0,True,no
7593,90,Brighton,João Pedro,91.0,38.0,False,yes
7594,8,Fulham,Willian,89.0,37.0,False,no
7594,25,Fulham,Andreas Pereira,90.0,35.0,False,no
7594,43,Wolves,Pedro Neto,8.0,41.0,True,no
7594,60,Fulham,Raúl Jiménez,91.0,36.0,False,no
7594,78,Wolves,Matheus Cunha,10.0,43.0,True,no
7594,92,Fulham,Willian,88.0,34.0,False,no
7595,4,Everton,Dwight McNeil,90.0,38.0,False,no
7595,22,Everton,Amadou Onana,89.0,36.0,False,no
7595,41,Burnley,Josh Brownhill,11.0,44.0,True,no
7595,58,Everton,Dominic Calvert-Lewin,88.0,35.0,False,yes
7595,76,Burnley,Wilson Odobert,9.0,42.0,True,no
7595,88,Everton,James Garner,91.0,37.0,False,no
"""

df = pd.read_csv(StringIO(corners_data))
#print(df.head())
df['goal_scored'] = df['goal_scored'].map({'yes': 1, 'no': 0})
df['under_pressure'] = df['under_pressure'].astype(int)
unique_teams = sorted(df['team'].unique())
unique_player = sorted(df['player'].unique())
team_to_code = { team: i for i, team in enumerate (unique_teams)}
player_to_code = { player: i for i, player in enumerate (unique_player)}
df['coded_team'] = df['team'].map(team_to_code)
df['coded_player'] = df['player'].map(player_to_code)
print(df.head())
X = df[['minute', 'coded_team', 'coded_player', 'location_x', 'location_y', 'under_pressure']]
y = df['goal_scored']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#train the model
models = {
    'Randomforest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradientboosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'logisticregressor': LogisticRegression(max_iter=1000, random_state=42)

}
results = {}
best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = {'accuracy': acc, 'model': model}
    print(f"{name}  , accuracy {acc}")
    if acc > best_score:
        best_score = acc
        best_model = model
    


print(f"Best  model {best_model}  with accuracy {best_score}")

data = {
    'models': models,
    'scaler': scaler,
    'teams': unique_teams,
    'player': unique_player,
    'coded_team': team_to_code,
    'coded_player': player_to_code,
    'best_model': best_model
}
joblib.dump(data, 'data.pkl')
joblib.dump(models, 'models.pkl')
data = df.to_csv('corners_data.csv')
def predict_corners():

    print("Enter the corner data")
    
