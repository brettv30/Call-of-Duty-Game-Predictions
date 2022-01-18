# Read in Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from matplotlib import pyplot as plt

# Read in the Data

dallas17 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2017-12-10-dallas.csv', error_bad_lines = False)
neworleans18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-01-14-neworleans.csv', error_bad_lines = False)
atlanta18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-03-11-atlanta.csv', error_bad_lines = False)
birmingham18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-01-birmingham.csv', error_bad_lines = False)
proleague118 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-08-proleague1.csv', error_bad_lines = False)
relegation18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-19-relegation.csv', error_bad_lines = False)
seattle18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-22-seattle.csv', error_bad_lines = False)
anaheim18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-06-17-anaheim.csv', error_bad_lines = False)
proleague218 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-07-29-proleague2.csv', error_bad_lines = False)
champs18 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-08-19-champs.csv', error_bad_lines = False)
proleaguequal19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-01-20-proleague-qual.csv', error_bad_lines = False)
fortworth19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-03-17-fortworth.csv', error_bad_lines = False)
london19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-05-05-london.csv', error_bad_lines = False)
anaheim19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-06-16-anaheim.csv', error_bad_lines = False)
proleague19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-07-05-proleague.csv', error_bad_lines = False)
proleaguefinals19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-07-21-proleague-finals.csv', error_bad_lines = False)
champs19 = pd.read_csv('https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-08-18-champs.csv', error_bad_lines = False)

def prep_18_data( season18 ):

    """

    Function that drops unnecessary columns, filters for Search & Destroy Matches, and returns that dataset.

    """

    season18 = season18.drop( ['series id', 'end time', 'map', 'player', 'score', '+/-', 'k/d', 'kills per 10min', 'deaths per 10min', 'accuracy (%)', 'avg time per life (s)',
                               'fave weapon', 'fave division', 'fave training', 'fave scorestreaks', 'hill time (s)', 'hill captures', 'hill defends',
                               'ctf captures', 'ctf returns', 'ctf pickups', 'ctf defends', 'ctf kill carriers', 'ctf flag carry time (s)', 
                               'scorestreaks earned', 'scorestreaks used', 'scorestreaks deployed', 'scorestreaks kills', 'scorestreaks assists',
                               '4-piece', '4-streak', '5-streak', '6-streak', '7-streak', '8+-streak'], axis = 1)

    season18['time alive (s)'] = season18['time alive (s)'].astype(int)

    season18 = season18[season18['mode'] == "Search & Destroy"]
    
    return season18

def prep_19_data( season19 ):

    """
    
    Function that drops unnecessary columns, filters for Search & Destroy matches, and returns that dataset.
    
    """
    season19 = season19.drop( ['series id', 'end time', 'map', 'player', 'score', '+/-', 'k/d', 'kills per 10min', 'deaths per 10min', 'player score', 'player spm', 'ekia',
                               'accuracy (%)', 'avg time per life (s)', 'fave weapon', 'fave specialist', 'fave scorestreaks', 
                               'hill time (s)', 'hill captures', 'hill defends', 'ctrl rounds', 'ctrl firstbloods', 'ctrl firstdeaths', 'ctrl captures',
                               '4-piece', '4-streak', '5-streak', '6-streak', '7-streak', '8+-streak'], axis = 1)

    season19['match id'] = season19['match id'].astype(str)

    season19 = season19[season19['mode'] == "Search & Destroy"]
    
    return season19

def join_sets( set1, set2 ):

    """
    
    Function that takes two datasets and joins them together
    
    """

    joined_set = pd.merge(set1, set2, on = ["match id", "duration (s)", "mode", "team", "win?", "kills", "deaths", "assists", "headshots", "suicides", "team kills", 
                                              "team deaths", "kills (stayed alive)", "hits", "shots", "num lives", "time alive (s)", 
                                              "snd rounds", "snd firstbloods", "snd firstdeaths", 'snd survives', 'bomb pickups', 'bomb plants', 'bomb defuses', 
                                              'bomb sneak defuses', 'snd 1-kill round', 'snd 2-kill round', 'snd 3-kill round', 'snd 4-kill round', '2-piece', '3-piece' ], how = 'outer' )

    return joined_set

def manipulate_sets( data ):

    """
    
    Function that rolls up the data to two observations per match id, one per team, removes the mode, creates a Win indicator, and aggregates all other variables
    
    """

    # Turn win? into a binary 0 - 1 variable. 0 for Loss & 1 for Win. 

    df_one = pd.get_dummies( data['win?'] )

    # Add both L & W columns back to the data

    data = pd.concat(( df_one, data ), axis = 1)

    # Drop the L column

    data = data.drop(['L', 'win?', 'mode'], axis = 1)

    # Rename W to outcome

    data = data.rename( columns = {"W": "outcome"})

    # aggregate the data

    data = data.groupby( ['match id', 'team'] ).agg( { 'outcome':['max'], 'duration (s)': ['max'], 'kills': ['sum'], 'deaths': ['sum'], 'assists': ['sum'], 'suicides':['sum'], 'headshots':['sum'], 'team kills':['sum'], 'team deaths': ['sum'],
                                                       'kills (stayed alive)':['sum'], 'hits':['sum'], 'shots':['sum'], 'num lives':['sum'], 'time alive (s)':['sum'], 'snd rounds':['max'], 'snd firstbloods':['sum'],
                                                       'snd firstdeaths':['sum'], 'snd survives':['sum'], 'bomb pickups':['sum'], 'bomb plants':['sum'], 'bomb defuses':['sum'], 'bomb sneak defuses':['sum'],
                                                       'snd 1-kill round':['sum'], 'snd 2-kill round':['sum'], 'snd 3-kill round':['sum'], 'snd 4-kill round':['sum'], '2-piece':['sum'], '3-piece':['sum'] } )

    data = data.reset_index()

    # Remove the second level of column titles

    data.columns = data.columns.droplevel(1)

    return data

def multicollinearity_check( dataset ):

    """
    
    Function that looks at each of the independent variables in the dataset and spits out their VIF scores as a DataFrame.
    
    """

    X = dataset[[ "assists", "headshots", "suicides",
                 "hits", "bomb plants", "bomb defuses", "bomb sneak defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round", "2-piece", "3-piece" ]]

    vif_data = pd.DataFrame()
    vif_data["features"] = X.columns

    vif_data["VIF"] = [ variance_inflation_factor( X.values, i ) for i in range( len( X.columns ) ) ]

    return vif_data

def optimize_model( x_set, y_set ):
    """
    
    Function that optimized the hyperparameters for the Random Forest Model
    
    """

    # Optimize hyperparameters 

    n_estimators = [ int(x) for x in np.linspace( start = 200, stop = 2000, num = 10)]

    max_features = ['auto', 'sqrt']

    max_depth = [int(x) for x in np.linspace( 100, 500, num = 11 )]

    # create random grid

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth
    }

    # Random search of parameters

    rfc_random = RandomizedSearchCV( estimator = rfc,
                                    param_distributions= random_grid,
                                    n_iter = 100,
                                    cv = 10,
                                    verbose = 0,
                                    random_state = 1,
                                    n_jobs = -1 )

    # Fit the model

    rfc_random.fit( x_set, y_set )

    # print results

    return rfc_random.best_params_ 

# Prepare the '18 data

dallas17 = prep_18_data( dallas17 )
neworleans18 = prep_18_data( neworleans18 )
atlanta18 = prep_18_data( atlanta18 )
birmingham18 = prep_18_data( birmingham18 )
proleague118 = prep_18_data( proleague118 )
relegation18 = prep_18_data( relegation18 )
seattle18 = prep_18_data( seattle18 )
anaheim18 = prep_18_data( anaheim18 )
proleague218 = prep_18_data( proleague218 )
champs18 = prep_18_data( champs18 )

# Prepare the '19 data

proleaguequal19 = prep_19_data( proleaguequal19 )
fortworth19 = prep_19_data( fortworth19 )
london19 = prep_19_data( london19 )
anaheim19 = prep_19_data( anaheim19 )
proleague19 = prep_19_data( proleague19 )
proleaguefinals19 = prep_19_data( proleaguefinals19 )
champs19 = prep_19_data( champs19 )


# Join all of the training data together

new_set = join_sets( dallas17, neworleans18 )
new_set = join_sets( atlanta18, new_set )
new_set = join_sets( birmingham18, new_set )
new_set = join_sets( new_set, proleague118 )
new_set = join_sets( new_set, relegation18 )
new_set = join_sets( new_set, seattle18 )
new_set = join_sets( new_set, anaheim18 )
new_set = join_sets( new_set, proleague218 )
new_set = join_sets( new_set, champs18 )
new_set = join_sets( new_set, proleaguequal19 )
new_set = join_sets( new_set, fortworth19 )
new_set = join_sets( new_set, london19 )
new_set = join_sets( new_set, anaheim19 )
new_set = join_sets( new_set, proleague19 )
new_set = join_sets( new_set, proleaguefinals19 )
df = join_sets( new_set, champs19 )

df = manipulate_sets( df )


"""

Logistic Regression Modeling Section

"""

# Drop the unnecessary variables from the training set

training_set = df[["outcome", "deaths", "assists", "headshots", "suicides", "snd firstbloods",
                 "hits", "bomb plants", "bomb defuses", "bomb sneak defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round", "2-piece", "3-piece"]]

# Create the X and Y data from the training set. Drop deaths & snd firstbloods due to multicollinearity issues

X = training_set[[ "assists", "headshots", "suicides",
                 "hits", "bomb plants", "bomb defuses", "bomb sneak defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round", "2-piece", "3-piece" ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Check for Multicollinearity

training_vif = multicollinearity_check( X_train )

# Implement the model & remove variables based on P-value

logit_model = sm.Logit( y_train, X_train )

result = logit_model.fit()

# Round 2

X = training_set[[ "assists", "headshots", "suicides",
                 "hits", "bomb plants", "bomb defuses", "bomb sneak defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round", "2-piece" ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Implement the model & remove variables based on P-value

logit_model = sm.Logit( y_train, X_train )

result = logit_model.fit()

# Round 3

X = training_set[[ "assists", "headshots", "suicides",
                 "hits", "bomb plants", "bomb defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round", "2-piece" ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Implement the model & remove variables based on P-value

logit_model = sm.Logit( Y, X )

result = logit_model.fit()

# Round 4

X = training_set[[ "assists", "headshots", "suicides",
                 "hits", "bomb plants", "bomb defuses", "snd 2-kill round",
                "snd 3-kill round", "snd 4-kill round" ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Implement the model & remove variables based on P-value

logit_model = sm.Logit( Y, X )

result = logit_model.fit()

# Rename variables so they can be used to verify Linearity of Logit assumption

training_set = training_set.rename( columns = { 'snd 2-kill round': 'snd_2kill_round',
                                                'snd 3-kill round': 'snd_3kill_round',
                                                'snd 4-kill round': 'snd_4kill_round',
                                                'bomb plants': 'plants',
                                                'bomb defuses': 'defuses'})

cont_vars = ['assists', 'headshots', 'suicides', 'hits', 'snd_2kill_round', 'snd_3kill_round', 'snd_4kill_round', 'plants', 'defuses']

#  Apply a constant to all variables

for var1 in cont_vars:
    training_set[var1] = training_set[var1] + 1

# Rename variables that we apply the natural log to

for var in cont_vars:
    training_set[f'{var}_Log_{var}'] = training_set[var].apply( lambda x: x * np.log(x))

cols_to_keep = cont_vars + training_set.columns.tolist()[-len(cont_vars):]

# Test the linearity of the logits assumption

model = smf.logit("outcome ~ assists + assists_Log_assists + headshots + headshots_Log_headshots + suicides + suicides_Log_suicides + snd_2kill_round + snd_2kill_round_Log_snd_2kill_round + snd_3kill_round + snd_3kill_round_Log_snd_3kill_round + snd_4kill_round + snd_4kill_round_Log_snd_4kill_round + defuses + defuses_Log_defuses", data = training_set).fit()

# All variables above meet the assumption of linearity

# Final Variable Set

X = training_set[[ "assists", "headshots", "suicides", "defuses",
                   "snd_2kill_round", "snd_3kill_round", "snd_4kill_round" ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Instantiate the model

logreg = LogisticRegression( max_iter = 345 )

# fit the model using the training data

logreg.fit( X_train, y_train )

# Use the model to make predictions on training data

y_pred = logreg.predict( X_train )

# Determine AUC

y_pred_proba = logreg.predict_proba( X_train )[::,1]

auc = metrics.roc_auc_score( y_train, y_pred_proba )

print('\n')
print("Logistic Regression Model Scoring Metrics: Training Data")
print('\n')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_train, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_train, y_pred))
print('\n')
print( "AUC:", str(auc) )
print('\n')

"""

Random Forest Modeling Section

"""

# Create the X and Y data from the training set

X = training_set[[ 'deaths', 'assists', 'headshots', 'suicides', 'hits', 'snd_2kill_round', 'snd_3kill_round', 'snd_4kill_round', 
                  'plants', 'defuses', "bomb sneak defuses", "2-piece", "3-piece", 'snd firstbloods' ]]

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Run a grid search to find the optimal tune for hyperparameters

"""
rfc = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

# Number of features to consider at every split
max_features = ['auto']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfc_random = RandomizedSearchCV( estimator = rfc, 
                                param_distributions = random_grid, 
                                n_iter = 100, 
                                cv = 10, 
                                verbose=0, 
                                random_state=21, 
                                n_jobs = -1,
                                scoring = 'roc_auc')

rfc_random.fit( X_train, y_train )

print( rfc_random.best_params_ )

"""

rfc = RandomForestClassifier(
    n_estimators=800,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features= 'auto',
    max_depth=3,
    random_state=21
)

rfc.fit( X_train, y_train )

# Predictions

rfc_predict = rfc.predict( X_train )

print("RF Classification Model Scoring Metrics: Training Data")
print('\n')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_train, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_train, rfc_predict))
print('\n')
print("=== AUC Score ===")
print( metrics.roc_auc_score(y_train, rfc.predict_proba(X_train)[:,1]))
print('\n')
print("=== RF Parameters ===")
print( rfc.get_params() )
print('\n')

feature_importance = pd.DataFrame( {'Variable': X.columns,
                                     'Importance': rfc.feature_importances_}).sort_values('Importance', ascending=False)

print("=== Random Forest Important Features ===")
print( feature_importance )

"""

XGBoost modeling section

"""

# Run a grid search to find the optimal tune for hyperparameters

"""

estimator = XGBClassifier(
    objective = "binary:logistic",
    nthread = 4,
    seed = 21
)

parameters = {
    'max_depth': range( 2, 10, 1),
    'n_estimators': range(500, 1000, 100),
    'learning_rate': [0.1, 0.05, 0.01]
}

grid_search = GridSearchCV(
    estimator = estimator,
    param_grid = parameters,
    scoring = 'roc_auc',
    n_jobs = -1, 
    cv = 10, 
    verbose=0
)

grid_search.fit( X_train, y_train )

print( grid_search.best_params_ )

"""

clf = XGBClassifier( max_depth = 3, 
                     learning_rate = 0.01, 
                     n_estimators = 900,
                     objective = 'binary:logistic', 
                     booster = 'gbtree',
                     seed = 21 )

# Creating the model on training data

XGB = clf.fit( X_train, y_train )

prediction = XGB.predict( X_train )

print("Optimized XGB Classification Model Scoring Metrics: Training Data")
print('\n')
print("=== Confusion Matrix ===")
print(metrics.confusion_matrix( y_train, prediction ))
print('\n')
print("=== Classification Report ===")
print(metrics.classification_report( y_train, prediction ))
print('\n')
print("=== AUC Score ===")
print( metrics.roc_auc_score(y_train, XGB.predict_proba(X_train)[:,1]))
print('\n')
print("=== XGB Parameters ===")
print( clf.get_params() )
print('\n')

feature_importance = pd.DataFrame( {'Variable': X.columns,
                                     'Importance': XGB.feature_importances_}).sort_values('Importance', ascending=False)

print("=== XGB Important Features ===")
print( feature_importance )
print('\n')

"""

Take Binary Logistic Regression Model to Test Data

"""

X = training_set[[ "assists", "headshots", "suicides", "defuses",
                   "snd_2kill_round", "snd_3kill_round", "snd_4kill_round" ]]

X = X.rename( columns = { 'snd_2kill_round': '2-Kill Round', 'snd_3kill_round': '3-Kill Round', 'snd_4kill_round':'4-Kill Round',
                                                            'assists': 'Assists', 'defuses': 'Defuses',
                                                            'headshots': 'Headshots', 'suicides': 'Suicides'})

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

# Instantiate the model

logreg = LogisticRegression( max_iter = 345 )

# fit the model using the training data

logreg.fit( X_train, y_train )

# Use the model to make predictions on test data

y_pred = logreg.predict( X_test )

# Determine AUC

y_pred_proba = logreg.predict_proba( X_test )[::,1]

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

print('\n')
print("Logistic Regression Model Scoring Metrics: Test Data")

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Logistic Regression Confusion Matrix', fontsize=18)
plt.show()

# Create Coefficient Interpretable Dataframe

coefs = pd.concat([pd.DataFrame(X_test.columns), pd.DataFrame(np.transpose(logreg.coef_))], axis = 1)

coefs.columns = ['Variable', 'Coefficient']

coefs['Coefficient'] = coefs['Coefficient'].astype( float )

coefs['Odds Ratio'] = np.exp( coefs['Coefficient'] )
coefs['Odds Ratio (percentage)'] = 100 * (coefs['Odds Ratio'] - 1)

# Create AUC Visual

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve

plt.plot(fpr, tpr, label="AUC="+str(auc), color = 'b')
plt.plot([0,1], [0,1], linestyle = 'dashed', color = 'k')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression AUC Graph')
plt.legend(loc=4)
plt.show()

"""

Take XGBoost model to Test Data

"""

X = training_set[[ 'deaths', 'assists', 'headshots', 'suicides', 'hits', 'snd_2kill_round', 'snd_3kill_round', 'snd_4kill_round', 'snd firstbloods',
                  'plants', 'defuses', "bomb sneak defuses", "2-piece", "3-piece" ]]

X = X.rename( columns = { 'snd_2kill_round': '2-Kill Round', 'snd_3kill_round': '3-Kill Round', 'snd_4kill_round':'4-Kill Round', 'deaths': 'Deaths', 'snd firstbloods': 'Firstbloods',
                                                            'hits': 'Shots Hit', 'assists': 'Assists', 'plants': 'Plants', 'defuses': 'Defuses',
                                                            'headshots': 'Headshots', 'suicides': 'Suicides', 'bomb sneak defuses': 'Sneak Defuses',
                                                            '2-piece': '2-Piece', '3-piece': '3-Piece'})

Y = training_set[ "outcome" ]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 21 )

clf = XGBClassifier( max_depth = 3, 
                     learning_rate = 0.01, 
                     n_estimators = 900,
                     objective = 'binary:logistic', 
                     booster = 'gbtree',
                     seed = 21 )

XGB = clf.fit( X_train, y_train )

# Predictions

prediction = XGB.predict( X_test )

# Create confusion matrix

conf_matrix = confusion_matrix(y_test, prediction)

print('\n')
print("XGBoost Classification Model Scoring Metrics: Test Data")
print('\n')

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('XGBoost Confusion Matrix', fontsize=18)
plt.show()

print("=== XGB Parameters ===")
print( XGB.get_params() )
print('\n')

y_pred_proba = XGB.predict_proba(X_test)[:,1]

# Create AUC Visual

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve

plt.plot(fpr, tpr, label="AUC="+str(auc), color = 'b')
plt.plot([0,1], [0,1], linestyle = 'dashed', color = 'k')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('XGBoost AUC Graph')
plt.legend(loc=4)
plt.show()

feature_importance = pd.DataFrame( {'Variable': X.columns,
                                     'Importance': XGB.feature_importances_}).sort_values('Importance', ascending=False)

sorted_idx = XGB.feature_importances_.argsort()
plt.barh( feature_importance.Variable[sorted_idx], XGB.feature_importances_[sorted_idx] )
plt.xlabel( "Mean Decrease Impurity" )
plt.title('XGBoost Variable Importance')
plt.show()
