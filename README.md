# Call of Duty World League: Search & Destroy Outcome Predictions
![CWL Image](https://charlieintel.com/wp-content/uploads/2017/10/be476c3b-9364-44f0-a9a3-7d8b482515b8.jpg)

Growing up as an avid Call of Duty player, I was always curious about what factors led to a team winning or losing a match. Was it strictly based on the number of kills each player obtained? Was it who played the objective more? Or was it something different? Finally, after years of waiting, I decided that it was time to find my answers. Coupling my love for Call of Duty and my passion for data science, I began to investigate predicting the outcome of Search & Destroy games from the Call of Duty World League's 2018 and 2019 seasons.

Utilizing Python, I created a Logistic Regression binary classification model that provided insight into the major factors that go into a team winning a Search and Destroy match. Did you know that every time a player has exactly two kills in around a team's odds of winning increase by 59%? Or that every time a team defuses the bomb, their odds of winning the match increase by 54%? What about when someone on the team commits suicide? The team's odds of winning the match decreased by a whopping 43%! 

I also bulit an XGBoost and a Random Forest model in order to see just how accurate I could predict a Search & Destroy match outcome. The XGBoost model was ~89% accurate when predicting Search & Destroy match outcomes on test data! This model found that one of the least important variables for predicting a team's win or loss is if the team had a sneak defuse at any point during the match. Although sneak defuses are beneficial to a team's success, it would be more impactful for the team if players remove all enemies from the round before defusing the bomb. 

# Project Goals
1. Learn about essential factors that play into a team's outcome for Search & Destroy matches
2. See how well we can predict a team's wins and losses for Search & Destroy matches

# What did I do?
I used data from 17 different CWL tournaments spanning two years. If you are curious, you can find each dataset within this Activision repository hosted [here](https://github.com/Activision/cwl-data). I excluded the data from the 2017 CWL Championships tournament because this set does not have all the Search & Destroy variables that the other datasets have. The final dataset had 3,128 observations with 30 variables. In total, there are 1,564 Search & Destroy matches in this dataset. All variables are continuous; there were no categorical variables within the final data used for modeling besides the binary indicator for the match's outcome. 

To reach the first goal of this project, I created a Logistic Regression model to learn about the crucial factors that can either help or hinder a team. To reach the second goal of this project, I elected to use both Random Forest and XGBoost models for classification to try and find the best model possible at predicting match outcomes. 

# How did I do it?
### Logistic Regression
After joining the data, I first needed to group the observations by each match and team, then I filtered for only Search & Destroy games. That way, we have observations for both wins and losses of only Search & Destroy matches. I used a set of 14 variables for the model development process. The variables are as follows: `Deaths`, `Assists`, `Headshots`, `Suicides`, `Hits`, `Bomb Plants`, `Bomb Defuses`, `Bomb Sneak Defuses`, `Snd Firstbloods`, `Snd 2-kill round`, `Snd 3-kill round`, `Snd 4-kill round`, `2-piece`, & `3-piece`. If you are curious, you can find an explanation of each variable in the entire dataset in the Activision repository linked above. 

Since we are using these models to classify wins and losses correctly, I elected to use the Area Under the Receiver Operating Characteristic (AUROC) curve as a metric for determining the best model. I used AUROC because of its balance between the True Positive Rate and the False Positive Rate. I found that the Logistic Regression model with the highest AUROC value on training data had the following variables: `Assists`, `Headshots`, `Suicides`, `Defuses`, `Snd 2-kill round`, `Snd 3-kill round`, & `Snd 4-kill round`. This model was then used to predict test data and produced the following AUROC curve:

<p align="center">
  <img src="https://github.com/brettv30/Call-of-Duty-Game-Predictions/blob/main/Model%20Assessment%20Visuals/Logistic%20Regression%20AUC%20Graph.png" alt="Logistic AUROC Graph"
</p>

It is worth noting that this model was 75% accurate when predicting wins and losses on test data. Overall, I expected this model to perform worse due to the small number of variables used. Still, it seems as if these variables do an excellent job at deciphering the wins and losses in Search & Destroy matches. You can find the actual values in the confusion matrix built by this model [here](https://github.com/brettv30/Call-of-Duty-Game-Predictions/blob/main/Model%20Assessment%20Visuals/Logistic%20Regression%20Confusion%20Matrix.png).

### Random Forest & XGBoost
For the second goal of this project, I used both Random Forest and XGBoost classification models to see just how well we could predict the outcome of a match. Neither of these algorithms has the same assumptions as Logistic Regression, so I used the complete set of 14 variables for each technique. I first built both models without optimizing hyperparameters to have a baseline model for both algorithms. After this, I decided to use a grid search on the hyperparameters in each model to find the best possible tune for the data. 
  
I found that the optimized XGBoost model had a higher AUROC value than the optimized Random Forest model on training data, so I used the XGBoost model to predict the test data. This model produced the following AUROC curve:
  
<p align="center">
  <img src="https://github.com/brettv30/Call-of-Duty-Game-Predictions/blob/main/Model%20Assessment%20Visuals/XGBoost%20AUC%20Graph.png" alt="XGBoost AUROC Graph"
</p>

As expected, this model did much better than the Logistic Regression for predicting match outcomes! This model is ~89% accurate when predicting wins and losses on test data. You can find the confusion matrix for this model [here](https://github.com/brettv30/Call-of-Duty-Game-Predictions/blob/main/Model%20Assessment%20Visuals/XGBoost%20Confusion%20Matrix.png). 
  
# What did I find?
From the Logistic Regression model, I found that a team's odds of winning the entire match increase by ~5% every time someone gets a kill with a headshot and ~54% every time the bomb gets defused. A team's odds of winning also increase by 59% every time a player has exactly two kills in a round, ~115% every time a player has precisely three kills in a round, and ~121% every time a player has exactly four kills in around. I also found that a team's odds of winning the entire match decrease by 43% every time a player commits suicide and (oddly enough) 0.34% every time a player receives an assist. 
  
I recommend that professional COD teams looking to up their Search & Destroy win percentage need to find and recruit players with a high amount of bomb defuses and many headshots in Search & Destroy games. If I were a coach, I would be looking to grab `Arcitys`, `Zer0`,  `Clayster`, `Rated`, & `Silly`. These are five players who have a high count of headshots and defuses in Search & Destroy matches. 

If you are curious to learn about the essential variables in the XGBoost model, head over [here](https://github.com/brettv30/Call-of-Duty-Game-Predictions/blob/main/Model%20Assessment%20Visuals/XGBoost%20Variable%20Importance.png)!
