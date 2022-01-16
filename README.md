# Call of Duty World League: Search & Destroy Outcome Predictions
![CWL Image](https://charlieintel.com/wp-content/uploads/2017/10/be476c3b-9364-44f0-a9a3-7d8b482515b8.jpg)

Growing up as an avid Call of Duty player, I was always curious about what factors went in to a team winning or losing a match. Was it strictly based on the number of kills each player obtained? Was it who played the objective more? Or was it something different? Finally, after years of waiting, I decided that it was time to have my answer. Coupling my love for Call of Duty and my passion for data science, I began to investigate predicting the outcome of Search & Destroy games from the Call of Duty World League's 2018 and 2019 seasons.

Utilizing Python, I was able to create a logistic regression binary classification model that provided insight into the major factors that go in to a team winning a Search and Destroy match. Did you know that every time a player has exactly two kills in a round a team's odds of winning increase by 59%? Or that every time a team defuses the bomb their odds of winning the match increase by 54%? What about when someone on the team commits suicide? The team's odds of winning the match decrease by a whopping 43%! Next time you're the last person alive, you might want to give your team a chance and go out fighting instead of pulling the plug early. 

Interestingly enough, the Random Forest Model that was built found that one of the least important variables for predicting a team's Win or Loss was if the team had a sneak defuse. A sneak defuse happens when the defending team is able to defuse the bomb while at least one player on the attacking team is still alive. I found this to be one of the more curious findings. In practice when someone on my team has performed a sneak defuse, the momentum seems to shift back in to my team's favor and we generally go on to win the match. Albiet, most of us playing Call of Duty aren't professionals so maybe a sneak defuse doesn't shake the best of the best like it does us laymen. 

# Project Goals
1. Learn about important factors that play into a team's outcome for Search & Destroy Matches
2. See how well we can predict a team's Wins and Losses

# What did I do?
I used data from 17 different CWL tournaments spanning two years. The datasets can be found within this Activision repository hosted [here](https://github.com/Activision/cwl-data). I decided to exclude the data from the 2017 CWL Champsionships tournament because that set did not have all Search & Destroy variables that the rest of the data had. The final dataset ended up having 3,128 observations with 30 variables. In total, there are 1,564 Search & Destroy matches in this dataset. All variables were treated as continuous, there were no categorical variables within the final data used for modeling besides the binary indicator for the outcome of the match. 

In order to reach the first goal of this project, I created a Logistic Regression model to pull some explain-ability from each of the variables in the dataset. To reach the second goal of this project, I elected to use both Random Forest and XGBoost for classification to try and find the best model possible at predicting game outcomes. 

# How did I do it?



# What did I find?


# What have I learned?
As noted above, I found that even though the Random Forest model had the highest AUC value on training data it did not transfer over to complete success on test data. Taking my time and really sifting through each possible model for the Logistic Regression model seemed to pay off more in this instance. I found this aspect of the project very interesting because it shows exactly how strong Logistic Regression can be when handled with care. 

This was the first project that I have completed in Python and it has really boosted my overall confidence with the language. It felt like a different way of thinking than when I have used R in the past, and its a language I want to continue using to polish my skills! I am now especially comfortable using `numpy`, `pandas`, `matplotlib`, and `sklearn`. 
