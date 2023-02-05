# nba_research

NOTE: Interactive plotly express plots are not displayed on github, please clone and run the notebook in your local machine to see the plots

The NBA style of play has changed dramatically since the birth of the league.

From the 50s to the early 70s, teams played at a frenetic pace, relying very heavily on their star players, some logging a season average of over 48 minutes.

From the late 70s to the 2000s, teams played at a slower pace, strong physical defense and relied on 2 point field goals, either from mid range, getting to the paint or palying from the post.

Moden playing styles have taken the pace from the early days but have adapted the shot distribution significantly, rely heavily on high quality shots - free throws, layups and 3 pointers, and we are seeing teams rely more heavily on depth (the days of superstar players logging over 40 minutes are gone).

The data obtained contains game level data by player showing individual statistics game by game. 

With the data above multiple analysis could be made, however in this repo the impact of 3 point field goals on winning will be analyzed.

The level of 3 pointers a game has greatly increased since its introduction in the 1980s. Teams used to attempt 3-6 3 point fields goals in the early days, however with the analytics revolution in the NBA, teams started taking 30 to up to 40 shots a game in the 2010s and 2020s. The 3 point field goal has now become one of the most popular shots in modern basketball and a shot by which modern day offenses are designed to optimize.

However, there comes a point when the impact of the 3 point field goal is saturated. There is little disputing that the 3 point field goal is a powerful shot for modern day offenses, since the Expected points for a 3 point field goal can be significantly higher than a 2 point field goal. However, there is a benefit towards a balanced shot distribution, since it mixes the offense and can create confusion for defense. If a defense knows that a team will go for a 3 point field goal, than it makes it much easier to plan a defense.

At this point it is important to note a difference in 2 variables, there are 3 point field goals attempted and conversion. Attempted is the total number of 3 point field goals attempted in a game, while conversion would be the 3 point field goal percentage (made / attempted).

This repo looks at analyzing both, with two hypotheses:
* 3 point field goals attempted do not have a significant impact on winning and losing
* 3 point field goal conversion has a significant impact on winning and losing

Therefore, to study the impact of 3 point field goals and prove the 2 hypotheses above, different visualizations will be used to analyze the impact of 3 point field goals on winning, and finally a feature importance analysis using the following models:

* Random Forest Classifier (sklearn package, tuned using a Grid Search)
* Gradient Boosting Model (sklearn package, tuned using a Random Search to reduce training time)
* XGBoost (xgboost package, tuned using a Random Search to reduce training time)

The feature scores will then be plotted with plotly express, and a better conclusion can be made about the impact of 3 point field goals on winning and losing in the modern day NBA.

It is important to note that this study is done using data from NBA regular season games from 2019-2022. Therefore I will only be researching about current day trends.
