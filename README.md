# kaggle_Predict_Customer_Churn
This is a self-study project. The Kaggle competition ‘Predict Customer Churn’ started on March 1, 2026. I’m using it to practice forecasting a binary classification problem. Traditionally, logistic regression is used for this type of task, but now we can also try more advanced machine learning models. Through this competition, I learned from the approaches of other participants and benchmarked my models against the leaderboard results to better understand effective forecasting methodologies.

Approach 1:  HistGradientBoostingClassifier
Accuracy: 0.91297

Approach 2: Since we have many categorical vars in the training dataset, we should try catboost model.
Accuracy: 0.91341

# I took average of the two prediction scores from these two models, the accuracy improved to 0.91346

<img width="2981" height="1754" alt="image" src="https://github.com/user-attachments/assets/9bbc0290-e26f-40af-bad9-eb6e03e3bca4" />

# The winner of this competition achieved a score of 0.91856, which was 0.005 better than mine. 
<img width="3049" height="1848" alt="image" src="https://github.com/user-attachments/assets/d5be09ab-183e-44fc-bc10-b2bbd7f97f91" />




