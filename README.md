
# Project Title:

# Predict Podcast Listening Time 

📘 README: 

Overview
Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Your task it to predict listening time of a podcast episode.
Submissions are scored on the root mean squared error. RMSE is defined as:

Submission File

For each id in the test set, you must predict the Listening_Time_minutes of the podcast. The file should contain a header and have the following format:

id,Listening_Time_minutes
750000,45.437
750001,45.437
750002,45.437
etc.


### 📂 Dataset Description

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

 ### Feature Descriptions:

- id – Unique identifier for each podcast episode (not used in modeling).

- Podcast_Name – Name of the podcast series (categorical).

- Episode_Title – Title of the episode (text, mostly for display or NLP).

- Episode_Length_minutes – Total length of the episode in minutes.

- Genre – Type of podcast content (e.g., News, Comedy, True Crime).

- Host_Popularity_percentage – Popularity score of the host (0–100 scale).

- Publication_Day – Day of the week the episode was released.

- Publication_Time – Time the episode was published (e.g., morning, evening, or exact hour).

- Guest_Popularity_percentage – Popularity score of the guest, if present (0–100).

- Number_of_Ads – Count of ads in the episode.

- Episode_Sentiment – Overall sentiment of the episode (e.g., Positive, Neutral, Negative).

- Listening_Time_minutes – Target variable: how many minutes users listened to the episode


 ## Primary Task

- Perform EDA
- Data cleaning
- Prediction Target: Listening_Time_minutes

### Tools & Libraries

- matplotlib.pyplot -	Plotting line charts, histograms, bar charts
- seaborn	Enhanced statistical plots like boxplots, scatterplots, heatmaps
- sklearn.preprocessing -	Encoding, scaling (StandardScaler, LabelEncoder, OneHotEncoder)
- sklearn.preprocessing - Encoding, scaling (StandardScaler, LabelEncoder, OneHotEncoder)
- sklearn.ensemble.RandomForestRegressor - Regression model using bagging
- xgboost.XGBRegressor -	High-performance gradient boosting model
- sklearn.linear_model.LinearRegression - Baseline model for regression
- sklearn.model_selection	- Train-test split, cross-validation, hyperparameter tuning
- sklearn.metrics	- Evaluation metrics: MSE, RMSE, R², MAE

### Exploratory Data Analysis (EDA)

EDA Insists:

- Dataset Structure: The dataset contains 750,000 episodes with a mix of numerical and categorical features. Some key columns like Episode_Length_minutes and Guest_Popularity_percentage had missing values.

- Missing Value Handling: I imputed missing values using the median within each podcast group to retain contextual accuracy and avoid dropping important features.

- Feature Distributions: Most numerical features showed right-skewed distributions, especially Episode_Length_minutes, Listening_Time_minutes, and Number_of_Ads, which also had visible outliers.

- Outlier Detection: Boxplots confirmed the presence of outliers in Episode_Length_minutes and Number_of_Ads. These charts helped visualize data spread and highlight anomalies more clearly than histograms.

- Publication Patterns: Sunday had the highest number of published episodes, while Tuesday had the fewest. Publication frequency remained relatively consistent throughout the week.

- Listening Time by Categories: Listening time was fairly consistent across genre, publication day, time of day, and sentiment, with slight variations but no dominant category.

- Episode Length vs Listening Time (Bivariate): A positive relationship exists between episode length and listening time. However, vertical bands suggested common formatting choices like fixed 60 or 120-minute episodes.

- Sentiment Impact: Positive sentiment episodes had slightly higher average listening times than neutral and negative ones. Sentiment distribution was nearly balanced across days.

- Multivariate Analysis: Adding genre to the episode length vs. listening time plot showed variation by genre, but no clear ranking. Genre affects engagement subtly, but episode length remains the stronger predictor of listening time.


### Data Cleaning

- I Imputed missing values using the median within each podcast group, preserving contextual accuracy while avoiding data loss.

- I verified missing values were filled by re-checking null counts after imputation to ensure data integrity.

- And converted categorical sentiment labels (Positive, Neutral, Negative) into numerical values using a custom mapping for modeling.

- Lastly I saved the remaining numerical features into a clean version of the dataset

### Model Training & Evaluation

I trained and evaluated both RandomForestRegressor and XGBoostRegressor using default settings and with hyperparameter tuning through RandomizedSearchCV. The default Random Forest model gave solid results, but tuning improved its performance, reaching an MSE of 172.857 and an R² score of 0.765. XGBoost with default parameters performed better, achieving an MSE of 170.072 and an R² score of 0.769. 

Surprisingly, after tuning XGBoost using RandomizedSearchCV, the performance slightly dropped to an MSE of 172.999 and an R² score of 0.765, matching the tuned Random Forest results.

From this process, I learned that default models can already provide strong baselines, and that tuning doesn’t always guarantee better performance. 

I chose to run and save predictions on the test using the default versions of both XGBoost and Random Forest models. After comparing them to their tuned counterparts, I found that the default XGBoost model performed the best overall.



