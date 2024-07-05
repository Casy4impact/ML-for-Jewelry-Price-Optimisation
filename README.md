# 1. Machine Learning for Jewelry Price Optimisation
The project aims to successfull predict the prices of jewelry this will allow the jewelry company to reduce its dependence on Gemologists and expensive jewelry appraisal experts.

# 2. Methodology
This project will be carried out using the Cross Industry Standard Process for Data Mining (CRISP-DM) Methodology. This is one of the most popular data science methodologies and it's characterised by six important phases:

1. Business understanding,
2. Data understanding,
3. Data preparattion,
4. Data Modeling,
5. Model Evaluation, and
6. Model Deployment.
It should be noted that these phases are recurrent in nature (i.e. some phases may be repeated). As such, they do not necessarily follow a linear progression.

# 3. Tools
- Pandas
- NumPy
- Matplotlib and seaborn: these are Python libraries trained for visualisation. The alternatives are Bokeh or Plotly
sci-kit learn: It's Python library which is extensively use for machine learning. It has a simple API centred around two main types of objects: i. Transformers and ii. Estimators

# Project Implementation vis CRISP-DM

## 1. Business understanding
### Jewelry Price Optimization with ML: 
- Pricing Data to Refine Pricing Strategies
Dive into the dynamic world of the jewelry market through this project, where I utilized machine learning to predict appropriate jewelry prices.
Throughout the project, I employed a range of Python libraries, including
- NumPy,
- Pandas,
- Matplotlib,
- Seaborn, RAPIDS, and
- Sci-kit Learn, to develop predictive models.
This hands-on experience not only enhanced my analytical skills but also provided practical insights into the factors influencing jewelry pricing in a competitive market.
## Business Overview/Problem
- Business Problem I
Having recently expanded operations, Gemineye Emporium has grown from a relatively small jewelry provider to a large-scale jewelry designing and trading company. Although this is good news, this transition is a very fast one, and while it has gone well all things considered, there were some edge cases the company board did not see coming. One glaring instance is the product prices. This is actually the major pain point for the company at the moment.

- Business Problem II
In the days where Gemineye was a small-scale holding, the price of any jewelry piece was very easy to determine or, in rare cases of confusion, ascertain. Now that operations have expanded exponentially, Gemineye finds itself not only importing pieces, but also manufacturing them in-house. With logistics concerns thrown into the mix, this becomes a more complicated system, and all these factors have two effects:
- A. Complicate the price determination process, and
- B. Inflate prices.
 
- Business Problem III
- Gemineye is in need of a means by which they can:
•	A. Make the most profit from their investment, and
•	B. Remain competitive by keeping their prices as affordable as possible.
 
In essence, they are in need of a way to maximize price to fill their pockets while minimizing it to suit the customers’ pockets. This is a tricky kettle of fish to boil!
- Business Problem IV
- To be more specific, the specific challenges of interest are:
•	A. Market Dynamics: The jewelry market is influenced by trends, fashion, and changing consumer preferences, making pricing decisions complex.
•	B. Competitive Pricing: Setting prices that are competitive with other jewelry retailers while offering unique and high-quality pieces.
•	C. Cost Management: Balancing the cost of materials, craftsmanship, and overhead expenses with market pricing is crucial for profitability.
### Rationale for the Project
More rationale for jewelry price optimization can be seen as follows:
 
- A. Profitability: Maximizing profits by balancing prices with costs and market demand
- B. Customer Satisfaction: Offering competitive prices for high-quality jewelry enhances customer satisfaction.
- C. Adaptability: The ability to adjust prices dynamically in response to market changes and trends.
### Aim of the Project
- The aims of the project are:
 
- A. Develop Price Optimization Models: Create machine learning models to predict optimal prices for different jewelry pieces based on market data and costs.
- B. Feature Selection (and Engineering): Identify (and engineer) relevant features that contribute to load capacity prediction accuracy.
- C. Ensure explainable predictions: Ensure that the developed price prediction models are explainable and can give insights to the business analysts as regards. This will help the business administration trust the model predictions even more.



## 2. Data understanding
- Business Problem IV
To be more specific, the specific challenges of interest are:
- A. Market Dynamics: The jewelry market is influenced by trends, fashion, and changing consumer preferences, making pricing decisions complex.
- B. Competitive Pricing: Setting prices that are competitive with other jewelry retailers while offering unique and high-quality pieces.
- C. Cost Management: Balancing the cost of materials, craftsmanship, and overhead expenses with market pricing is crucial for profitability.
### Rationale for the Project
More rationale for jewelry price optimization can be seen as follows:
 
- A. Profitability: Maximizing profits by balancing prices with costs and market demand
- B. Customer Satisfaction: Offering competitive prices for high-quality jewelry enhances customer satisfaction.
- C. Adaptability: The ability to adjust prices dynamically in response to market changes and trends.
### Aim of the Project
The aims of the project are:
 
•	A. Develop Price Optimization Models: Create machine learning models to predict optimal prices for different jewelry pieces based on market data and costs.
•	B. Feature Selection (and Engineering): Identify (and engineer) relevant features that contribute to load capacity prediction accuracy.
•	C. Ensure explainable predictions: Ensure that the developed price prediction models are explainable and can give insights to the business analysts as regards. This will help the business administration trust the model predictions even more.
  

This involves the process of Exploratory Data Analysis (EDA)
The required libraries and packages are imported first. The high-level steps to follow are:
- import the required libraries
- Load in the dataset
- Analyse and observe its properties
- Missing
- Outliers
- Inconsistent values
- Low categorical cardinality
- Data Imbalance
- Feature correlations
Report on these properties and how they might affect the final solution


## Analysis and Recommendations
1. Performance Comparison:
R2 Score:
The R2 score measures the proportion of the variance in the dependent variable that is predictable from the independent variables. The closer the R2 score is to 1, the better the model explains the variability of the response data.

a. The CatBoost model has the highest R2 score for both the training (0.3129) and test data (0.4115), indicating it explains the variability of the response data better than the other models.
a. The Linear Regression and AdaBoost models have relatively low R2 scores, indicating they do not explain the variance in the data well.
c. The ExtraTrees model performs moderately well with R2 scores of 0.2147 (train) and 0.2632 (test).
## RMSE:
The RMSE measures the average magnitude of the error. It’s the square root of the average squared differences between predicted and actual values. Lower RMSE values indicate better model performance.

a. The CatBoost model has the lowest RMSE for the test data (282.85), indicating it has the least average prediction error.
b. The ExtraTrees model has a moderate RMSE (316.49), followed by Linear Regression and AdaBoost with higher RMSE values.
2. Generalization:
The generalization error is the difference in performance between the training and test datasets. Smaller differences indicate better generalization to new, unseen data.
a. The CatBoost model shows a negative generalization error in R2 (-0.0986) and a higher RMSE generalization error (88.01), indicating some overfitting but generally performing well.
b. ExtraTrees also shows moderate generalization with slightly less overfitting compared to CatBoost.
c. Linear Regression and AdaBoost have relatively higher generalization errors, indicating these models might not generalize well to unseen data.

## Professional Recommendations
### Model Selection:
Given the metrics, the CatBoost model is the best performer in terms of both R2 score and RMSE. It should be the primary model for deployment, especially since it explains the variability better and has the lowest prediction error on the test data.
## Model Improvement:
- CatBoost: To further reduce overfitting and improve generalization, consider using techniques such as cross-validation, hyperparameter tuning, and increasing the dataset size if possible.
- ExtraTrees: As a secondary option, the ExtraTrees model can be fine-tuned further for potential improvements.
- Further Testing:
Conduct additional tests with cross-validation to ensure the robustness of the CatBoost model.
Experiment with feature engineering and additional preprocessing steps to enhance model performance.
- Deployment:
Deploy the CatBoost model while continuously monitoring its performance. Set up a feedback loop to regularly update the model with new data and retrain it to maintain accuracy.
- Documentation and Reporting:
Document the findings, model selection process, and performance metrics comprehensively. Present these insights to stakeholders to justify the choice of the CatBoost model for production use.
- Data Completeness:
The dataset contains a significant amount of missing data. It is crucial for the company to implement strategies to generate more complete and comprehensive datasets. This can involve improving data collection processes, integrating additional data sources, and employing methods to handle missing data more effectively.
- Data Quality:
A large portion of the data was found to be corrupted. The company should focus on obtaining cleaner data to ensure better problem-solving capabilities. This can be achieved by enhancing data validation protocols, implementing automated data cleaning tools, and conducting regular audits of the data collection processes.
- Data Collection:
Consider revisiting the data collection methods to ensure that all necessary information is captured accurately. This may include investing in better data collection technologies, training staff on proper data entry procedures, and using standardized formats for data entry.
- Data Management:
Implement robust data management practices to maintain the integrity and quality of the data. This includes establishing data governance policies, using reliable data storage solutions, and ensuring data security to prevent corruption.
- Regular Reviews:
Set up a process for regular reviews of the data to identify and rectify any issues promptly. This will help maintain the quality and reliability of the dataset over time, ensuring that the models built using this data are accurate and effective. By addressing these recommendations, the company can improve the overall quality of its data, leading to more accurate models and better decision-making capabilities for jewelry price optimization.
