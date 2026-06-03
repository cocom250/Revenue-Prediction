# Event Revenue Forecasting and Operational Planning
The UBC Film Society hosts multiple weekly film screenings and special events, generating revenue through concession sales such as popcorn, soft drinks, and candy. Event attendance can vary significantly depending on factors such as movie selection and promotional engagement. When attendance is lower than expected, excess concessions may go unsold, leading to food waste, while staffing levels may exceed operational needs.

This project aims to forecast event revenue using historical concession sales data, Instagram engagement metrics, and movie metadata. By predicting expected revenue prior to an event, the model can help estimate concession demand and staffing requirements, enabling more efficient event planning and reducing unnecessary waste and labour costs.

To accomplish this, I collected and analyzed data from ~90 historical Film Society events, engineered features related to audience engagement and movie popularity, and evaluated multiple machine learning models to identify the factors most strongly associated with event performance.

## Dataset 
The dataset was compiled from 84 historical events hosted by the UBC Film Society in the 2025-2026 school year. Each observation represents a single screening or special event and includes information available prior to the event as well as the resulting concession revenue.

### Event and Revenue Data
Historical event data was collected from internal Film Society records and Square point-of-sale reports. Revenue serves as the target variable for the forecasting task and represents total concession sales generated during an event.

Collected event attributes included:
- Event date
- Event type (double feature, special event, etc.)
- Collaboration status with other clubs or organizations
- Number of promotional posts
- Concession revenue

### Social Media Engagement Data
Promotional engagement metrics were collected manually from the Film Society's Instagram account. These metrics were used to estimate audience interest and marketing effectiveness prior to each event.

Collected engagement features included:
- Post likes
- Comments
- Shares
- Aggregate engagement metrics derived through feature engineering

### Movie Metadata
Initial models built using event and engagement data demonstrated limited predictive performance. To capture audience interest in the films themselves, an additional movie metadata dataset was constructed and merged with the event data.

Movie information was collected manually from Letterboxd and included:
- Movie title(s)
- Letterboxd average rating
- Letterboxd review count
- Genre
- Release year

For events featuring multiple films, metadata was collected for each movie and aggregated into event-level features such as average rating, total review count, and double-feature indicators.

### Final Dataset
After cleaning, preprocessing, and feature engineering, the final dataset combined operational, marketing, and movie-level information to support revenue forecasting and exploratory analysis. Feature engineering produced additional variables such as engagement rates, movie popularity measures, release-age features, and temporal event characteristics that were used during model development.

## Project Workflow
### 1. Data Collection

Historical event data was collected from UBC Film Society records, Square point-of-sale reports, and Instagram engagement metrics from the 2025-2026 school year. Each observation represented a single screening or special event and included concession revenue, promotional engagement, and event characteristics.
### 2. Data Cleaning and Preprocessing

The collected data was cleaned and standardized to ensure consistency across events. Missing values were handled using imputation within the machine learning pipeline, categorical variables were encoded using one-hot encoding, and numerical features were standardized where appropriate.
### 3. Exploratory Data Analysis

Exploratory analysis was conducted to understand revenue distributions, identify relationships between engagement metrics and revenue, and investigate patterns across event types. Visualizations were used to examine trends, detect outliers, and generate hypotheses for feature engineering.
### 4. Feature Engineering

Additional features were created to capture information not directly available in the raw dataset, including:
  - Total engagement and engagement per post
  - Temporal features such as month and day of week
  - Movie popularity metrics derived from Letterboxd review counts
  - Movie age and release-year features
  - Double-feature indicators
### 5. Baseline Modeling

Initial forecasting models were trained using event and Instagram engagement data. Ridge Regression and Random Forest models were evaluated using repeated cross-validation and compared against a dummy baseline predictor.
### 6. Dataset Enrichment

The baseline models demonstrated limited predictive power, suggesting that important explanatory variables were missing. To address this, movie metadata was collected from Letterboxd, including ratings, review counts, genres, and release years. These features were merged with the original event dataset to create an enriched dataset.
### 7. Model Development and Evaluation

Models were retrained using the enriched feature set and evaluated using repeated cross-validation. Performance was compared across baseline, movie-only, and combined feature sets to determine the contribution of movie popularity and engagement metrics to revenue forecasting.
### 8. Feature Importance Analysis

Permutation feature importance was used to identify the variables that contributed most strongly to model predictions. This analysis provided insight into the relative influence of social engagement, event characteristics, and movie popularity on event revenue.
### 9. Dashboard Development

The final model and analysis were integrated into an interactive Streamlit dashboard that allows organizers to explore historical trends, view model insights, and generate revenue forecasts for future events.

## Exploratory Analysis 
<img width="944" height="578" alt="image" src="https://github.com/user-attachments/assets/cf0eea8c-cd66-4249-9155-7ff717f235d6" />

- Revenue is highly right-skewed, with most events generating relatively modest concession sales and a small number of events producing substantially higher revenue. This distribution highlights the challenge of forecasting event performance and motivated the use of predictive modeling rather than relying on average historical revenue.


<img width="902" height="694" alt="image" src="https://github.com/user-attachments/assets/8bf35a00-b76a-4c89-83b1-de91e00fcf01" />

- Events with higher Instagram engagement generally tended to generate higher concession revenue, suggesting that social media activity captures some level of audience interest. However, the considerable variation among events indicates that engagement alone is insufficient to accurately explain revenue outcomes.


<img width="970" height="702" alt="image" src="https://github.com/user-attachments/assets/42db5e44-3cb2-40c4-bc42-072309065d37" />

- Collaboration status does not appear to have a strong relationship with concession revenue in this dataset. While some collaborative events achieved high revenue, the distributions overlap substantially, suggesting that other factors may play a larger role in driving event performance.


<img width="952" height="596" alt="image" src="https://github.com/user-attachments/assets/ea299b01-af45-4d68-ba21-598fc27ca018" />

- A positive relationship exists between movie popularity and event revenue. Events featuring films with higher Letterboxd review counts were more likely to generate stronger concession sales, suggesting that audience familiarity and interest in the selected films contribute meaningfully to event performance.


<img width="996" height="614" alt="image" src="https://github.com/user-attachments/assets/7ccfcd2b-88f0-4da2-a48f-06dde0101b6d" />

- Revenue increased consistently across movie popularity tiers, with events featuring highly reviewed films achieving substantially higher median revenue. This trend provided evidence that movie popularity contains predictive information beyond social media engagement metrics and motivated the inclusion of movie metadata in the forecasting models.


<img width="950" height="480" alt="image" src="https://github.com/user-attachments/assets/6aa631bc-496f-4970-92e5-feee7f97e76e" />

- Models trained using movie metadata outperformed those trained solely on event and Instagram features. The strongest performance was achieved by combining movie and engagement data, improving cross-validated predictive performance from approximately 0.18 R² in the baseline model to 0.35 R² in the final model. This suggests that both promotional engagement and movie popularity contribute valuable information when forecasting event revenue

## Results
### Model Performance
Three feature sets were evaluated using repeated cross-validation:
1. Baseline features consisting of event and Instagram engagement data.
2. Movie metadata features only.
3. A combined feature set incorporating both engagement and movie information.

The baseline models demonstrated limited predictive performance, indicating that social media engagement alone could not fully explain variations in event revenue.

The strongest performance was achieved using the combined feature set, which improved cross-validated performance from approximately 0.18 R² to 0.35 R². This indicates that both movie popularity and social engagement contribute complementary information when forecasting event revenue.

### Key Findings
Several important insights emerged from the analysis:
- Movie popularity exhibited a strong positive relationship with event revenue.
- Events featuring highly reviewed films consistently generated higher concession sales.
- Social media engagement provided useful predictive signal but was less informative than movie popularity metrics.
- Combining engagement and movie metadata produced the most accurate forecasting models.

### Business Implications
The results suggest that historical engagement metrics alone are insufficient for planning future events. Incorporating information about the popularity of the selected films can improve revenue forecasting and support more informed operational decisions.

These forecasts can be used to estimate concession demand and staffing requirements prior to an event, helping reduce food waste and improve resource allocation for screenings.

## Feature Importance Analysis
Feature importance analysis revealed that movie popularity was the strongest driver of event revenue. The number of Letterboxd reviews significantly outperformed all other features, supporting the hypothesis that audience interest in the selected films plays a major role in event performance.

Social media engagement also contributed to predictions, with shares and likes emerging as the most informative engagement metrics. Together, these results suggest that both movie selection and marketing influence revenue, though movie popularity provides the strongest predictive signal.

<img width="984" height="568" alt="image" src="https://github.com/user-attachments/assets/97543639-a655-4092-af35-37d3fc743ae5" />

## Limitations
Several limitations should be considered when interpreting the results of this project.
- The dataset contains only 84 historical events, limiting the amount of training data available for model development and increasing the sensitivity of performance estimates.
- Revenue was used as a proxy for event attendance and concession demand. Direct attendance counts were not available and may provide additional predictive signals.
- Some potentially important variables, such as room capacity, weather conditions, competing campus events, and Instagram reach/impressions, were not available for collection.
- Movie metadata was collected manually, limiting the scale and automation of the data collection process.

## Future Improvements
Several opportunities exist to improve both the forecasting model and the decision-support system.
- Expand the dataset with additional years of event history to improve model robustness and predictive performance.
- Incorporate attendance data, ticket reservations, and venue capacity information to better capture event demand.
- Automate movie metadata collection using external APIs to streamline data updates and support real-time predictions.
- Generate prediction intervals in addition to point estimates to communicate forecasting uncertainty.
- Evaluate additional forecasting approaches and incorporate new external factors such as weather, academic schedules, and competing campus events.


















