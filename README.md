# 🚗 Car Price Prediction in Poland

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)

## 📌 Project Overview
This repository contains an end-to-end machine learning project focused on predicting used car prices in the Polish automotive market. The objective of the project is to develop a **production-ready pricing engine** capable of estimating the market value of a vehicle based on its technical specifications, usage characteristics, and market context.

The model leverages a variety of vehicle attributes such as brand, model, production year, mileage, engine parameters, and equipment features to generate accurate price predictions. By analyzing patterns present in historical market data, the system is able to capture complex relationships between vehicle characteristics and their corresponding market prices.

The solution is designed to support **data-driven decision-making** for both professional dealerships and private sellers. It can be used as a tool for quickly estimating competitive listing prices, understanding depreciation trends, and identifying key factors that influence vehicle valuation in the used car market.

This project demonstrates a complete **machine learning workflow**, including data preprocessing, feature engineering, model development, hyperparameter optimization, and detailed model evaluation. The final model is built using advanced gradient boosting techniques and is optimized to provide reliable predictions across a wide range of vehicle types and price segments.

The project workflow follows a structured machine learning pipeline consisting of the following stages:

1. **Data acquisition & ingestion** – collect, load, and clean raw vehicle listings obtained from Polish online car sales platforms.

2. **Exploratory Data Analysis (EDA)** – analyze feature distributions, detect anomalies and outliers, and generate visual insights to better understand the structure of the dataset.

3. **Data preprocessing & feature engineering** – handle missing values, encode categorical variables, create derived features (such as vehicle age or power-to-weight ratio), and remove extreme outliers that could negatively affect model training.

4. **Model experimentation** – evaluate multiple modeling approaches, starting with baseline linear models, followed by tree-based ensemble methods, and ultimately an optimized **XGBoost regressor**.

5. **Hyperparameter tuning** – apply **Optuna** to efficiently search the hyperparameter space and identify optimal model configurations.

6. **Evaluation & validation** – assess model performance using standard regression metrics such as **RMSE, MAE, MAPE, and R²**, complemented by residual analysis and diagnostic visualizations.

7. **Deployment & delivery** – serialize the trained model and publish it to **Hugging Face**, develop an interactive **Streamlit dashboard** for price prediction, and organize the workflow into reproducible scripts.

## 🚀 Live Demo & Models

### 🖥️ Streamlit Dashboard
Explore the interactive application to predict car prices in real-time:
**[Launch App](https://cars-price-prediction-in-poland-93x3kme8tvdopec5f4vxul.streamlit.app/)**

### 🤗 Hugging Face Model Registry
Due to file size constraints, the trained models are hosted on the Hugging Face Hub:
**[View Models on Hugging Face](https://huggingface.co/Przemsonn/poland-car-price-model)**

---

## 📚 Table of Contents
1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Workflow Steps](#workflow-steps)
   * [Data Ingestion](#data-ingestion)
   * [Preprocessing & Cleaning](#preprocessing--cleaning)
   * [Exploratory Data Analysis](#exploratory-data-analysis)
   * [Feature Engineering](#feature-engineering)
   * [Model Training & Baselines](#model-training--baselines)
   * [Optimization & Tuning](#optimization--tuning)
   * [Evaluation](#evaluation)
   * [Deployment](#deployment)
4. [Results & Business Impact](#results--business-impact)
5. [Tech Stack](#tech-stack)
6. [Installation & Usage](#installation--usage)
7. [Future Work](#future-work)

---

## 📁 Dataset

The raw dataset is stored in `data/Car_sale_ads.csv` and contains **over 200,000 vehicle listings** scraped from popular Polish automotive marketplaces.

The dataset includes a wide range of attributes describing vehicle specifications, usage, and market context. Key fields include:

* **Vehicle information:** `brand`, `model`, `year`, `mileage`
* **Technical specifications:** `fuel_type`, `power_hp`, `type`, `transmission`, `displacement_cm3`, `colour`, `origin_country`, `doors_number`, `first_owner`, `condition`
* **Pricing information:** `price` (target variable, expressed in PLN or EUR), `currency`
* **Registration\Offer details:** `registration_date`, `offer_publication_date`
* **Text-based attributes:** `features` and `offer_location`, which may contain additional signals affecting vehicle price (e.g., optional equipment or regional market differences)

The `notebooks/` directory contains Jupyter notebooks used for **exploratory data analysis (EDA)** and early-stage experimentation, including the main analysis notebook `cars_price_prediction.ipynb`.

---

## 📂 Project Structure
```
├── data/               
│   └── Car_sale_ads.csv
├── images/             
├── notebooks/          
├── src/               
│   ├── data.py         
│   ├── features.py  
│   ├── model.py       
│   └── utils.py      
├── app.py           
├── .gitignore  
├── requirements.txt  
├── LICENSE
└── README.md          
```

---

## 🔁 Workflow Steps
Each stage of the project is documented below, along with the corresponding code files and notebooks used during the development process.

### 🗂 Data Ingestion

The data ingestion stage initializes the environment and prepares the dataset for further analysis and modeling.

At this step, all required Python libraries for data processing, visualization, and machine learning are imported to ensure a consistent development environment.

The raw dataset is then loaded from the CSV file using `pandas.read_csv` within the `src/data.py` module. This script centralizes the data loading logic, allowing the dataset to be easily accessed and reused across different parts of the project pipeline, including exploratory analysis, preprocessing, and model training.

---

### 🔧 Data Preprocessing & Quality Assessment

Data preprocessing ensures data quality, consistency, and prepares the dataset for feature engineering. This stage addresses missing values, outliers, data types, and currency standardization while carefully avoiding data leakage.

#### 💱 Currency Standardization

**Challenge:** Dataset contains prices in multiple currencies (PLN, EUR, USD)

**Solution:** Convert all prices to PLN using official exchange rates from the National Bank of Poland (NBP) API

---

### 🔍 Exploratory Data Analysis (EDA)

Comprehensive exploratory analysis was conducted to understand the dataset structure, distributions, and key patterns before modeling. The analysis revealed critical insights about the Polish used car market.

#### 📊 Target Variable Analysis: Price Distribution

Understanding price depreciation patterns is crucial for accurate predictions.

![Depreciation Curve/Rate](images/eda_depreciation_analysis.png)

**Key Insights:**

- **Rapid early depreciation:** Vehicles lose ~50% of their value within the first 5 years
- **Stable decline period:** Between 5-25 years, depreciation follows a consistent downward trend
- **Classic car effect:** Vehicles older than 25 years show price stabilization or slight increases, indicating transition into the collectible/vintage segment
- **Peak depreciation rate:** Occurs between years 2-3 of ownership (highest annual loss)
- **Lowest depreciation:** Around 13 years of age, depreciation stabilizes at 11-15% annually

**Implications for modeling:** Log transformation recommended due to right-skewed distribution and wide price range.

---

#### 🔢 Feature Relationships: Key Predictors vs Price

Analysis of the four most important numerical features reveals strong predictive patterns.

![Top 4 Important Features vs Price](images/eda_scatterplots_relations_clean.png)

**Key Insights:**

**1. Price vs Production Year**
- Clear upward trend: newer vehicles command higher prices
- Sharp increase post-2015 due to modern technology, safety systems, and market demand
- Vehicles from 2020+ show significant premium

**2. Price vs Mileage**
- Strong inverse relationship: lower mileage = higher price
- High concentration around 180,000 km with wide price dispersion
- Variation driven by brand, condition, and vehicle segment
- Critical predictor for depreciation modeling

**3. Price vs Power (HP)**
- Positive correlation: higher horsepower = higher price
- High-performance vehicles (>500 HP) show extreme price variance
- Strongest numerical predictor identified
- Reflects luxury level, brand prestige, and rarity

**4. Price vs Engine Displacement (cm³)**
- Common engine sizes: 1,600-2,000 cm³, 3,000 cm³, 4,000 cm³
- Generally positive relationship, but less linear than HP
- Large engines (7,000 cm³) appear in American SUVs and supercars
- Displacement alone doesn't capture performance (turbocharging effect)

**Modeling impact:** Strong candidates for polynomial features and interaction terms.

---

#### 🎯 Interaction Effects: Mileage × Age × Segment

Examining how multiple features jointly influence price reveals complex patterns.

![Mileage vs Price (colored by age)](images/eda_mileage_vs_price_by_age.png)

**Key Insights by Vehicle Age:**

**New Cars (<3 years)**
- Smallest group in dataset
- Clustered around 0-20,000 km
- Wide price range: 50,000 - 1,000,000 PLN
- Demo vehicles show slight mileage with premium pricing (200,000-500,000 PLN)

**Recent Cars (3-8 years)**
- Mileage typically <100,000 km
- Broad price distribution: 50,000-300,000 PLN
- Premium brands retain high value despite higher mileage
- Some supercars exceed 300,000 PLN even with age

**Used Cars (9-16 years)**
- Mileage range: 50,000-300,000 km
- Prices rarely exceed 200,000 PLN
- Clear negative mileage-price relationship
- Mass-market segment dominates

**Old Cars (>16 years)**
- Highest mileage (up to 400,000+ km)
- Prices generally <50,000 PLN
- Exceptions: vintage/collectible vehicles

**Modeling implications:** 
- Strong interaction between `Vehicle_age`, `Mileage_km`, and `Brand`
- Tree-based models ideal for capturing non-linear relationships
- Segment-specific patterns require feature engineering

---

#### ⚡ Categorical Features: Fuel Type Evolution

Analyzing price trends across fuel types reveals market shifts and technological adoption.

![Car Price Over the Years by Fuel Type](images/eda_average_car_price_over_the_years_by_fuel_type.png)

**Key Insights:**

**Historical Trends (1920s-1990s)**
- Gasoline dominated production
- Gradual price fluctuations across decades
- Diesel gained popularity in 1980s-90s (slightly lower prices)

**Modern Era (2000-2010)**
- Structured price differentiation by fuel type
- Diesel positioned for fuel efficiency
- Gasoline remained standard choice

**Recent Shift (2010-present)**
- **Electric vehicles:** Sharp price increase (premium positioning, battery costs, technology)
- **Hybrid:** Moderate growth (mid-range segment)
- **Diesel & Gasoline:** Steady increase
- **CNG & LPG:** Minimal growth (cost-sensitive segment)

**Feature interactions identified:**
- Fuel type × Production year (electric = newer)
- Fuel type × Brand (electric = premium brands)
- Fuel type × Power (performance variants)

**Modeling impact:** TargetEncoder recommended for high-cardinality categorical variables.

---

#### 🔗 Correlation Analysis: Feature Dependencies

Correlation heatmap reveals multicollinearity and predictive relationships.

![Correlation Heatmap](images/eda_correlation_heatmap.png)

**Key Findings:**

**Strong Positive Correlations:**
- `Power_HP` ↔ `Displacement_cm3`: **0.81** (expected: larger engines = more power)
- `Price` ↔ `Power_HP`: **0.58** (high-performance = expensive)
- `Price` ↔ `Production_year`: **0.52** (newer = pricier)

**Strong Negative Correlations:**
- `Production_year` ↔ `Vehicle_age`: **-0.99** (redundant: age = 2026 - year)
- `Price` ↔ `Vehicle_age`: **-0.45** (older = cheaper, moderate due to vintage cars)

**Moderate Correlations:**
- `Price` ↔ `Displacement_cm3`: **0.44** (engine size matters, but not linearly)
- `Number_of_features` ↔ `Vehicle_age`: **-0.38** (newer cars = better equipped)

**Weak Correlations:**
- `Price` ↔ `Doors_number`: **-0.25** (minimal impact)

**Important Notes:**
- Correlation **≠** feature importance (tree models capture non-linear patterns)
- Moderate correlation (0.4-0.6) is still valuable for prediction
- `Production_year` and `Vehicle_age` are redundant → remove one

**Feature engineering decisions:**
- Create `HP_per_liter` to reduce `Power_HP` ↔ `Displacement_cm3` multicollinearity
- Use `Vehicle_age` instead of `Production_year` (more interpretable)
- Consider polynomial and interaction terms for moderate correlations

---

#### 📋 EDA Summary

Overall, the EDA provided crucial insights into distributions, feature importance, and relationships, informing subsequent preprocessing, feature engineering, and modeling steps.

---

### 🔧 Feature Engineering

To capture the complex, non-linear dynamics of the automotive market, a robust feature engineering strategy and automated preprocessing pipeline were implemented. The goal was to transform raw categorical and numerical data into high-signal inputs while maintaining strict separation between training and test sets to prevent **data leakage**.

---

#### 1. Domain-Driven Feature Synthesis

Several new features were engineered to better represent vehicle depreciation, performance, and market positioning:

- **Operational Metrics**  
  - `mileage_per_year` and `usage_intensity` distinguish between “highway cruisers” and “city-driven” vehicles of the same age.

- **Performance Ratios**  
  - `hp_per_liter` (specific power) captures differences between high-performance modern engines and older, less efficient blocks.

- **Market Segmentation (Heuristics)**  
  - `is_premium` & `is_supercar`: Binary flags based on brand prestige and power thresholds.  
  - `is_collector`: Identifies vintage vehicles where value is driven by rarity rather than utility.  
  - `age_category`: Discretizes vehicle age into lifecycle stages (New, Standard, Old, Vintage).

---

#### 2. Non-Linear & Interaction Modeling

Since car prices do not depreciate linearly, mathematical transformations were applied to assist the XGBoost regressor:

- **Polynomial Features**  
  - Squared terms for `vehicle_age`, `power_hp`, and `mileage_km` capture accelerating depreciation in early years.

- **Interaction Terms**  
  - `age_mileage_interaction` and `power_age_interaction` reflect how the effect of high mileage or power changes depending on vehicle age.

- **Logarithmic Scaling**  
  - Applied `log(x + 1)` transformations to highly skewed features (e.g., price, mileage, power) to stabilize variance and reduce the influence of extreme outliers during gradient descent.

---

#### 3. Automated Preprocessing & Encoding Pipeline

A `ColumnTransformer` pipeline was designed to handle diverse data types with tailored strategies:

- **Numerical Imputation**  
  - Median imputation for missing technical specifications, ensuring robustness to outliers.

- **Categorical Encoding**  
  - *OneHotEncoder*: Applied to low-cardinality features (e.g., `fuel_type`, `transmission`) to avoid imposing an artificial order.  
  - *Target Encoding with Smoothing*: Used for high-cardinality features like `vehicle_model`. A smoothing factor of 500 prevents the model from overfitting to rare models with few observations.

- **Standardization**  
  - Continuous features were scaled using `StandardScaler` to improve convergence for gradient boosting.

---

#### 4. Data Integrity & Leakage Prevention

- **Temporal Consistency**  
  - Rows with irrecoverable data (missing target or critical identifiers) were removed before splitting.

- **Pipeline Isolation**  
  - All transformations (imputation, scaling, encoding) were **fitted only on the training set** and then applied to the test set, ensuring a valid, unbiased evaluation of model performance.

This approach ensures that the model can learn complex, non-linear relationships while maintaining reliability, stability, and strict adherence to data science best practices.

---

### 📈 Model Training & Performance

### 🏗 Modeling & Iterative Development

The modeling phase followed an **incremental complexity approach**, progressing from interpretable baselines to high-performance ensemble methods. Each model was evaluated using **R², MAE, RMSE, and MAPE** to track improvements in predictive accuracy and error reduction.

---

#### 1. Baseline: Linear Regression (Naive Benchmark)

- **Role:** Establish a performance "floor" for comparison.  
- **Outcome:** Achieved an R² of 83.1% with a high MAPE of 29.3%.  
- **Insights:**  
  - Confirmed that car depreciation is **not a simple linear process**.  
  - Failed to capture accelerated early depreciation and the “premium effect” of specific brands.  
  - Useful as a reference for improvement.

**Performance Details:**  
The Linear Regression model explains approximately 83% of the variance in car prices. Training and test R² scores are consistent (0.783 vs 0.831), indicating reasonable generalization.  

- **MAE:** 14,798 PLN — average deviation from actual prices.  
- **MAPE:** 29.3% — relative error across the dataset.  
- **RMSE:** 34,358 PLN — indicates sensitivity to outliers, particularly luxury, supercars, and rare collector vehicles.  

*Conclusion:* Linear Regression provides a solid baseline but cannot capture complex, non-linear relationships in the market.

---

#### 2. Non-Linearity: Random Forest Regressor

- **Role:** Capture **non-linear relationships** and feature interactions.  
- **Outcome:** R² increased to 93.8%.  
- **Insights:**  
  - Bagged decision trees identified thresholds in mileage and age (e.g., crossing 100k km).  
  - Effectively modeled non-linear depreciation patterns.  
  - Struggled with extreme variance in luxury and vintage segments.

**Performance Highlights:**  
- **Test MAPE:** ~20% — more accurate predictions for most vehicle segments.  
- **Error Reduction:** MAE reduced by ~50% compared to Linear Regression.  
- **Generalization:** Training and test metrics are balanced, showing minimal overfitting.

*Conclusion:* Random Forest significantly improves predictive power and captures complex interactions in pricing dynamics.

---

#### 3. Gradient Boosting: XGBoost (Base Model)

- **Role:** Sequentially boost weak learners to minimize residuals.  
- **Outcome:** Peak R² = 94.3%, MAPE = 16.8%.  
- **Insights:**  
  - Highly sensitive to engineered features and interactions.  
  - Strongest performer on training data but overfits, as indicated by learning curve gaps.

**Performance Details:**  
- **RMSE:** 19,970 PLN  
- **MAE:** 7,990 PLN  
- **MAPE:** 16.8%  

*Conclusion:* XGBoost captures non-linear relationships more effectively than Random Forest, though base models can overfit without regularization.

---

#### 4. Final Optimization: XGBoost (Hyperparameter-Tuned)

- **Role:** Refine the model for **generalization** using **Optuna hyperparameter optimization**.  
- **Strategy:**  
  - Applied strong regularization (Gamma, Alpha, Lambda).  
  - Early stopping and smoothing strategies enforced stable learning.  
- **Outcome:** R² = 92.5%, balanced train and test performance.  
- **Insights:**  
  - Slight decrease in raw metrics offset by **robust generalization**.  
  - Handles non-linear interactions without overfitting, suitable for real-market deployment.

**Metrics on Test Set:**  
- **RMSE:** 22,918 PLN — larger errors occur mainly for high-priced or rare vehicles.  
- **MAE:** 8,062 PLN — average deviation from actual prices.  
- **MAPE:** 17.2% — acceptable for a diverse market with prices ranging from a few thousand PLN to millions.  

---

#### 🛠 Error Analysis

The main contributors to prediction errors are **rare and niche vehicles** (e.g., Syrena, Warszawa, Nysa) and **luxury supercars** (e.g., Lamborghini, Aston Martin, Rolls-Royce). These segments are underrepresented in the dataset, causing higher residuals.  

- **Observation:** Vehicles older than 30 years and high-end supercars exhibit the largest residuals.  
- **RMSE for old vehicles:** ~58,318 PLN, roughly three times higher than for newer cars.  
- **Residual Patterns:** Scatter plots confirm high errors in rare and collector vehicles, while mass-market cars remain well-predicted.

**Recommendation:** Additional feature engineering to explicitly capture rarity, collector status, and luxury brand membership may reduce errors for these extreme cases.

![Error Analysis](images/corrected_residuals_vs_year_of_production_xgb_before_cleaning.png)

---

#### 📊 Learning Curves

The learning curve shows a **healthy bias–variance trade-off**:

- Training and validation curves converge smoothly, indicating minimal overfitting.  
- Training error does not approach zero, confirming that the model generalizes rather than memorizes.  
- Increasing the dataset size further (e.g., +50k samples) is unlikely to substantially improve performance.  

![Learning Curves](images/tuned_model_learning_curves.png)

---

This **staged modeling approach** demonstrates the value of progressing from interpretable baselines to advanced ensemble methods, culminating in a **production-ready XGBoost model** optimized for accuracy, stability, and generalizability.

- **Hyperparameter Tuning:** Employed Optuna for Bayesian search in `src/model.py`.  
- **Tuned Parameters:** learning rate, max depth, subsample ratio, `reg_alpha`, `reg_lambda`.

![Model Comparison](images/model_comparison.png)

#### 🏁 Model Selection & Performance

- **Linear Regression (Model 1):** Baseline model; R² = 83.1%, MAPE = 29.3%. Could not capture non-linear depreciation or brand premiums.  

- **Random Forest (Model 2):** R² = 93.8%. Captures non-linear relationships and feature interactions (age, mileage, brand) but struggles with luxury/vintage vehicles.  

- **Base XGBoost (Model 3):** R² = 94.3%, MAPE = 16.8%. High raw performance but showed overfitting on training data.  

- **Tuned XGBoost (Model 4, Production-Ready):** Optimized via Optuna with stronger regularization and smoothing. R² ≈ 92.6%, MAE ~8,000 PLN, MAPE ~17%. Offers **robust generalization**, handling typical market vehicles reliably while acknowledging higher errors for rare or luxury cars.  

> Model 4 prioritizes **stability and real-world applicability** over marginally higher but overfitted metrics, making it ideal for deployment.

---

## 🚀 Deployment

- **Model Serialization:** The final Tuned XGBoost model was saved using `joblib.dump` and uploaded to the **Hugging Face Hub** for versioned storage and easy access.  

- **Interactive Web App:** A **Streamlit** application (`app.py`) allows users to input vehicle specifications and receive real-time price predictions.  

- **Automated Deployment:** The app is deployed via Streamlit sharing, providing a live demo link for immediate interaction.

### 🖥 Application Interface (Streamlit)

When launching the application (built with **Streamlit**), the user is presented with a simple interface containing two main sections.

![Interface1](images/app_interface1.png)

The **right card** contains a short description of the project, explaining:
- how the prediction model works,
- why **XGBoost** was selected as the final algorithm,
- the reasoning behind the modeling approach,
- key business insights and performance metrics.

The **second card** is responsible for the **car price prediction tool**.  
Users must enter several vehicle attributes so the model can estimate the price, including:

- mileage
- production year
- engine displacement
- engine power
- body type
- transmission
- fuel type

![Interface2](images/app_interface2.png)

---

### 🔎 Example Prediction – Hyundai i20

To validate the model, we can test it on a real-world example.  
The first example is a **Hyundai i20**, where we enter realistic values such as mileage, production year, displacement, and engine power.

![Interface3](images/app_interface3.png)

Vehicles with similar specifications (production year, mileage, power, and displacement) typically fluctuate between **12,000 PLN and 20,000 PLN** depending on:

- vehicle condition (e.g., accident history),
- additional features or equipment,
- maintenance and ownership history.

Although the model does not have access to detailed vehicle history, the predicted price falls within the **realistic market range**, indicating good model performance.

---

### 🔎 Example Prediction – Audi RS3

The second example is an **Audi RS3**, which belongs to a premium vehicle segment and therefore has a significantly higher market value.

After entering the required specifications (production year, mileage, power, displacement, etc.), the model estimates the price at approximately **250,000 PLN**.

![Interface4](images/app_interface4.png)

Comparing this prediction with similar listings for **Audi RS3 models from 2024** with comparable specifications (sedan body type, engine power, displacement, and mileage), the estimated price is **consistent with current market offers**.

As mentioned earlier, the final market price of a vehicle often depends on factors such as optional features, condition, and ownership history. While this information is not available in the dataset, the model still produces **reliable and realistic predictions** for typical vehicles.

---

## 📊 Results & Business Impact

The final **Tuned XGBoost model (Model 4)** was selected for deployment due to its superior generalization and stable error distribution. While the base XGBoost model showed slightly higher raw accuracy, the tuned version reduced overfitting, providing a **reliable foundation for automated commercial vehicle valuation**.

### Comparative Performance & Strategic Value

#### 1. Precision-Driven Valuation & Margin Protection
- Achieved a **45.5% reduction in Mean Absolute Error (MAE)** compared to the linear baseline.  
- Reduced average error to ~8,000 PLN across a highly volatile market.  
- Ensures tighter pricing spreads, protecting profit margins for dealerships and fleet managers.

#### 2. Market Responsiveness & Dynamic Pricing
- Captures real-time effects of mileage, age, engine power, and other interactions.  
- Enables dealerships to adjust prices dynamically based on market trends and depreciation patterns.

#### 3. Strategic Insights
- Identifies patterns in vehicle depreciation and market demand.  
- Helps forecast inventory turnover and informs purchase/sales strategies.

#### 4. Scalability & Automation
- Can process thousands of vehicles quickly with minimal human intervention.  
- Makes large-scale fleet or marketplace management feasible and efficient.

---

## 🛠 Tech Stack
* **Python 3.12+**
* **Pandas, NumPy, Scikit‑Learn** for data manipulation and pipelines
* **XGBoost** for gradient boosting
* **Optuna** for hyperparameter optimization
* **Matplotlib, Seaborn** for visualization
* **Joblib** for model serialization
* **Requests** for API interactions
* **Streamlit** for deployment

---

## 📥 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Car-Price-Prediction.git
   cd Car-Price-Prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run preprocessing and training (optional):**
   ```bash
   python src/model.py --train
   ```
4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
5. **Interact with the app** or inspect notebooks in `notebooks/` for experiments.

---

## 🔮 Future Work
* **NLP / text features:** Incorporate descriptions using Polish-language BERT models from Hugging Face.
* **Docker & CI/CD:** Containerize the application and add GitHub Actions for automated testing and retraining.
* **Ensemble strategies:** Try stacking XGBoost with LightGBM or CatBoost.
* **Real-time API:** Wrap the model with a REST API for integration into dealer platforms.
* **Data expansion:** Continuously scrape new ads to keep the model up‑to‑date with market trends.

<div align="center">

**⭐ If you found this project helpful, please star the repository!**

[![GitHub stars](https://img.shields.io/github/stars/Przemsonn/car-price-prediction?style=social)](https://https://github.com/Przemsonn05/Cars-Price-Prediction-in-Poland)

</div>

---