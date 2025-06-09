# %%
import pandas as pd

# %%
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# %%
df

# %%
df.isna().sum().sum()

# %%
df['crop_disease_status'] = df['crop_disease_status'].fillna('None')
df['irrigation_type'] = df['irrigation_type'].fillna('None')

# %%
df.isna().sum().sum()

# %%
df.dtypes

# %%
import numpy as np

# %%
def ciclic_encoding(df, col, max_val):

  #result[date_col] = pd.to_datetime(result[date_col])


  df[col] = pd.to_datetime(df[col])
  month = df[col].dt.month


  df[col + '_sin'] = np.sin(2 * np.pi * month/max_val)
  df[col + '_cos'] = np.cos(2 * np.pi * month/max_val)


  return df

# %%
df = ciclic_encoding(df, 'harvest_date', 12)
df = ciclic_encoding(df, 'sowing_date', 12)

# %%
#df['sowing_date'] = pd.to_datetime(df['sowing_date'], format='%Y-%m-%d')
#df['harvest_date'] = pd.to_datetime(df['harvest_date'], format='%Y-%m-%d')

# %%
df.drop(['farm_id', 'sensor_id', 'timestamp', 'latitude', 'longitude', 'sowing_date', 'harvest_date'], axis=1, inplace=True)

# %%
df.dtypes

# %%
df

# %%
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_absolute_percentage_error, make_scorer
import sklearn
import optuna

# %%
x = df.drop(columns=['yield_kg_per_hectare'])
y = df['yield_kg_per_hectare']

# %%
# per ricevere in input un dataframe e dare in output un dataframe
sklearn.set_config(transform_output='pandas')

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
encoding = ColumnTransformer(
    [
        ('encoder', "passthrough", ['crop_disease_status', 'irrigation_type', "region", "fertilizer_type", "crop_type"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

# %%
pipe = Pipeline(
    [
        ("encoder", encoding),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor())
    ]
)

# %% [markdown]
# # Model selection

# %%
from scipy.stats import randint

# %%
pipe.get_params()

# %%
params = [
    {
        'model__n_estimators' : randint(low=100, high=200),
        'model__criterion': ['squared_error', 'absolute_error'],
        'encoder__encoder': [OneHotEncoder(sparse_output=False, drop="first")],
    },
    {
        'model__n_estimators' : randint(low=100, high=200),
        'model__criterion': ['squared_error', 'absolute_error'],
        'encoder__encoder': [TargetEncoder(target_type="float")]
    },
    {
        'model__n_estimators' : randint(low=100, high=200),
        'model__criterion': ['squared_error', 'absolute_error'],
        'encoder__encoder': [OrdinalEncoder()],
    },  
]

# %%
grid_seach = RandomizedSearchCV(
    estimator=pipe,
    n_iter=20,
    param_distributions=params,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    # n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    refit=True,
    verbose=4
)

# %%
grid_seach.fit(x_train, y_train)

# %%
encoding = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="first"), ['crop_disease_status', 'irrigation_type', "region", "fertilizer_type", "crop_type"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)

# %%
pipe = Pipeline(
    [
        ("encoder", encoding),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(criterion="absolute_error", n_estimators=184))
    ]
)

# %%
pipe.fit(x_train, y_train)

# %%
y_test_pred = pipe.predict(x_test)

# %%
mean_absolute_percentage_error(y_test, y_test_pred) * 100

# %%
mean_absolute_error(y_test, y_test_pred)

# %% [markdown]
# # Optuna

# %%
study = optuna.create_study(storage="sqlite:///model_selection.db", study_name="study", direction="minimize")

# %%
def objective_func(trial):
    model__n_estimators = trial.suggest_int("n_estimators", 100, 400)
    model__criterion = trial.suggest_categorical("criterion", ['squared_error', 'absolute_error'])
    encoder__name = trial.suggest_categorical("encoder", ["onehot", "ordinal"])

    if encoder__name == "onehot":
        encoder = OneHotEncoder(sparse_output=False, drop="first")
    elif encoder__name == "ordinal":
        encoder = OrdinalEncoder()

    pipe.set_params(
        model__n_estimators=model__n_estimators,
        model__criterion=model__criterion,
    )

    values = cross_validate(
        pipe,
        x_train,
        y_train,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=KFold(shuffle=True, random_state=42))


    return abs(sum(values["test_score"]) / len(values["test_score"]))

# %%
study.optimize(objective_func, n_trials=20)

# %%



