##training ml project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from pydantic import BaseModel

class Members(BaseModel):
    exercise: str
    age: int
    gender: str
    duration: int
    heartRate: int
    bmi: int
    weatherConditions: str

def mygymapp(members:Members):
    print("hellow")


    state = np.random.RandomState(12345)

    df = pd.read_csv('C:\\Users\\anuda\\OneDrive\\Desktop\\reate_cirse_work\\backendreact\\src\\Dataset\\exercise_dataset.csv')

   
    df.head()
    print(df.head())

    df.info()
    print(df.info())

    df.drop('ID', axis=1, inplace=True)
    df['Exercise'] = df['Exercise'].map(lambda x: ''.join([i for i in x if i.isdigit()]))

    ints = [
    'Exercise', 
    'Age', 
    'Duration', 
    'Heart Rate', 
    'Exercise Intensity'    
    ]

    categories = [
        'Gender', 
        'Weather Conditions'
    ]

    for col in ints:
        df[col] = df[col].astype('int16')

    for col in categories:
        df[col] = df[col].astype('category')

    df.describe()
    print(df.describe())


    _, axs = plt.subplots(3, 2, figsize=[10,15])
    cols = [
        (axs[0,0], 'Calories Burn'),
        (axs[0,1], 'Dream Weight'),
        (axs[1,0], 'Actual Weight'),
        (axs[1,1], 'Age'),
        (axs[2,0], 'Duration'),
        (axs[2,1], 'Heart Rate'),
    ]
    for i, (ax, col) in enumerate(cols):
        ax.set(title=col)
        ax.hist(df[col], bins=40)
    plt.figure(figsize=[10,4])
    plt.title('BMI')
    plt.hist(df['BMI'], bins=40)

    _, axs = plt.subplots(2, 2, figsize=[11,11])
    cols = [
        (axs[0,0], 'Exercise'),
        (axs[0,1], 'Exercise Intensity'),
        (axs[1,0], 'Gender'),
        (axs[1,1], 'Weather Conditions')
    ]

    cmap = plt.get_cmap('Blues')

    for i, (ax, col) in enumerate(cols):
        ax.set(title=col)
        values = df[col].value_counts().sort_index()
        ind = values.index
        colors = list(cmap(np.linspace(0.1, 0.8, len(values))))
        ax.pie(
            values, 
            labels=ind, 
            autopct='%.1f%%', 
            colors=colors
    )
        
    cols = [
    'Calories Burn', 
    'Dream Weight',
    'Actual Weight',
    'Age',
    'Duration',
    'Heart Rate',
    'BMI',
    'Exercise Intensity'
    ]

    sns.heatmap(df.phik_matrix(interval_cols=cols), cmap ='seismic')

    def gain(x):
        if x < 0:
            return 'Gain'
        else: return 'Lose'

    df['Weight Difference'] = df['Actual Weight'] - df['Dream Weight']
    df['Gain'] = df['Weight Difference'].apply(gain).astype('category')
    df['Weight Difference'] = abs(df['Weight Difference'])

    _, axs = plt.subplots(1, 2, figsize=[11,4])
    cols = [
        (axs[0], 'Weight Difference'),
        (axs[1], 'Gain')
    ]
    for i, (ax, col) in enumerate(cols):
        ax.set(title=col)
        ax.hist(df[col], bins=40)
    
    colors = cmap(np.linspace(0.1, 0.8, 2))
    df.pivot_table(
        columns='Gain',
        index='Gender',
        values='Exercise',
        aggfunc='count'
    ).plot(
        kind='pie', 
        autopct='%.1f%%',
        colors=colors, 
        subplots=True, 
        figsize=[11,11], 
        legend=False
    )

    colors = cmap(np.linspace(0.1, 0.8, 2))
    df.pivot_table(
        columns='Gender',
        values=['Actual Weight', 'Dream Weight'],
        aggfunc='median'
    ).plot(
        kind='bar', 
        figsize=[11,5], 
        rot=0, 
        color=colors
    );

    colors = cmap(np.linspace(0.1, 0.8, 4))
    df.pivot_table(
        columns='Gender',
        values=['Actual Weight', 'Dream Weight'],
        aggfunc=['min', 'max']
    ).plot(
        kind='bar', 
        figsize=[11,5], 
        rot=0, 
        color=colors
    );

    #MODEL TRAINING

    cat_features = [
    'Exercise', 
    'Gender', 
    'Weather Conditions', 
    'Gain'
    ]

    num_features = [
        'Dream Weight', 
        'Actual Weight', 
        'Age', 
        'Duration', 
        'Heart Rate', 
        'BMI', 
        'Weight Difference'
    ]

    target = 'Exercise Intensity'

    X_train, X_test, y_train, y_test = train_test_split(
        df[cat_features+num_features], 
        df[target], 
        test_size=0.33, 
        random_state=state
    )


    # Pipeline stuff
    # adding imputer in case future df updates will be with NaNs

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('normalizer', Normalizer())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('encoder', OrdinalEncoder())
        ]
    )

    preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, num_features),
        ('categorical', categorical_transformer, cat_features)
    ]
    )


    regressors = [
    KNeighborsRegressor(), 
    DecisionTreeRegressor(), 
    RandomForestRegressor(), 
    GradientBoostingRegressor()
    ]

    models = []
    scores = []

    for regressor in regressors:
        steps = [
            ('preprocess', preprocessor),
            ('reg', regressor)
        ]
        pipeline = Pipeline(steps)
        scorer = cross_val_score(
            pipeline, 
            X_train, 
            y_train, 
            cv=5,
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        models.append(str(regressor))
        scores.append(scorer.mean())

    plt.figure(figsize=(10,5))
    plt.barh(models, scores)
    plt.show();



    loss = ['quantile', 'squared_error', 'absolute_error', 'huber']
    max_features = ['sqrt', 'log2', None]
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 15)]
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 15)]
    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 15)]

    hyperparameter_grid = {
        'reg__loss': loss,
        'reg__max_features': max_features,
        'reg__n_estimators': n_estimators,
        'reg__max_depth': max_depth,
        'reg__min_samples_split': min_samples_split,
        'reg__min_samples_leaf': min_samples_leaf
    }

    random_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=hyperparameter_grid,
        cv=3,
        n_iter=200,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        random_state=state
    )

    random_cv.fit(X_train, y_train)

    print()
    print('Best params:')
    print(random_cv.best_params_)
    print()
    print('Best score:', random_cv.best_score_)

    rs_df = pd.DataFrame(random_cv.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    rs_df.loc[rs_df['param_reg__max_features'].isna(), 'param_reg__max_features'] = 'None'

    cols = [
        'param_reg__loss', 
        'param_reg__max_features',
        'param_reg__n_estimators',
        'param_reg__max_depth',
        'param_reg__min_samples_split',
        'param_reg__min_samples_leaf'
    ]
    pref = 'param_reg__'

    fig, axs = plt.subplots(ncols=2, nrows=3)
    fig.set_size_inches(30,25)
    sns.set(font_scale=2)
    color = 'lightblue'
    i = 0
    j = 0

    for col in cols:
        sns.barplot(
            x=col,
            y='mean_test_score', 
            data=rs_df, 
            ax=axs[i,j], 
            color=color
        )
        axs[i,j].set_title(
            label=col.replace(pref, ''), 
            size=30, 
            weight='bold'
        )
        axs[i,j].set_xlabel('')
        j += 1
        if j == 2:
            i += 1
            j = 0

    loss = ['absolute_error']
    max_features = ['log2', None]
    n_estimators = range(45, 55)
    max_depth = range(1, 6)
    min_samples_split = range(2, 9, 2)
    min_samples_leaf = range(2, 9, 2)

    hyperparameter_grid = {
        'reg__loss': loss,
        'reg__max_features': max_features,
        'reg__n_estimators': n_estimators,
        'reg__max_depth': max_depth,
        'reg__min_samples_split': min_samples_split,
        'reg__min_samples_leaf': min_samples_leaf
    }

    grid_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=hyperparameter_grid,
        cv=3, 
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_cv.fit(X_train, y_train)
    best_params = grid_cv.best_params_

    print()
    print('Best params:')
    print(best_params)
    print()
    print('Best score:', grid_cv.best_score_)



    pipeline.set_params(**best_params)

    pipeline.fit(X_train, y_train)
    y_pred = np.round(pipeline.predict(X_test))
    print(f'MSE for test subset: {mean_squared_error(y_test, y_pred)}')


    return "hellow"
