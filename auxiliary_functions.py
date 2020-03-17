#Basic imports: pandas, numpy, visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Imports to predictive models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def get_dummy_columns(df, column):
    '''
    USAGE: 
        It transforms a column into dummy columns and save their name into a list.
        It also saves the name of the category used as intercept.
        
    INPUT
        df: Dataframe with the categorical column tha we want to convert.
        column: Column that we want to convert in dummy columns
    OUTPUT
        df: The same dataframe with the dummy columns and the original column removed
        columns: A list with the name of the dummy columns.
        intercept: The name of the dummy column used as intercept.
    '''
    
    #It creates a list with with the names of the future dummy columns
    df = df.dropna(subset = [column])
    value_list = list(df[column].unique())
    columns = [column + '_' + s for s in value_list]
    
    #Separate the columns used as dummy and the one used as intercept
    columns.sort()
    intercept = columns[0]
    
    #Apply the method get_dummies of pandas.
    df = pd.concat([df.drop(column, axis=1), 
                    pd.get_dummies(df[column], 
                                   prefix=column, 
                                   prefix_sep='_', 
                                   drop_first=False)], 
                   axis=1)
    
    return df, columns, intercept



def scatter_map(df, map_pic, color_col, size_col, box, scale_size, annot_limit, figsize = (10,16)):
    '''
    USAGE: 
        It draws a scatter plot wiht a picture of a map in the background
    INPUT    
        df: Dataframe with the columns 'logngitud', 'latitude', a column with the size of the points
            and another column for the colors of the points.    
        map_pic: picture with the map downloaded for example from OpenStreetMaps.    
        color_col: the column that represents the colors in the scatter plot.    
        size_col: the column that represents the size of the points.   
        box: maximum and minimum coordenates of the map   
        scale_size: a factor to multiply for the value in size_col to see better the points.
        figsize: a tuple with the size of the plot.
        annot_limit:  only puts annotation in the points where size_col is greater than this value. 
    OUTPUT
        ax: an Axe object with the compostition of the scatter plot
    '''
    colors = {'high_price':'red', 'medium_price':'blue', 'low_price':'green'}
    
    #It loads the picture with the map 
    ruh_m = plt.imread(map_pic)
    
    #It creates a figure with an Axe
    fig, ax = plt.subplots(figsize = figsize)
    
    
    sc = ax.scatter(df.longitude, df.latitude, 
               zorder=1, alpha= 0.4, 
               c=df[color_col].apply(lambda x: colors[x]), 
               s=df[size_col] * scale_size)
   
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
     
    #produce a legend with the elements of colors
    circle1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", alpha = 0.5, markersize=20)
    circle2 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue", alpha = 0.5, markersize=20)
    circle3 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", alpha = 0.5, markersize=20)
    legend1 = ax.legend([circle1,circle2,circle3],
                        ["Low Price", "Medium Price", "High Price"],
                        fontsize = 15, 
                        title="Neighbourhood Group", 
                        title_fontsize = 'x-large',
                        loc="upper left")
    ax.add_artist(legend1)
    
    #produce a legend with the elements of sizes
    handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
    kw = dict(prop="sizes", num=5, color=sc.cmap(0.7), fmt="{x:.0f}",
              func=lambda s: s/scale_size)
    legend2 = ax.legend(*sc.legend_elements(**kw),
                        loc="upper right", title=size_col, fontsize = 15, title_fontsize = 'x-large')
    
    #put anotations in some points
    for i, row in df.iterrows():
        if row[size_col] > annot_limit:
            ax.annotate(row.neighbourhood_cleansed,(row.longitude, 
                                                    row.latitude), 
                        color='darkblue', 
                        fontsize = 12,
                        rotation=35)
            
    #Show the scatter plot with the map.
    ax.imshow(ruh_m, zorder=0, extent = box, aspect= 'equal');
    
    return ax


def prepare_feature_by_neighbourhood(df_boston, df_seattle, columns_boston, columns_seattle, feature_name, limit):
    '''
    USAGE: 
        It recives the dataframe for Boston and Seatle with the dummy columns we want to compare. It returns 
        a new dataframe with the mean of this columns for each neighbourhood group and for each city. It shows 
        us also the difference between the two cities.
    INPUT
        df_boston: dataframe with the data of Boston
        df_boston: dataframe with the data of Seattle
        columns_boston: list of dummy colums with the properties we want to compare for Boston
        columns_seattle: list of dummy colums with the properties we want to compare for Seattle
        feature_name: string with the name of the category
        limit: minimum differen between Boston and Seattle we want to compare.
    OUTPUT
        comp_df: a datraframe with five columns: 
            - One column named with the variabel feature_name. It has the name of each feature.
            - neighbourhood_group: three neighbourhood groups.
            - percentage_boston: the mean for each feature in each neighbourhood group in Boston
            - percentage_seattle:  the mean for each feature in each neighbourhood group in Seattle
            - percentage_diff: the difference between the two previous columns
    '''
    #Define an axiliar dataframe for boston with the percentage of each verification type per neighbourhood group
    df_boston_aux = df_boston.groupby(['neighbourhood_group']).mean()[columns_boston].reset_index()
    df_boston_aux = pd.melt(df_boston_aux, id_vars="neighbourhood_group", var_name=feature_name, value_name="percentage")
    df_boston_aux = df_boston_aux.set_index([feature_name, 'neighbourhood_group'])
    df_boston_aux.columns = ['percentage_boston']

    #we repeat the same steps as before but with seattle
    df_seattle_aux = df_seattle.groupby(['neighbourhood_group']).mean()[columns_seattle].reset_index()
    df_seattle_aux = pd.melt(df_seattle_aux, id_vars="neighbourhood_group", var_name=feature_name, value_name="percentage")
    df_seattle_aux = df_seattle_aux.set_index([feature_name, 'neighbourhood_group'])
    df_seattle_aux.columns = ['percentage_seattle']

    #we do a concat with the information of boston and seattle in a single dataframe
    comp_df = pd.concat([df_boston_aux, df_seattle_aux], axis=1, join = 'outer')
    #some amenities do not exist in both cities, so we fill them with zeros.
    comp_df = comp_df.fillna(0)

    #add a new column with the tifference of the percentages for Boston and Seattle
    comp_df['percentage_diff'] = ((comp_df['percentage_boston'] - comp_df['percentage_seattle'])*100).astype(int)

    #It drops out those Verification Types where the maximum difference for the three neighbourhood grousp is smaller
    #than 15%. There are a lot of Verification Types, so we get only the most relevant differences.
    comp_df = comp_df.reset_index()
    property_list = comp_df[feature_name].unique()
    for prop in property_list:    
        if (max(abs(comp_df[comp_df[feature_name] == prop].percentage_diff)) <= limit):
            comp_df = comp_df[comp_df[feature_name] != prop]



    #It sets also neighbourhood_group as a categorical ordered variable
    neig_order = ['high_price', 'medium_price', 'low_price']
    ordered_neig = pd.api.types.CategoricalDtype(ordered = True, categories = neig_order) 
    comp_df['neighbourhood_group'] = comp_df['neighbourhood_group'].astype(ordered_neig)   

    #It defines a order for the Verification Types to draw the fifference in descending order.        
    feat_type_order = list(comp_df.groupby([feature_name]).max(). #group by amenities and get the max for the three groups
                           sort_values(by = 'percentage_diff', ascending = False). #sort the result values
                           index)  #get the index of Verification Type as a list
    
    #It sets feature_names as a categorical ordered variable
    ordered_feat = pd.api.types.CategoricalDtype(ordered = True, categories = feat_type_order)  
    comp_df[feature_name] = comp_df[feature_name].astype(ordered_feat)

    #It orderes the dataframe by feature_name and 'neighbourhood_group'
    comp_df = comp_df.sort_values(by = [feature_name, 'neighbourhood_group'])
    comp_df = comp_df.reset_index() 
    
    return comp_df


def apply_lm_mod(X, y, test_size = .30, random_state=42, normalize = True):
    '''
    USAGE: 
        A function that applies the Linear Regression model over a matrix of features X and a response y
    INPUT
        X - pandas dataframe, X matrix
        y - pandas dataframe, response variable
        random_state - int, default 42, controls random state for train_test_split
        normalize - boolean, normalizes or not.

    OUTPUT
        coefs_df - a dataframe with the coeficients
        lm_model.intercept_ - The value of the intercept
        r2_scores_test_max - list of floats of r2 scores on the test data
        r2_scores_train_max - list of floats of r2 scores on the train data
    '''
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

    #fit the model and obtain pred response
    lm_model = LinearRegression(normalize)
    lm_model.fit(X_train, y_train)
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #append the r2 value from the test set
    r2_scores_test = r2_score(y_test, y_test_preds)
    r2_scores_train = r2_score(y_train, y_train_preds)
    
    #prepare a dataframe with the coeficients to be reported
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('coefs', ascending=False)
    
    lm_model.intercept_
    
    return coefs_df, lm_model.intercept_, r2_scores_test, r2_scores_train
    
     
def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    USAGE: 
        A function to reduce the number of features applying cutoffs (maximum number of zeros in a variable).
        Finally use the Liner Regression model for each cutoff.
    INPUT
        X - pandas dataframe, X matrix
        y - pandas dataframe, response variable
        cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
        test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
        random_state - int, default 42, controls random state for train_test_split
        plot - boolean, default 0.3, True to plot result

    OUTPUT
        coefs_df - a dataframe with the coeficients
        r2_scores_test_max - list of floats of r2 scores on the test data
        r2_scores_train_max - list of floats of r2 scores on the train data
        lm_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, mse_test, mse_train, num_feats, results = [], [], [], [], [], dict()
    
    for cutoff in cutoffs:
        
        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model  = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        
        #calculates the mse and r2 for the train dataset
        y_train_preds= lm_model.predict(X_train)
        mse_train.append(np.sqrt(mean_squared_error(y_train,y_train_preds)))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        
        #calculates the mse and r2 for the test dataset
        y_test_preds= lm_model.predict(X_test)
        mse_test.append(np.sqrt(mean_squared_error(y_test,y_test_preds))) 
        r2_scores_test.append(r2_score(y_test, y_test_preds))

        #append the r2 value from the test set
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.subplots(figsize=(15,7))
        
        #plot the results for mse values
        plt.subplot(1, 2, 1)
        plt.plot(num_feats, mse_test, label="Test", alpha=.5)
        plt.plot(num_feats, mse_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('MSE by Number of Features')
        plt.legend(loc=1)
        
        #plot the results for r-squared values
        plt.subplot(1, 2, 2)
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=4)
        plt.show()
    
    #the cutoff with the highest r-squared
    best_cutoff = max(results, key=results.get)
    
    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)
    
    #load the coeficients in a dataframe to be reported
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('coefs', ascending=False)
    
    #calculates the mse and r2 for the train dataset
    y_train_preds= lm_model.predict(X_train)
    mse_train_min = np.sqrt(mean_squared_error(y_train,y_train_preds))
    r2_scores_train_max = r2_score(y_train, y_train_preds)

    #calculates the mse and r2 for the test dataset
    y_test_preds= lm_model.predict(X_test)
    mse_test_min = np.sqrt(mean_squared_error(y_test,y_test_preds))
    r2_scores_test_max = r2_score(y_test, y_test_preds)

    return coefs_df, r2_scores_test_max, r2_scores_train_max, X_train, X_test, y_train, y_test, lm_model
    
    

def find_optimal_lr_mod(X, y, alpha_list, model = 'Lasso', test_size = .30, random_state=42, plot=True, normalize = True):
    '''
    USAGE: 
        A function to apply Ridge and Lasso.
    INPUT
        X - pandas dataframe, X matrix
        y - pandas dataframe, response variable
        alpha_list - a list with values of alpha
        test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
        random_state - int, default 42, controls random state for train_test_split
        plot - boolean, default 0.3, True to plot result
        normalize - apply normalization or not

    OUTPUT
        coefs_df: a dataframe with the coefiecients
        r2_scores_test_max - list of floats of r2 scores on the test data
        r2_scores_train_max - list of floats of r2 scores on the train data
        lm_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, mse_test, mse_train, num_feats, results = [], [], [], [], [], dict()
    
    for alpha in alpha_list:

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        if model == 'Lasso':
            lm_model = Lasso(alpha, normalize)
        elif model == 'Ridge':
            lm_model = Ridge(alpha, normalize)
            
        lm_model.fit(X_train, y_train) 
        
        #calculates the mse and r2 for the train dataset
        y_train_preds= lm_model.predict(X_train)
        mse_train.append(np.sqrt(mean_squared_error(y_train,y_train_preds)))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        
        #calculates the mse and r2 for the test dataset
        y_test_preds= lm_model.predict(X_test)
        mse_test.append(np.sqrt(mean_squared_error(y_test,y_test_preds))) 
        r2_scores_test.append(r2_score(y_test, y_test_preds))

        #append the r2 value from the test set
        results[str(alpha)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.subplots(figsize=(15,7))
        
        #plot the results for mse values
        plt.subplot(1, 2, 1)
        plt.plot(alpha_list, mse_test, label="Test", alpha=.5)
        plt.plot(alpha_list, mse_train, label="Train", alpha=.5)
        plt.xlabel('Lambda')
        plt.ylabel('Rsquared')
        plt.title('MSE by Alpha')
        plt.legend(loc=1)
        
        #plot the results for r-squared values
        plt.subplot(1, 2, 2)
        plt.plot(alpha_list, r2_scores_test, label="Test", alpha=.5)
        plt.plot(alpha_list, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Lambda')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Alpha')
        plt.legend(loc=4)
        plt.show()
    
    #the value of alpha with a biggest r-squeared
    best_alpha = max(results, key=results.get)

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)

    #fit the model
    if model == 'Lasso':
        lm_model = Lasso(float(best_alpha), normalize=True)
    else:
        lm_model = Ridge(float(best_alpha), normalize=True)
        
    lm_model.fit(X_train, y_train)
    
    #load the coeficients in a dataframe to be reported
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('coefs', ascending=False)
    
    #calculate the mse and r2 for the train dataset
    y_train_preds= lm_model.predict(X_train)
    mse_train_min = np.sqrt(mean_squared_error(y_train,y_train_preds))
    r2_scores_train_max = r2_score(y_train, y_train_preds)
    
    #calculates the mse and r2 for the test dataset
    y_test_preds= lm_model.predict(X_test)
    mse_test_min = np.sqrt(mean_squared_error(y_test,y_test_preds))
    r2_scores_test_max = r2_score(y_test, y_test_preds)

    return coefs_df, r2_scores_test_max, r2_scores_train_max, X_train, X_test, y_train, y_test, lm_model
    
    
def find_optimal_rf_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    USAGE: 
        A function that applies the Random Forest model.
        It receives a list of cutoffs as a parameter. This cutoffs determine the number of 0's
        that a feature can have to be included in the feature matrix.
    INPUT
        X - pandas dataframe, X matrix
        y - pandas dataframe, response variable
        cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
        test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
        random_state - int, default 42, controls random state for train_test_split
        plot - boolean, default 0.3, True to plot result
        kwargs - include the arguments you want to pass to the rf model
    
    OUTPUT
        r2_scores_test_max - list of floats of r2 scores on the test data
        r2_scores_train_max - list of floats of r2 scores on the train data
        rf_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    
    r2_scores_test, r2_scores_train, mse_test, mse_train, num_feats, results = [], [], [], [], [], dict()
    for cutoff in cutoffs:
        
        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response

        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        
        #calcs the predicted values for y int the test and train datasets
        y_test_preds = rf_model.predict(X_test)
        y_train_preds = rf_model.predict(X_train)
        
        #append the r2 and mse values from the test set
        mse_test.append(np.sqrt(mean_squared_error(y_test,y_test_preds)))
        mse_train.append(np.sqrt(mean_squared_error(y_train,y_train_preds)))
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)
    
    if plot:
        plt.subplots(figsize=(15,7))
        
        #plot the results for mse values
        plt.subplot(1, 2, 1)
        plt.plot(num_feats, mse_test, label="Test", alpha=.5)
        plt.plot(num_feats, mse_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('MSE')
        plt.title('MSE by Number of Features')
        plt.legend(loc=1)
        
        #plot the results for r-squared values
        plt.subplot(1, 2, 2)
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=4)
        plt.show()

    best_cutoff = max(results, key=results.get)
    
    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    rf_model = RandomForestRegressor() 
    rf_model.fit(X_train, y_train)
    
    #calculate the mse and r2 for the train dataset
    y_train_preds= rf_model.predict(X_train)
    mse_train_min = np.sqrt(mean_squared_error(y_train,y_train_preds))
    r2_scores_train_max = r2_score(y_train, y_train_preds)
    
    #calculate the mse and r2 for the test dataset
    y_test_preds= rf_model.predict(X_test)
    mse_test_min = np.sqrt(mean_squared_error(y_test,y_test_preds))
    r2_scores_test_max = r2_score(y_test, y_test_preds)

    return r2_scores_test_max, r2_scores_train_max, rf_model, X_train, X_test, y_train, y_test
    

def find_optimal_rf_gs_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True, param_grid=None):
    '''
    USAGE: 
        This function applies Random Forest over Grid Search Cross Validation
    INPUT
       X - pandas dataframe, X matrix
       y - pandas dataframe, response variable
       cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
       test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
       random_state - int, default 42, controls random state for train_test_split
       plot - boolean, default 0.3, True to plot result
       param_grid - include the arguments you want to pass to the rf model
    
    OUTPUT
       r2_scores_test_max - Best R-squared obtained in the test dataset
       r2_scores_train_max - Best R-squared obtained in the train dataset
       best_rf_model - best model obtained. Model object from sklearn
       X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''

    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        if param_grid==None:
            rf_model = RandomForestRegressor()  #no normalizing here, but could tune other hyperparameters
        else:
            rf_inst = RandomForestRegressor(n_jobs=-1, verbose=1)
            rf_grid = GridSearchCV(rf_inst, param_grid, n_jobs=-1) 
            
        rf_grid.fit(X_train, y_train)
        
        best_rf_model = rf_grid.best_estimator_
        best_rf_model.fit(X_train, y_train)
        
        y_test_preds = best_rf_model.predict(X_test)
        y_train_preds = best_rf_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.subplots(figsize=(8,8))
        #plot the results for r-squared values
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=4)
        plt.show()      
        
    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    if param_grid==None:
        rf_model = RandomForestRegressor()

    else:
        rf_inst = RandomForestRegressor(n_jobs=-1, verbose=1)
        rf_grid = GridSearchCV(rf_inst, param_grid, n_jobs=-1) 
    rf_grid.fit(X_train, y_train)
    
    best_rf_model = rf_grid.best_estimator_
    best_rf_model.fit(X_train, y_train)
    
    #calculate the mse and r2 for the train dataset
    y_train_preds= best_rf_model.predict(X_train)
    mse_train_min = np.sqrt(mean_squared_error(y_train,y_train_preds))
    r2_scores_train_max = r2_score(y_train, y_train_preds)
    
    #calculate the mse and r2 for the test dataset
    y_test_preds= best_rf_model.predict(X_test)
    mse_test_min = np.sqrt(mean_squared_error(y_test,y_test_preds))
    r2_scores_test_max = r2_score(y_test, y_test_preds)
       
    return r2_scores_test_max, r2_scores_train_max, best_rf_model, X_train, X_test, y_train, y_test