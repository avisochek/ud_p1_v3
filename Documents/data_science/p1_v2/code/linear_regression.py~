import numpy as np
import pandas
from ggplot import *




def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma


def compute_cost(features, values, theta):
    pred = np.dot(features,theta) 
    error = np.subtract(pred, values)
    cost = (np.square(error)/(2*len(values))).sum()
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    m = len(values)
    cost_history = []
    for i in range(num_iterations):
        # your code here
        predict = np.dot(features,theta)
        error = np.subtract(predict,values)
        a = np.dot(error,alpha)
        grad  = np.dot(a/m,features)
        cost_history.append(compute_cost(features,values,theta))
        theta = np.subtract(theta,grad)
    
    print "theta: " , theta
    return theta, pandas.Series(cost_history)


def predictions(dataframe):
    # Select Features (try different features!)
    #print "dataframe",dataframe
    features = dataframe[['rain','weekday']]
    
    # Add UNIT to features using dummy variables
    unit_dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')

    # Add another set of dummy variables for time of day and station
    hour_dummy_units = pandas.get_dummies(dataframe['hour'],prefix='hour')
    #station_dummy_units = pandas.get_dummies(dataframe['station'],prefix='station')
    #conds_dummy_units = pandas.get_dummies(dataframe['conds'],prefix='conds')    

    features = features.join(unit_dummy_units)
    features = features.join(hour_dummy_units)
    #features = features.join(conds_dummy_units)
    #features = features.join(station_dummy_units)
    
    print features
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 75 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    #print len(features.columns)
    #print len(theta_gradient_descent)
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    ## Compute rsquared...
    rsquared = compute_r_squared(values_array,np.dot(features_array,theta_gradient_descent))


    plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    plot = plot_cost_history(alpha, cost_history)
    # 
    # Please note, there is a possibility that plotting
    # this in addition to your calculation will exceed 
    # the 30 second limit on the compute servers.
    
    predictions = np.dot(features_array, theta_gradient_descent)
    #print plot
    return predictions, plot,rsquared


def plot_cost_history(alpha, cost_history):
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def compute_r_squared(data, predictions):
    N = len(data)
    mu = np.mean(data)
    A = np.subtract(mu,data)
    B = np.square(A)
    SS_tot = np.sum(B)
    SS_res = np.sum(np.square(np.subtract(data,predictions)))
    r_squared = 1-(SS_res/(SS_tot))
    return r_squared

filepath = '../turnstile_weather_v2.csv'
df = pandas.read_csv(filepath)

predictions,plot,rsquared = predictions(df)
print "rsquared:", rsquared
print plot
