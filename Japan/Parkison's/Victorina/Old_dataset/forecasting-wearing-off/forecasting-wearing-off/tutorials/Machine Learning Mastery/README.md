## Installation

[Interactive Matplots](https://towardsdatascience.com/how-to-produce-interactive-matplotlib-plots-in-jupyter-environment-1e4329d71651)

    conda install -c conda-forge ipympl
    conda install -c conda-forge nodejs
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter lab build

## JupyterLab User Setting

    {
        "codeCellConfig": {
            "codeFolding": true,
            "rulers": [80, 100]
        }
    }
    
## References

### Useful Code

#### [Pandas Filter by Column Value](https://sparkbyexamples.com/pandas/pandas-filter-by-column-value/)

    df.query("`heart_rate` >= 50")
    
### [Pandas Select All Columns except One Column](https://sparkbyexamples.com/pandas/pandas-select-all-columns-except-one-column-in-dataframe/)

    df.drop("Unwanted column", axis=1)
    df.loc[:, df.columns != "Unwanted column"]

### Main References
* [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
* [LSTMs for Human Activity Recognition Time Series Classification - With Accelerometer](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)
* [How to Develop LSTM Models for Time Series Forecasting - More Examples](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)
* [Evaluate the Performance Of Deep Learning Models in Keras - Cross-Validate Deep Learning](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)
* [Tensorflow & Keras Preprocessing Layer from DataFrame](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#build_the_preprocessing_head)
* [Tensorflow & Keras Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
* [Interpretting Learning Curves: Underfit, Overfit, Good Fit](https://towardsdatascience.com/learning-curve-to-identify-overfitting-underfitting-problems-133177f38df5)
* [Shading Areas in PyPlot](https://stackoverflow.com/a/43234308/2303766)
* [Keras LSTM Input Shape Explainer 1](https://medium.com/@raman.shinde15/understanding-sequential-timeseries-data-for-lstm-4da78021ecd7)
* [Keras LSTM Input Shape Explainer 2](https://towardsdatascience.com/how-to-convert-pandas-dataframe-to-keras-rnn-and-back-to-pandas-for-multivariate-regression-dcc34c991df9)
* [Convert Keras Prediction Percentage to Binary](https://stackoverflow.com/a/48644320/2303766)


### Others
* [Time Series Classification Tutorial: Combining Static and Sequential Feature Modeling using Recurrent Neural Networks - Predict Cardiac Arrest](https://omdena.com/blog/time-series-classification-model-tutorial/)
* [Multivariate Multi-step Time Series Forecasting using Stacked LSTM sequence to sequence Autoencoder in Tensorflow 2.0 / Keras](https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/)
* [Fraud Detection using LSTM in Keras](https://github.com/abulbasar/neural-networks/blob/master/Keras%20-%20Multivariate%20time%20series%20classification%20using%20LSTM.ipynb)
