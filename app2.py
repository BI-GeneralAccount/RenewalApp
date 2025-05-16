import pandas as pd
import streamlit as st
import great_expectations as gx
import pickle
from lightgbm import LGBMClassifier
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np

from functions import *

TEST = 0;

# st.set_page_config(layout="wide")

def main():
    
    # Page Header area for logo & text
    app_intro_text()

    # file upload area
    file_upload = st.file_uploader("", type=["csv"])

    if file_upload is not None:

        # read in dirty data from data collection
        df_dirty_data = process_csv_to_df(file_upload);
    
        if(TEST):{
            st.dataframe(df_dirty_data.head())
        }

        df_new_unseen_data = clean_data(df_dirty_data)

        if(TEST):{
            st.dataframe(df_new_unseen_data.head())
        }

        # great expectation validation
        # create the gx suite
        suite, batch = create_gx_suite(df_new_unseen_data);

        # load the gx suite with the expectations
        load_gx_suite(suite);

        validation_results = batch.validate(expect=suite)

        # File upload -> Great Expectation failure
        # if csv file is NOT set up correctly, output error dataframe
        if(not validation_results.success):
            app_failure(validation_results);

        # else: process the model and get down to business
        else:
            df_model_output, df_model, model = run_model(df_new_unseen_data);

            if(TEST):{
                st.dataframe(df_model_output.head())
            }

            generate_charts(df_model_output)
            avg_liklihood_info(df_model_output)
            generate_shap(df_model_output, df_model, model)
            csv_download_buttons(df_model_output)         

    return 0;

main();
