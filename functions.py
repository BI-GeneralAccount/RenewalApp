import pandas as pd
import streamlit as st
import great_expectations as gx
import pickle
from lightgbm import LGBMClassifier
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np

TEST = 0

def categorize_state(state):
    """
    Return the region a state belongs to.

    :param a: state name (str).
    :return: name of the region the state belongs to.
    """

    # Define categories
    territories = ['AS', 'FM', 'GU', 'MH', 'MP', 'PW', 'PR', 'VI']
    military = ['AP', 'AE']
    top_5_states = ['CA', 'NY', 'TX', 'IL', 'FL']
    northeast = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'PA']
    midwest = ['IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
    south = ['DE', 'GA', 'KY', 'MD', 'NC', 'SC', 'TN', 'VA', 'WV', 'AL', 'MS', 'AR', 'LA', 'OK', 'DC']
    west = ['AK', 'AZ', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
    
    clean_state = str(state).strip().upper()
    if state in top_5_states:
        return state
    elif state in northeast:
        return 'Northeast'
    elif state in midwest:
        return 'Midwest'
    elif state in south:
        return 'South'
    elif state in west:
        return 'West'
    elif state in territories:
        return 'Territories'
    elif state in military:
        return 'Overseas military'
    else:
        return 'International'

def categorize_practice(practice):
    """
    Categorizes practice type.

    :param a: practice.
    :return: the practice category.
    """

    practice_type = ['0', 'Solo Practitioner', 'Government', 'Small Firm (2-5 Attorneys)', 'Corporate',
            'Private Practice (6+ Attorneys)', 'Non-Profit','Public Interest/Legal Aid']
    
    clean_state = str(practice).strip().lower()
    if practice in practice_type:
        return practice
    else:
       return 'Other' 

def process_csv_to_df(file_upload):
    """
    Return a dataframe ready for app processing.

    :param a: csv file uploaded to the app.
    :return: dataframe ready for the model.
    """

    # convert csv to a dataframe
    df_new_unseen_data = pd.read_csv(file_upload)

    # drop weird 'Unnamed: 0' column if it exists
    if ('Unnamed: 0' in df_new_unseen_data.columns):
        print("yes, df contained Unnamed: 0")
        print("Unnamed: 0 is dropped")
        df_new_unseen_data.drop(columns=['Unnamed: 0'], inplace=True)

    return df_new_unseen_data;

def clean_data(df_raw_data):
    """
    Processes and cleans data from data collection phase and prepares it for
    model input.

    :param a: DataFrame of new, dirty data
    :return: DataFrame ready for the model.

    """
    # Convert 'customer_id' column to str, then pads values w/ leading zeros to 
    # ensure a length of 8 characters.
    df_raw_data['customer_id'] = df_raw_data['customer_id'].astype(str).str.zfill(8)

    # Drop duplicate customer_id, keeping only the last occurrence
    df_raw_data = df_raw_data.drop_duplicates(subset='customer_id', keep='last')

    # Create a new DataFrame 'new_joined_before_2017_df' by filtering rows 
    # from 'tenure_cal_df'.
    mask = df_raw_data['MOST_RECENT_ADD_DATE'] < '2017-01-01'
    new_joined_before_2017_df = pd.DataFrame(df_raw_data[mask])

    # Filter rows where the 'cycle_begin_date' column has a value of '0'
    mask = df_raw_data['cycle_begin_date'] == '0'
    one_renewal_date_df = df_raw_data[mask]

    # Combine the 'customer_id' columns from 'new_joined_before_2017_df' and 
    # 'one_renewal_date_df' into a single Series.
    exclude_ids = pd.concat([new_joined_before_2017_df['customer_id'], 
                             one_renewal_date_df['customer_id']]).drop_duplicates(keep=False)

    # Filter rows from 'renewal_raw_df' where the 'customer_id' is not in the 'exclude_ids' list.
    mask = ~df_raw_data['customer_id'].isin(exclude_ids)
    renewal_filtered_df = df_raw_data[mask] 

    # Create a new DataFrame by dropping unnecessary columns from 'renewal_filtered_df'.
    renewal_df = renewal_filtered_df.drop(['MOST_RECENT_ADD_DATE', 'CYCLE_BEGIN_DATE', 
                                           'CYCLE_END_DATE', 'GRACE_DATE', 
                                           'Group Member Dues','PAYMENT_STATUS', 
                                           'AS_OF_DATE', 'ABASET_SUBCODE_DESCR', 
                                           'cycle_begin_date', 'Member Dues',
                                           'cycle_end_date', 'product_code', 
                                           'order_no', 'order_line_no', 'grace_date', 
                                           'DOB','the_rank', 'member_renewal_indicator', 
                                           'earliest_begin_date', 'order_count'], axis=1)
    
    # Assign new columns for bundled data to the `renewal_df` dataframe by summing specific groups of columns
    renewal_df = renewal_df.assign(
        article_order=renewal_df[['Article Download', 'Journal', 'Magazine', 'Newsletter', 'Single Issue']].sum(axis=1),
        books_order=renewal_df[['Book', 'E-Book', 'Chapter Download']].sum(axis=1),
        contribution_order=renewal_df[['Contribution', 'Donation']].sum(axis=1),
        digital_education_order=renewal_df[['Webinar', 'On-Demand']].sum(axis=1),
        ecd_misc_order=renewal_df[['Course Materials Download']].sum(axis=1),
        events_misc_order=renewal_df[['Product', 'Exhibitor', 'Sponsorship Non-UBIT', 'Sponsorship UBIT']].sum(axis=1),
        inventory_misc_order=renewal_df[['Brochure', 'CD-ROM', 'Directory', 'Errata', 'Letter', 'Loose Leaf', 'Pamphlet', 'Standing Order']].sum(axis=1),
        meeting_order=renewal_df[['Meeting', 'Virtual Meeting', 'Invite Only Meeting', 'ABA Midyear', 'In-Person']].sum(axis=1),
        merchandise_order=renewal_df[['General Merchandise', 'Clothing']].sum(axis=1),
        misc_order=renewal_df[['Audio Download', 'Inventory Product Package']].sum(axis=1)
    ).drop(columns=[
        # Drop all the original columns that were summed into the new columns
        'Article Download', 'Journal', 'Magazine', 'Newsletter', 'Single Issue',
        'Book', 'E-Book', 'Chapter Download', 'Contribution', 'Donation',
        'Webinar', 'On-Demand', 'Course Materials Download','Product',
        'Exhibitor', 'Sponsorship Non-UBIT', 'Sponsorship UBIT',
        'Brochure', 'CD-ROM', 'Directory', 'Errata', 'Letter',
        'Loose Leaf', 'Pamphlet', 'Standing Order','Meeting',
        'Virtual Meeting', 'Invite Only Meeting','ABA Midyear',
        'In-Person', 'General Merchandise', 'Clothing',
        'Audio Download', 'Inventory Product Package'
    ])

    # Apply categorization
    renewal_df['STATE'] = renewal_df['STATE'].apply(categorize_state)

    # Apply categorization
    renewal_df['ABASET_CODE_DESCR'] = renewal_df['ABASET_CODE_DESCR'].apply(categorize_practice)

    # formatting
    renewal_df.columns = renewal_df.columns.str.lower()
    renewal_df.columns = renewal_df.columns.str.replace(' ', '_')
    renewal_df.columns = renewal_df.columns.str.replace('-', '_')

    # dropping the following columns as they are imbalanced as per the eda report
    drop_cols = ['events_cle', 'misc_order', 'disability_indicator', 
                 'ethnicity_code', 'auto_enroll_section_count', 'gender_code', 
                 'descr']
    renewal_df = renewal_df.drop(columns=drop_cols, axis=1)

    # List of categorical columns to frequency-encode
    categorical_cols = ['abaset_code_descr', 'state']

    # Apply frequency encoding using a loop
    for col in categorical_cols:
        frequency  = renewal_df[col].value_counts(normalize=True)
        renewal_df[col + '_encoded'] = renewal_df[col].map(frequency)

    columns_to_check= ['dues_required_section_count', 'no_charge_section_count', 
                       'member_groups', 'article', 'books','on_demand_video',
                       'news_aba', 'podcast', 'aba_advantage', 'article_order', 
                       'age', 'books_order', 'contribution_order', 'digital_education_order', 
                       'ecd_misc_order','events_misc_order', 'inventory_misc_order', 
                       'meeting_order', 'merchandise_order']
    
    skewness = renewal_df[columns_to_check].skew()
    ## < ±0.5: Fairly symmetrical (no need to transform)
    ## 0.5–1: Moderate skewness (may need transformation)
    ## > 1: Highly skewed (need transformation) log transformation
    columns_to_log_transform = skewness[skewness > 0.5].index.tolist()

    renewal_df_log = renewal_df.copy()
    renewal_df_log[columns_to_log_transform] = renewal_df_log[columns_to_log_transform].apply(np.log1p)

    renewal_df_log = renewal_df_log.drop(['abaset_code_descr', 'state', 
                                        'ecd_misc_order', 'events_misc_order', 
                                        'inventory_misc_order', 'merchandise_order', 
                                        'article_order', 'news_aba', 'contribution_order', 
                                        'podcast', 'books_order', 'on_demand_video'], 
                                        axis=1)
    
    # drop member_renewed_indicator column if it exists
    if ('member_renewed_indicator' in renewal_df_log.columns):
        renewal_df_log.drop(columns=['member_renewed_indicator'], inplace=True)

    # convert 'customer_id' to int
    renewal_df_log['customer_id'] = renewal_df_log['customer_id'].astype('int64')
    
    return renewal_df_log;

def create_gx_suite(dataframe):
    """
    Return a great expectation suite.

    :param a: pandas dataframe to be validated.
    :return: gx suite.
    """

    # Create Data Context
    context = gx.get_context()

    # Create pandas Data Source, Data Asset, and Batch Definition
    data_source = context.data_sources.add_pandas(
        name="pandas_datasource"
    )

    # create the data asset
    data_asset = data_source.add_dataframe_asset(
        name="renewal_asset"
    )

    # create the batch definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="my_batch_definition"
    )

    # pass your dataframe into a batch. A batch is a group of records that a
    # validation can be run on 
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": dataframe}
    )

    suite = gx.ExpectationSuite(
        name="renewal_suite"
    )

    return suite, batch;

def load_gx_suite(suite):
    """
    Loads up the Great Expectation suite with expectations.

    :param a: gx suite.
    :return: 0.
    """

    # column count == 12
    expectation = gx.expectations.ExpectTableColumnCountToEqual(
        value=12
    )
    suite.add_expectation(
        expectation=expectation
    )
    #__________________________________________________________________________

    # ensure all columns are named as we expect    
    column_list = ['customer_id', 'dues_required_section_count', 'no_charge_section_count',
       'member_groups', 'article', 'books', 'aba_advantage', 'age',
       'digital_education_order', 'meeting_order', 'abaset_code_descr_encoded',
       'state_encoded']

 
    expectation = gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=column_list,
        exact_match=True
    )

    suite.add_expectation(
        expectation=expectation
    )
    #__________________________________________________________________________

    # ensure all columns are of certain type

    # customer_id is the only int64 column
    expectation = gx.expectations.ExpectColumnValuesToBeOfType(
        column="customer_id",
        type_="int64"
    )

    suite.add_expectation(
        expectation=expectation
    )

    # the rest of the columns are of type float64
    cols = ['dues_required_section_count', 'no_charge_section_count',
       'member_groups', 'article', 'books', 'aba_advantage', 'age',
       'digital_education_order', 'meeting_order', 'abaset_code_descr_encoded',
       'state_encoded']

    for col in cols:

        expectation = gx.expectations.ExpectColumnValuesToBeOfType(
            column=col,
            type_="float64"
        )   

        suite.add_expectation(
            expectation=expectation
        )
    #__________________________________________________________________________
    
    # check that there are no missing values in any of the columns
    column_list = ['customer_id', 'dues_required_section_count', 'no_charge_section_count',
       'member_groups', 'article', 'books', 'aba_advantage', 'age',
       'digital_education_order', 'meeting_order', 'abaset_code_descr_encoded',
       'state_encoded']


    for col in column_list:
    
        expectation = gx.expectations.ExpectColumnValuesToNotBeNull(
            column=col,
        )

        suite.add_expectation(
            expectation=expectation
        )
    #__________________________________________________________________________

    # all customer_id are unique
    expectation = gx.expectations.ExpectColumnValuesToBeUnique(
        column="customer_id"
    )

    suite.add_expectation(
        expectation=expectation
    )
    #__________________________________________________________________________

    # columns that we expect to contain mostly 0
    cols = ['books', 'aba_advantage']

    for col in cols:
    
        expectation = gx.expectations.ExpectColumnMostCommonValueToBeInSet(
            column=col,
            value_set=[0],
            ties_okay=True
        )

        suite.add_expectation(
            expectation=expectation
        )
    #__________________________________________________________________________

    # no return necessary, returning 0 for cleanliness
    return 0;

def app_failure(validation_results):
    """
    Produces the app state when Great Expectation suite fails.

    :param a: gx validation results.
    :return: 0.
    """

    failed = []

    for result in validation_results.results:
        if(result["success"] == False):
            failed.append(result["expectation_config"]["type"])
        
    df = pd.DataFrame(failed, columns=['Failed Quality Tests'])
    df.index = range(1, len(df) + 1)
    st.write('##### The reason the app cannot process the file is because it did ' \
             'not pass data quality checks. Below is the list of failed tests. ' \
             'Please review the errors below and make appropriate changes.')
    st.dataframe(df)

    return 0;

def app_intro_text():
    """
    Produces the app intro text and logo.

    :return: 0.
    """

    st.image("logo.jpg")
    st.write("# Renewal Application")
    st.write("This web application will forecast the likelihood of a membership "
             "renewal three months prior to the membership expiration. The application "
             "makes these predictions based on member demographics and member benefit "
             "usage. The data is processed by a robust predictive machine learning "
             "model.")

    return 0;

def run_model(df_new_unseen_data):
    """
    Process new, unseen data through the model to produce results df.

    :param a: DataFrame of new, unseen data.
    :return: Dataframe with model results.
    """

    # st.write("Business as usual")

    # pull the model from the pickle file
    with open('lgbm_yl_v1_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # with open('lgbm_model.pkl', 'rb') as file:
    #     model = pickle.load(file)

    # extract customer_id column for later use
    customer_id  = df_new_unseen_data['customer_id']

    # df w/o customer_id column specifically for the model
    df_model     = df_new_unseen_data.drop(columns=['customer_id'])

    # predicted class output
    pred_class   = model.predict(df_model)

    # predicted probability output
    pred_proba   = model.predict_proba(df_model)

    # new df to output results
    df_model_output = pd.DataFrame({
        'customer_id': customer_id,
        'predicted_class': pred_class,
        'class_0_predicted_prob' : pred_proba[:, 0],
        'class_1_predicted_prob' : pred_proba[:, 1]
    })

    return df_model_output, df_model, model;

def generate_charts(df_model_output):
    """
    Process first row of visualizations: bar, pie, bar

    :param a: DataFrame of new, unseen data.
    :return: 0.
    """

    # create bins for class 0 label >=0.5
    df_model_output['class_0_bins'] = pd.cut(df_model_output['class_0_predicted_prob'], \
                               bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], \
                               right=True)
    
    # create bins for class 1 label >=0.5
    df_model_output['class_1_bins'] = pd.cut(df_model_output['class_1_predicted_prob'], \
                                bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], \
                                right=True)
    
    col1, col2, col3 = st.columns(3)

    # chart 1: bar chart class 0 probabilities
    with col1:
        counts_class_0 = df_model_output['class_0_bins'].value_counts().sort_index()
        fig1 = px.bar(
            x=[str(index) for index in counts_class_0.index],
            y=counts_class_0.values,
            labels={'x': 'Probability Range', 'y': 'Count'},
            title='Probabilities of Dropping',
            color_discrete_sequence=['#86ccfb']
        )
        st.plotly_chart(fig1, use_container_width=True)

    # chart 2: pie chart of predicted class
    with col2:
        label_map = {0: "Drop", 1: "Renew"}
        pie_counts = df_model_output['predicted_class'].value_counts().sort_index()
        fig2 = px.pie(
            values=pie_counts.values,
            names=pie_counts.index.map(label_map),
            title='Predicted Class Proportion',
            color_discrete_sequence=['#046ccc', '#86ccfb']
        )
        st.plotly_chart(fig2, use_container_width=True)

    # chart 3: bar chart of class 1 probabilities
    with col3:
        counts_class_1 = df_model_output['class_1_bins'].value_counts().sort_index()
        fig3 = px.bar(
            x=[str(b) for b in counts_class_1.index],
            y=counts_class_1.values,
            labels={'x': 'Probability Range', 'y': 'Count'},
            title='Probabilities of Renewing',
            color_discrete_sequence=['#046ccc']
        )
        st.plotly_chart(fig3, use_container_width=True)

    return 0;

def avg_liklihood_info(df_model_output):
    """
    Generate average liklihood of drop/renewal that are >= 0.5

    :param a: DataFrame of new, unseen data.
    :return: 0.
    """

    # three columns for next row of information
    col1, col2, col3 = st.columns(3)

    with col1:
        # get only the class_0_probs from df
        class_0_probs = df_model_output['class_0_predicted_prob']
        # find the mean of probabilities (>= 0.5)
        class_0_probs_avg = class_0_probs[class_0_probs >= 0.5].mean()

        st.write("**Average Likelihood of Dropping**")
        st.write(round(class_0_probs_avg, 2))

    with col2:
        st.write("") # hold blank space in column

    with col3:
        # get only the class_0_probs from df
        class_1_probs = df_model_output['class_1_predicted_prob']
        # find the mean of probabilities (>= 0.5)
        class_1_probs_avg = class_1_probs[class_1_probs >= 0.5].mean()

        st.write("**Average Likelihood of Renewing**")
        st.write(round(class_1_probs_avg, 2))

    return 0;

def generate_shap(df_model, model):
    """
    Generate SHAP chart visualization

    :param a: DataFrame ready to be fed to a model
    :param b: model from the pickle file
    :return: DataFrame of SHAP values for every customer_id.
    """
    # Load the explainer (use model you already trained)
    explainer = shap.TreeExplainer(model)

    # df_features is the data used for prediction
    shap_values = explainer.shap_values(df_model)

    shap_df = pd.DataFrame(shap_values, columns=df_model.columns)

    if(TEST):{
        st.dataframe(shap_df)
    }

    st.subheader("Feature Importance")

    # summary plot in bar chart format
    fig3, ax3 = plt.subplots()
    shap.summary_plot(shap_values, df_model, plot_type="bar")
    st.pyplot(fig3)

    return shap_df;

def convert_for_download(df):
    """
    Converts a dataframe into csv for user download
    :param a: DataFrame from the model output 
    :return: DataFrame split by Drop/Renewals
    """
    return df.to_csv().encode("utf-8")

def csv_download_buttons(df_model_output, shap_df):
    """
    Generate join/drop buttons for csv download capability

    :param a: DataFrame of model output.
    :param b: DataFrame of SHAP values for every row.
    :return: 0.
    """

    # drop cols needed for class 0/1 bins
    cleaned_df = df_model_output.drop(columns=['class_0_bins', 'class_1_bins'])

    # join df_model_output and shap_df side-by-side on index
    combined_df = pd.concat([cleaned_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    if(TEST):{
        st.dataframe(combined_df)
    }


    # generate class 0 output df (drops)
    mask = combined_df['predicted_class'] == 0
    # class_0_output = combined_df[mask][['customer_id', 'predicted_class', 
    #                                         'class_0_predicted_prob']]

    class_0_output = combined_df[mask].drop(columns=['class_1_predicted_prob'])

    if(TEST):{
        st.dataframe(class_0_output.head())
    }

    # generate class 1 output df (renewals)
    mask = combined_df['predicted_class'] == 1
    # class_1_output = combined_df[mask][['customer_id', 'predicted_class', 
    #                                         'class_1_predicted_prob']]

    class_1_output = combined_df[mask].drop(columns=['class_0_predicted_prob'])
    
    if(TEST):{
        st.dataframe(class_1_output.head())
    }
    
    # three columns for button arrangement
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download Drops Data",
            data=convert_for_download(class_0_output),
            file_name="likely_drops.csv",
            mime="text/csv"
        )

    with col2:
        st.write("") # hold blank space in column

    with col3:
        st.download_button(
            label="Download Renewals Data",
            data=convert_for_download(class_1_output),
            file_name="likely_renewals.csv",
            mime="text/csv"
        )

    return 0;
