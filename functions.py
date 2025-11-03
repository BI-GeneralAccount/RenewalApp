import pandas as pd
import streamlit as st
import great_expectations as gx
import pickle
from lightgbm import LGBMClassifier
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

TEST = 0
model_file_name = '250930_hierarchy_lgb_model.pkl'

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

def process_csv_to_df(file_upload):
    """
    Return a dataframe ready for app processing.

    :param a: csv file uploaded to the app.
    :return: DataFrame ready for the model.
    """

    # convert csv to a dataframe
    df_new_unseen_data = pd.read_csv(file_upload)

    # drop weird 'Unnamed: 0' column if it exists
    if ('Unnamed: 0' in df_new_unseen_data.columns):
        df_new_unseen_data.drop(columns=['Unnamed: 0'], inplace=True).reset_index()

    return df_new_unseen_data;

def categorize_state(state):
    """
    Return the region a state belongs to.

    :param a: state name (str).
    :return: name of the region the state belongs to.
    """

    # Define categories
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
    else:
        return 'other';
    
def categorize_practice(practice):
    """
    Categorizes practice type.

    :param a: practice.
    :return: the practice category/type.
    """

    practice_type = ['0', 'Corporate', 'Private Practice (6+ Attorneys)']
    no_profit = ['Academic', 'Non-Profit','Public Interest/Legal Aid']
    small_firms = ['Small Firm (2-5 Attorneys)', 'Solo Practitioner', 'Inactive/Retired']

    if practice in practice_type:
        return practice
    elif practice in no_profit:
        return 'non_profit'
    elif practice in small_firms:
        return 'small_firms'
    else:
       return 'government';

def categorize_payment_method(row):
    """
    Function to categorize payment methods.

    :param a: a row from a DataFrame of new, dirty data
    :return: a string (the method that was used)

    """
    method = row['receipt_type_descr']
    
    if isinstance(method, str) and method.strip().lower().startswith('web'):
        return 'Automated'
    else:
        return 'Manual';

def categorize_gender(gender):
    """
    Categorizes the gender of a row item based on gender code.

    :param a: a gender
    :return: a string, the gender assigned to that row
    """

    male = ['MALE']
    female = ['FEMALE']

    if gender in male:
        return 'male'
    elif gender in female:
        return 'female'
    else:
        return 'other';

def categorize_ethnicity(ethincity_code):
    """
    Categorizes the ethnicity of a row item.

    :param a: an ethnicity code
    :return: a string, the ethnicity assigned to that row
    """

    ethnicity = ['HISPANIC', 'BLK/AFAM']
    white = ['WHITECAUC', 'MULTI', 'DECLINE']
    native_american = ['NATIVAMER', 'HAW/PACISL']
    asian = ['ASIAN', 'MIDEAST']

    #clean_code = str(code).strip().upper()
    if ethincity_code in ethnicity:
        return ethincity_code
    elif ethincity_code in white:
        return 'white'
    elif ethincity_code in native_american:
        return 'native_american'
    elif ethincity_code in asian:
        return 'asian'
    else:
        return 'others';

def categorize_sexuality(sex_orientation):
    """
    Categorizes the sexual orientation of a row item.

    :param a: a sexual orientation item
    :return: a string, the sexual orientation assigned to that row
    """
    
    no_response = ['0']
    hetro = ['Heterosexual']

    if sex_orientation in no_response:
        return 'no_response'
    elif sex_orientation in hetro:
        return 'Heterosexual'
    else:
        return 'other';

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

    df_raw_data.columns = df_raw_data.columns.str.lower()

    # Drop rows where activity is 'Other'
    df_raw_data = df_raw_data[df_raw_data['activity'] != 'Other']
    df_raw_data.reset_index(drop=True, inplace=True)
    # Create the member_renewed_indicator column
    df_raw_data['member_renewed_indicator'] = df_raw_data['activity'].apply(lambda x: 1 if x.lower() == 'renewed' else 0)

    # Drop duplicate customer_id, keeping only the last occurrence
    renewal_df = df_raw_data.drop_duplicates(subset='customer_id', keep='last')

    # Apply the function row-wise
    renewal_df['payment_type'] = renewal_df.apply(categorize_payment_method, axis=1)

    # create hierarchy for the date
    payment_hierarchy = {
        'Manual' : 2,
        'Automated' : 1
    }
    renewal_df['payment_type_h'] = renewal_df['payment_type'].map(payment_hierarchy)
    
    # Apply categorization
    renewal_df['state'] = renewal_df['state'].apply(categorize_state)

    # create a state hierarchy
    state_hierarchy = {
        'South': 10,
        'Midwest': 9,
        'West': 8,
        'TX': 7,
        'Northeast': 6,
        'IL': 5,
        'CA': 4,
        'FL': 3,
        'other': 2,
        'NY': 1
    }

    renewal_df['state_h'] = renewal_df['state'].map(state_hierarchy)

    # Apply categorization
    renewal_df['practice_type'] = renewal_df['abaset_code_descr'].apply(categorize_practice)

    # create hierarchy/order for practice type
    practice_order = {
    '0' : 6,
    'Private Practice (6+ Attorneys)' : 5,
    'small_firms' : 4,
    'Corporate' : 3,
    'government' : 2,
    'non_profit' : 1,
    }

    renewal_df['practice_type_h'] = renewal_df['practice_type'].map(practice_order)

    # Apply categorization
    renewal_df['gender']= renewal_df['gender_code'].apply(categorize_gender)

    # create hierarchy/order for gender
    gender_order = {
    'male' : 3,
    'female' : 2,
    'other' : 1,
    }

    renewal_df['gender_h'] = renewal_df['gender'].map(gender_order)

    # Apply categorization
    renewal_df['ethnicity'] = renewal_df['ethincity_code'].apply(categorize_ethnicity)

    # create hierarchy/order for ethnicity
    ethnicity_order = {
    'white' : 6,
    'BLK/AFAM' : 5,
    'asian' : 4,
    'native_american' : 3,
    'HISPANIC' : 2,
    'others' : 1}

    renewal_df['ethnicity_h'] = renewal_df['ethnicity'].map(ethnicity_order)

    # Apply categorization
    renewal_df['sexual_orientation_group'] = renewal_df['sexual_orientation'].apply(categorize_sexuality)

    # create hierarchy/order for sexual orientation
    s_orient_order = {
    'Heterosexual' : 3,
    'other' : 2,
    'no_response' : 1
    }

    renewal_df['s_orientation_h'] = renewal_df['sexual_orientation_group'].map(s_orient_order)

    # create hierarchy/order for disability
    disability_order = {
    'Y' : 2,
    'N' : 1
    }

    renewal_df['disability_h'] = renewal_df['disability_indicator'].map(disability_order)

    # Assign new columns for bundled data to the `renewal_df` dataframe by summing specific groups of columns
    renewal_df = renewal_df.assign(
        article_order=renewal_df[['article_download', 'journal', 'magazine', 'newsletter', 'single_issue']].sum(axis=1),
        books_order=renewal_df[['book_sales', 'e_book', 'audio_download']].sum(axis=1),
        contribution_order=renewal_df[['contribution', 'donation']].sum(axis=1),
        digital_education_order=renewal_df[['webinar', 'on_demand', 'chapter_download', 'course_materials_download', 'on_demand_video']].sum(axis=1),
        events_misc_order=renewal_df[['product', 'exhibitor', 'sponsorship_non_ubit', 'sponsorship_ubit']].sum(axis=1),
        inventory_misc_order=renewal_df[['brochure', 'cd_rom', 'directory', 'errata', 'letter', 'loose_leaf', 'pamphlet', 'standing_order', 'inventory_product_package']].sum(axis=1),
        meeting_order=renewal_df[['meeting', 'virtual_meeting', 'invite_only_meeting', 'in_person']].sum(axis=1),
        merchandise_order=renewal_df[['general_merchandise', 'clothing']].sum(axis=1),
    ).drop(columns=[
        'article_download', 'journal', 'magazine', 'newsletter', 'single_issue',
        'book_sales', 'e_book', 'audio_download', 'contribution', 'donation',
        'webinar', 'on_demand', 'chapter_download', 'course_materials_download',
        'product', 'exhibitor', 'sponsorship_non_ubit', 'sponsorship_ubit',
        'brochure', 'cd_rom', 'directory', 'errata', 'letter', 'loose_leaf',
        'pamphlet', 'standing_order', 'inventory_product_package', 'meeting',
        'virtual_meeting', 'invite_only_meeting', 'in_person',
        'general_merchandise', 'clothing', 'on_demand_video'])

    # dropping the following columns as the ordinal hierarchy is already created
    renewal_df = renewal_df.drop(['gender_code', 'ethincity_code', 'abaset_code_descr', 'receipt_type_descr',
                                'sexual_orientation', 'member_groups_orders', 'activity'], axis=1)
    
    renewal_df.columns = renewal_df.columns.str.lower()
    renewal_df.columns = renewal_df.columns.str.replace(' ', '_')
    renewal_df.columns = renewal_df.columns.str.replace('-', '_')

    columns_not_used_in_the_model = ['state', 'payment_type', 'practice_type', 'gender','ethnicity', 'sexual_orientation_group', 'disability_indicator',
                                     'dues_required_section_count','no_charge_section_count', 'auto_enroll_section_count', 'disability_h',
                                     'events_cle', 'contribution_order', 'events_misc_order', 'inventory_misc_order', 'merchandise_order','article_order',
                                     'group_member_dues']

    renewal_df = renewal_df.drop(columns_not_used_in_the_model, axis=1) 

    # drop member_renewed_indicator column if it exists
    if ('member_renewed_indicator' in renewal_df.columns):
        renewal_df.drop(columns=['member_renewed_indicator'], inplace=True)

    # convert 'customer_id' to int
    renewal_df['customer_id'] = renewal_df['customer_id'].astype('int64')
    renewal_df = renewal_df.reset_index(drop=True)
    
    return renewal_df;

def create_gx_suite(dataframe):
    """
    Return a great expectation suite.

    :param a: pandas DataFrame to be validated.
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

    # column count = 19
    expectation = gx.expectations.ExpectTableColumnCountToEqual(
        value=19
    )
    suite.add_expectation(
        expectation=expectation
    )
    #__________________________________________________________________________

    # ensure all columns are named as we expect        
    column_list = ['customer_id', 'inferred_age', 'aoi_count',
                   'member_dues', 'article', 'books', 'news_aba', 'podcast',
                   'aba_advantage', 'member_groups', 'payment_type_h', 'state_h',
                   'practice_type_h', 'gender_h', 'ethnicity_h', 's_orientation_h',
                   'books_order', 'digital_education_order', 'meeting_order']

 
    expectation = gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=column_list,
        exact_match=True
    )

    suite.add_expectation(
        expectation=expectation
    )
    #__________________________________________________________________________

    # ensure all columns are of certain type

    # the rest of the columns are of type int64
    cols = ['customer_id', 'aoi_count', 'member_dues', 
                   'member_groups', 'payment_type_h', 'state_h', 'practice_type_h', 
                   'gender_h', 'ethnicity_h', 's_orientation_h', 'books_order', 
                   'meeting_order']

    for col in cols:

        expectation = gx.expectations.ExpectColumnValuesToBeOfType(
            column=col,
            type_="int64"
        )   

        suite.add_expectation(
            expectation=expectation
        )

    # the rest of the columns are of type float64
    cols = ['inferred_age', 'article', 'books', 'news_aba', 'podcast', 
            'aba_advantage', 'digital_education_order']
    
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
    column_list = ['customer_id', 'inferred_age', 'aoi_count',
                   'member_dues', 'article', 'books', 'news_aba', 'podcast',
                   'aba_advantage', 'member_groups', 'payment_type_h', 'state_h',
                   'practice_type_h', 'gender_h', 'ethnicity_h', 's_orientation_h',
                   'books_order', 'digital_education_order', 'meeting_order']

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
    cols = ['books', 'news_aba', 'podcast', 'aba_advantage',
            'books_order', 'meeting_order']

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

    # loop through results and create list of failures (if any)
    for result in validation_results.results:
        if(result["success"] == False):
            failed.append(result["expectation_config"]["type"])

    # create a df for output to the user  
    df = pd.DataFrame(failed, columns=['Failed Quality Tests'])

    # changing index column to start from 1 instead of 0
    df.index = range(1, len(df) + 1)

    st.write('##### The reason the app cannot process the file is because it did ' \
             'not pass data quality checks. Below is the list of failed tests. ' \
             'Please review the errors below and make appropriate changes.')
    st.dataframe(df)

    return 0;

def run_model(df_new_unseen_data):
    """
    Process new, unseen data through the model to produce results df.

    :param a: DataFrame of new, unseen data.
    :return: Dataframe with model results.
    """

    # pull the model from the pickle file
    with open(model_file_name, 'rb') as file:
        model = pickle.load(file)

    # extract customer_id column for later use
    customer_id  = df_new_unseen_data['customer_id']

    # df w/o customer_id column specifically for the model
    df_model     = df_new_unseen_data.drop(columns=['customer_id'])

    # predicted class output from running data through the model
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

    # create bins for class 0 predictions >=0.5
    df_model_output['class_0_bins'] = pd.cut(df_model_output['class_0_predicted_prob'], \
                               bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], \
                               right=True)
    
    # create bins for class 1 predictions >=0.5
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

def avg_likelihood_info(df_model_output):
    """
    Generate average likelihood of both drop/renewal that are >= 0.5

    :param a: DataFrame of new, unseen data.
    :return: Series: Class 0 Prob Avg, Class 1 Prob. Avg.
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

    return class_0_probs_avg, class_1_probs_avg;

def generate_class_average(combined_df, class_int, class_probs_avg):
    """
    Generate barchart for class_0 and class_1, showing features of realized and
    unrealized value, for the customer_ids with probabilities higher than the mean

    :param a: DataFrame of Class 0 and Class 1 data
    :param b: The class int identifier (0 or 1)
    :param c: The Average Likelihood of renewing for particular class
    :return: pandas series of shap averages for a specific class (0 or 1)
    """

    # sanity check to ensure class_int is 0/1
    if(class_int not in [0, 1]):
        return "invalid class_int input"

    # drop these columns, they're not needed for the shap visualization 
    drop_cols = ['customer_id', 'predicted_class', 'class_0_predicted_prob', 'class_1_predicted_prob']

    ###### Class 0 Section
    mask = combined_df['predicted_class'] == class_int
    class_shap = combined_df[mask]

    # build string (based on class_int input) to grab a column of df
    class_predicted_prob = 'class_' + str(class_int) + '_predicted_prob'

    if(TEST):
        st.write("Class Int:", class_int)
        st.write("Class", class_int, " Avg Prob: ", class_probs_avg)

    # filter df for rows ony >= average
    mask = class_shap[class_predicted_prob] >= class_probs_avg
    class_shap = class_shap[mask]
    
    class_shap.drop(columns=drop_cols, inplace=True)

    # get the averages and drop all positive (leaving only negative to plot)
    class_avg  = class_shap.mean()

    if(class_int == 0):
        mask = class_avg <= 0
        ascend = False
    else:
        mask = class_avg >= 0
        ascend = True

    averages  = class_avg[mask].sort_values(ascending=ascend)

    return averages

def generate_class_dfs(combined_df):
    """
    Generate dataframes that contain class_0 & class_1 information, separately

    :param a: DataFrame of combined info from both class, all info/data
    :return dfs: DataFrame for class_0 & DataFrame for class_1 
    """

    if(TEST):{
        st.dataframe(combined_df)
    }

    # generate class 0 output df (drops)
    mask = combined_df['predicted_class'] == 0
    class_0_output = combined_df[mask].drop(columns=['class_1_predicted_prob'])

    if(TEST):{
        st.dataframe(class_0_output.head())
    }

    # generate class 1 output df (renewals)
    mask = combined_df['predicted_class'] == 1
    class_1_output = combined_df[mask].drop(columns=['class_0_predicted_prob'])
    
    if(TEST):{
        st.dataframe(class_1_output.head())
    }

    return class_0_output, class_1_output

def generate_shap(df_model_output, df_model, model, class_0_probs_avg, class_1_probs_avg):
    """
    Generate barchart for class_0 and class_1, showing features of realized and
    unrealized value, for the customer_ids with probabilities higher than the mean

    :param a: DataFrame from model output
    :param b: DataFrame that was fed into the model
    :param c: The model from the pickle file
    :param d: Series of Class 0 probability averages
    :param e: Series of Class 1 probability averages
    :return: DataFrames for later csv output: Class 0/1 dfs
    """

    # load the explainer (use model already trained from pickle file)
    explainer = shap.TreeExplainer(model)

    # generate shap values from the df created to feed into the model
    shap_values = explainer.shap_values(df_model)

    # create df of the shap values and rename their column names
    shap_df = pd.DataFrame(shap_values, columns=df_model.columns)

    # drop cols needed for class 0/1 bins from df_model_output
    cleaned_model_output = df_model_output.drop(columns=['class_0_bins', 'class_1_bins'])

    # join df_model_output and shap_df side-by-side on index
    combined_df = pd.concat([cleaned_model_output.reset_index(drop=True), 
                             shap_df.reset_index(drop=True)], axis=1)
    
    # drop demographic & non-product columns

    non_product_cols = ["inferred_age", "aoi_count", "payment_type_h", "state_h",
                        "practice_type_h", "gender_h", "ethnicity_h",
                        "s_orientation_h", "member_dues"]
    
    combined_df = combined_df.drop(columns=non_product_cols)

    # dfs containing only Class 0 & Class 1 info, respectively
    class_0_output, class_1_output = generate_class_dfs(combined_df)

    # Generate series of negative/positive averages for visualization
    negative_averages = generate_class_average(combined_df, 0, class_0_probs_avg)
    positive_averages = generate_class_average(combined_df, 1, class_1_probs_avg)

    ###########################################################################
    #
    #
    #
    #
    #       Filter out the averages pertaining to demographics
    #
    #
    #
    ###########################################################################

    st.write("## Feature Importance")

    # three columns for button arrangement
    col1, col2 = st.columns(2)

    with col1: # barchart for Unrealized Values (Drops)
        fig, ax = plt.subplots()
        negative_averages.plot(kind='barh', 
                               color='lightcoral', 
                               title='Products With Unrealized Value')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

    with col2: # barchart for Realized Value (Renewals)
        fig, ax = plt.subplots()
        positive_averages.plot(kind='barh', 
                               color='#046ccc', 
                               title='Products Leading to Renewal')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

    return class_0_output, class_1_output

def convert_for_download(df):
    """
    Converts a dataframe into csv for user download

    :param a: DataFrame from the model output concat with SHAP values
    :return: DataFrame split by Drop/Renewals
    """
    return df.to_csv().encode("utf-8")

def csv_download_buttons(class_0_output, class_1_output):
    """
    Generate join/drop buttons with csv download capability

    :param a: All of Class 0 data/info
    :param b: All of Class 1 data/info
    :return: 0
    """

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

def single_customer_shap(df_model, model, index_int):
    """
    Create SHAP waterfall chart for a single customer_id

    :param a: DataFrame that was fed into the model
    :param b: The model from the pickle file
    :param c: The index of the DataFrame for a specific customer_id to analyze
    :return: 0
    """

    # load the explainer (use model already trained from pickle file)
    explainer = shap.TreeExplainer(model)

    # pick a sample row (e.g., first row)
    sample_index = index_int

    # get the single SHAP values
    shap_value_single = explainer(df_model.iloc[[sample_index]])

    # waterfall plot
    st.subheader(f"SHAP Waterfall Plot for Row {sample_index}")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_value_single[0], show=False)
    st.pyplot(fig)

    return 0;
   