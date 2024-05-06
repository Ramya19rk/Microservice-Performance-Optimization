import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import date

# Setting up page configuration

st.set_page_config(page_title= "Trace Data Analytics",
                   layout= "wide",
                   initial_sidebar_state= "expanded"                   
                  )

# Creating Background

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                    background:url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRh0U_PfLzl1ULRLbSk9nZogVva1EAvFXEzZQ&s");
                    background-size: cover}}
                </style>""", unsafe_allow_html=True)
setting_bg()

# Creating option menu in the side bar

with st.sidebar:

    selected = option_menu("Menu", ["Home","Prediction","EDA Analysis"], 
                           icons=["house","list-task","bar-chart-line"],
                           menu_icon= "menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                                   "nav-link-selected": {"background-color": "blue"}}
                          )
    
# Home Menu

if selected == 'Home':

    st.markdown(f'<h1 style="text-align: center; color: magenta;">TRACE DATA ANALYTICS</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Domain :*] Microservices")
        col1.markdown("# ")
        col1.markdown("## :violet[*Technologies used :*]")
        col1.markdown("##  Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit. ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Overview :*]")
        col1.markdown("##   Build Regression Model to Predict Duration of Span")
        col1.markdown("##   Engaging in Exploratory Data Analysis (EDA) on the dataset.")
        col1.markdown("# ")

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/trace1.jpg")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/trace2.jpeg")

# Prediction Menu

mapping_Name =  {
    0: '/',
    1: '/featureflags',
    2: '/featureflags/:id',
    3: '/featureflags/:id/edit',
    4: '/oteldemo.FeatureFlagService/GetFlag',
    5: '/oteldemo.ProductCatalogService/ListProducts',
    6: '/oteldemo.RecommendationService/ListRecommendations',
    7: 'CurrencyService/Convert',
    8: 'CurrencyService/GetSupportedCurrencies',
    9: 'EXPIRE',
    10: 'HGET',
    11: 'HMSET',
    12: 'HTTP GET',
    13: 'HTTP POST',
    14: 'POST /getquote',
    15: 'POST /send_order_confirmation',
    16: 'calculate-quote',
    17: 'charge',
    18: 'click',
    19: 'dns.lookup',
    20: 'documentFetch',
    21: 'documentLoad',
    22: 'featureflagservice.repo.query',
    23: 'featureflagservice.repo.query:featureflags',
    24: 'fs existsSync',
    25: 'fs open',
    26: 'fs readFileSync',
    27: 'fs readdirSync',
    28: 'fs realpathSync',
    29: 'fs stat',
    30: 'fs statSync',
    31: 'getAdsByCategory',
    32: 'getRandomAds',
    33: 'get_product_list',
    34: 'grpc.oteldemo.AdService/GetAds',
    35: 'grpc.oteldemo.CartService/AddItem',
    36: 'grpc.oteldemo.CartService/GetCart',
    37: 'grpc.oteldemo.CheckoutService/PlaceOrder',
    38: 'grpc.oteldemo.CurrencyService/GetSupportedCurrencies',
    39: 'grpc.oteldemo.PaymentService/Charge',
    40: 'grpc.oteldemo.ProductCatalogService/GetProduct',
    41: 'grpc.oteldemo.ProductCatalogService/ListProducts',
    42: 'grpc.oteldemo.RecommendationService/ListRecommendations',
    43: 'ingress',
    44: 'orders process',
    45: 'orders receive',
    46: 'orders send',
    47: 'oteldemo.AdService/GetAds',
    48: 'oteldemo.CartService/AddItem',
    49: 'oteldemo.CartService/EmptyCart',
    50: 'oteldemo.CartService/GetCart',
    51: 'oteldemo.CheckoutService/PlaceOrder',
    52: 'oteldemo.CurrencyService/Convert',
    53: 'oteldemo.FeatureFlagService/GetFlag',
    54: 'oteldemo.PaymentService/Charge',
    55: 'oteldemo.ProductCatalogService/GetProduct',
    56: 'oteldemo.ProductCatalogService/ListProducts',
    57: 'oteldemo.ShippingService/GetQuote',
    58: 'oteldemo.ShippingService/ShipOrder',
    59: 'prepareOrderItemsAndShippingQuoteFromCart',
    60: 'reqwest-http-client',
    61: 'resourceFetch',
    62: 'send_email',
    63: 'sinatra.render_template',
    64: 'tcp.connect',
    65: '{closure}'
}

mapping_serviceName = {0: 'accountingservice',
 1: 'adservice',
 2: 'cartservice',
 3: 'checkoutservice',
 4: 'currencyservice',
 5: 'emailservice',
 6: 'featureflagservice',
 7: 'frauddetectionservice',
 8: 'frontend',
 9: 'frontend-proxy',
 10: 'frontend-web',
 11: 'loadgenerator',
 12: 'paymentservice',
 13: 'productcatalogservice',
 14: 'quoteservice',
 15: 'recommendationservice',
 16: 'shippingservice'}



if selected == 'Prediction':

    st.title(':red[Prediction Page]')

    with st.form("form1"):

        # Date
        date_column = st.date_input(label='Date', min_value=date(2023, 11, 23),
                                          max_value=date(2023, 11, 23), value=date(2023, 11, 23))

        #Time
        time_hours = st.number_input('Hours', min_value=0, max_value=59 ,value=0 )

        time_minutes = st.number_input('Minutes', min_value=0, max_value=59 ,value=0 )

        time_seconds = st.number_input('Seconds', min_value=0, max_value=59 ,value=0 )

        st.write('Time: ', time_hours, ':', time_minutes, ':', time_seconds)

        # SpanID
        swapped_mapping = {}
        with open('mapping_spanID_swapped.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                swapped_mapping[value.strip()] = int(key.strip())

        # Create a dropdown list using the values as options
        selected_value = st.selectbox('Span ID', list(swapped_mapping.keys()))

        # Get the corresponding key from the selected value
        if selected_value in swapped_mapping:
            selected_key = swapped_mapping[selected_value]
            st.write(f"You selected: {selected_value} - {selected_key}")
        else:
            st.write("No key found for the selected value.")

        # TraceID
        swapped_mapping1 = {}
        with open('mapping_traceID_swapped.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                swapped_mapping1[value.strip()] = int(key.strip())

        # Create a dropdown list using the values as options
        selected_value1 = st.selectbox('Trace Id', list(swapped_mapping1.keys()))

        # Get the corresponding key from the selected value
        if selected_value1 in swapped_mapping1:
            selected_key1 = swapped_mapping1[selected_value1]
            st.write(f"You selected: {selected_value1} - {selected_key1}")
        else:
            st.write("No key found for the selected value.")

        # ParentSpanID
        swapped_mapping2 = {}
        with open('mapping_parentspanID_swapped.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                swapped_mapping2[value.strip()] = int(key.strip())

        # Create a dropdown list using the values as options
        selected_value2 = st.selectbox('Parent Span ID', list(swapped_mapping2.keys()))

        # Get the corresponding key from the selected value
        if selected_value2 in swapped_mapping2:
            selected_key2 = swapped_mapping2[selected_value2]
            st.write(f"You selected: {selected_value2} - {selected_key2}")
        else:
            st.write("No key found for the selected value.")    

        # Name
        mapping_Name = st.selectbox("Name", options=mapping_Name.keys(), format_func=lambda x: mapping_Name[x])

        # Service Name
        mapping_serviceName = st.selectbox("Service Name", options=mapping_serviceName.keys(), format_func=lambda x: mapping_serviceName[x])


        st.markdown("# ")
        st.markdown("# ")
        submit_button = st.form_submit_button("Submit")

        if submit_button is not None:
                    with open('TraceData.pkl', 'rb') as f:
                        pick_model = pickle.load(f)


                        new_sample = np.array(
                                            [[date_column.day, date_column.month, date_column.year, time_hours, time_minutes, time_seconds, selected_key, selected_key1, selected_key2, mapping_Name, mapping_serviceName]])                       
                        new_pred = pick_model.predict(new_sample)
                        st.markdown(
                        f"<h1 style='font-size: 40px;'><span style='color: orange;'>Predicted DurationNano Second : </span><span style='color: green;'> {np.exp(new_pred[0])}</span> </h1>",
                        unsafe_allow_html=True)

# EDA Analysis Menu

if selected == 'EDA Analysis':

    st.title(":red[EDA Analysis Page]")

    # Load the data into a DataFrame
    df = pd.read_csv('EDA_TraceData1.csv')

    # Define the list of analysis options
    analysis_options = [
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis",
        "Scatter plot of DurationNano vs. Service Name",
        "Correlation Heatmap",
        "Count plot of Service Names",
        "Temporal and Distribution Analysis",
        "Missing Values Analysis",
    ]

    # Display selectbox for analysis options
    selected_analysis = st.selectbox("Select Analysis", analysis_options)

# Univariate Analysis

    # Perform analysis based on user selection
    if selected_analysis == "Univariate Analysis":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">1. Univariate Analysis</h3>', unsafe_allow_html=True)

        st.markdown("# ")

        st.subheader(':violet[Frequency of ServiceName]')

        # Calculate frequency of serviceName
        service_counts = df['serviceName'].value_counts()

        # Define custom colors for the bar chart
        colors = px.colors.qualitative.Set3

        # Create Plotly bar chart with custom colors
        fig = px.bar(x=service_counts.index, y=service_counts.values, labels={'x': 'Service Name', 'y': 'Frequency'}, 
                    title='Frequency of ServiceName', color=service_counts.index, color_discrete_sequence=colors)

        # Rotate x-axis labels for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Function to plot histogram with overlaid values
        def plot_histogram_with_overlay(data, bins, title, xlabel, ylabel):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data, bins=bins, kde=True, ax=ax)
            
            # Overlay values on histogram bars
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            st.pyplot(fig)

        st.markdown("# ")

        # Plot histogram with overlaid values
        st.subheader(':violet[DurationNano Distribution]')

        plot_histogram_with_overlay(df['durationNano'], bins=20, title='Distribution of DurationNano', xlabel='Duration (nanoseconds)', ylabel='Frequency')

        st.markdown("# ")
        # Display summary statistics of durationNano
        st.subheader(":violet[Summary Statistics of DurationNano:]")
        st.write(df['durationNano'].describe())

# Bivariate Analysis

    elif selected_analysis == "Bivariate Analysis":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">2. Bivariate Analysis</h3>', unsafe_allow_html=True)

        st.markdown("# ")

        st.subheader(':violet[Box plot of DurationNano across Service Names]')

        # Create Plotly box plot
        fig = px.box(df, x='serviceName', y='durationNano', title='Distribution of DurationNano across Service Names')

        # Customize x-axis labels rotation for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the box plot in Streamlit
        st.plotly_chart(fig)

        st.markdown("# ")

        # Create Plotly scatter plot     
        st.subheader(':violet[Relationship between DurationNano and Timestamp]')

        # Create Plotly scatter plot
        fig = px.scatter(df, x='Timestamp', y='durationNano', title='Relationship between DurationNano and Timestamp')

        # Display the scatter plot in Streamlit
        st.plotly_chart(fig)

        st.markdown("# ")

        st.subheader(':violet[Trend of DurationNano over Time]')

        # Create Plotly line chart
        fig = px.line(df, x='Timestamp', y='durationNano', title='Trend of DurationNano over Time')

        # Display the line chart in Streamlit
        st.plotly_chart(fig)

# Multivariate Analysis

    elif selected_analysis == "Multivariate Analysis":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">3. Multivariate Analysis</h3>', unsafe_allow_html=True)

        st.markdown("# ")        

        st.subheader(':violet[Relationship between DurationNano, ServiceName, and Name]')
        st.markdown("# ")

        # Create a grouped bar chart using Plotly
        fig = px.bar(df, x='serviceName', y='durationNano', color='Name', 
                    title='Relationship between DurationNano, ServiceName, and Name', 
                    barmode='group')

        # Customize x-axis labels rotation for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

# Scatter plot of DurationNano vs. Service Name

    elif selected_analysis == "Scatter plot of DurationNano vs. Service Name":
        
        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">4. Scatter plot of DurationNano vs. Service Name</h3>', unsafe_allow_html=True)

        st.markdown("# ")
        # Create Plotly scatter plot
        fig = px.scatter(df, x='serviceName', y='durationNano', title='DurationNano vs. Service Name')

        # Customize x-axis labels rotation for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the scatter plot in Streamlit
        st.plotly_chart(fig)

# Correlation Heatmap

    elif selected_analysis == "Correlation Heatmap":
        
        df1 = pd.read_csv('Corr_TraceData.csv')

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">5. Correlation Heatmap</h3>', unsafe_allow_html=True)

        st.markdown("# ")
        fig = px.imshow(df1.corr(), color_continuous_scale='Viridis')  # Using 'Viridis' as a valid colorscale
        st.plotly_chart(fig)

# Count plot of Service Names

    elif selected_analysis == "Count plot of Service Names":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">6. Count plot of Service Names</h3>', unsafe_allow_html=True)

        st.markdown("# ")

        # Calculate counts of serviceName
        service_counts = df['serviceName'].value_counts()

        # Define custom colors for the bar chart
        colors = px.colors.qualitative.Set3

        # Create Plotly bar chart with custom colors
        fig = px.bar(x=service_counts.index, y=service_counts.values, labels={'x': 'Service Name', 'y': 'Count'}, 
                    title='Count of Service Names', color=service_counts.index, color_discrete_map={name: color for name, color in zip(service_counts.index, colors)})

        # Rotate x-axis labels for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the bar chart in Streamlit
        st.plotly_chart(fig)

# Temporal and Distribution Analysis

    elif selected_analysis == "Temporal and Distribution Analysis":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">7. Temporal and Distribution Analysis</h3>', unsafe_allow_html=True)

        st.markdown("# ")

        st.subheader(':violet[Trend of DurationNano over Time]')

        # Create Plotly line chart
        fig = px.line(df, x='Timestamp', y='durationNano', title='Trend of DurationNano over Time')

        # Display the line chart in Streamlit
        st.plotly_chart(fig)

        st.markdown("# ")

        st.subheader(':violet[Distribution of DurationNano within ServiceName Categories]')

        # Create Plotly violin plot
        fig = px.violin(df, x='serviceName', y='durationNano', title='Distribution of DurationNano within ServiceName Categories')

        # Customize x-axis labels rotation for better readability
        fig.update_layout(xaxis={'tickangle': 45})

        # Display the violin plot in Streamlit
        st.plotly_chart(fig)

# Missing Values Analysis

    elif selected_analysis == "Missing Values Analysis":

        st.markdown("# ")

        st.markdown('<h3 style="color: maroon;">8. Missing Values Analysis</h3>', unsafe_allow_html=True)

        st.markdown("# ")
        st.subheader(':violet[Missing Values:]')

        st.write(df.isnull().sum())

