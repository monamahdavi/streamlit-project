
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px


st.title('World Happiness Dashboard')
st.write('Explore Insights from the World Happiness Report Dataset with Interactive Charts.')

# Load the data
df = pd.read_csv('/Users/mona/Desktop/Northeastern/1st Semester/6600/Project/Project-02/2019.csv')

# Sidebar filters
st.sidebar.header('Interactive Elements')

score_min, score_max = st.sidebar.slider(
    'Select Score Range', 
    float(df["Score"].min()), 
    float(df["Score"].max()), 
    (6.0, 8.0))

# Filter the data based on score (score of happiness)
filtered_df = df[(df['Score'] >= score_min) & (df['Score'] <= score_max)]

#Dropdown for country selection
selected_country = st.sidebar.selectbox('Select a Country', df['Country or region'].unique())

#Checkbox for showing top countries by GDP per capita
show_top_gdp = st.sidebar.checkbox('Show Top 10 Countries by GDP per capita')

country_data = df[df['Country or region'] == selected_country]


# 1. Bar Chart: Score by Country 

fig, ax = plt.subplots(figsize=(12, 15))  
sns.barplot(y='Country or region', x='Score', data=filtered_df, ax=ax, color= 'purple')
ax.set_title('Score by Country')
ax.set_xlabel('Score')
ax.set_ylabel('Country')

st.pyplot(fig)

#filtered 10 country based GDP per capita
top_10_gdp_countries = df.nlargest(10, 'GDP per capita')

# 2. Line Chart: GDP per capita for top 10 countries
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Country or region', y='GDP per capita', data=top_10_gdp_countries, marker="o", ax=ax, color='orange')
ax.set_title('GDP per Capita by Top 10 Countries')
ax.set_xlabel('Country')
ax.set_ylabel('GDP per capita')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)



# 3. Map: Generosity by Country
fig = px.choropleth(df, locations='Country or region',locationmode='country names',color='Generosity',hover_name='Country or region',color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(title='Generosity by Country',geo=dict(showframe=False, projection_type='natural earth'))
st.plotly_chart(fig)




# 4. Slider for filtering by Healthy life expectancy

top_countries = df.nlargest(10, 'Freedom to make life choices')
min_life_exp, max_life_exp = st.slider('Select range for Healthy life expectancy',float(df['Healthy life expectancy'].min()), float(df['Healthy life expectancy'].max()),(0.8, 1.0))
filtered_data = df[(df['Healthy life expectancy'] >= min_life_exp) & (df['Healthy life expectancy'] <= max_life_exp)]
fig,ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Country or region', y='Healthy life expectancy', data=filtered_data, marker='o', ax=ax,color='skyblue')
ax.set_title('Healthy life expectancy by Country')
ax.set_xlabel('Country')
ax.set_ylabel('Healthy life expectancy')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)


# 5. Dropdown for selecting a specific country and displaying its data
country = st.selectbox('Select a country', df['Country or region'].unique())
country_data = df[df['Country or region'] == country]
st.write(f'Data for {country}')
st.write(country_data)



# 6.  Freedom and Corruption by Top 10 Countries
fig, ax = plt.subplots(figsize=(12, 8))
x = range(len(top_countries))
width = 0.4
ax.bar(x, top_countries['Freedom to make life choices'], width=width, label='Freedom to Make Life Choices', color='blue', align='center')
ax.bar([p + width for p in x], top_countries['Perceptions of corruption'], width=width, label='Perceptions of Corruption', color='orange', align='center')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(top_countries["Country or region"], rotation=45, ha='right')
ax.set_title('Freedom and Corruption by Top 10 Countries')
ax.set_xlabel('Country')
ax.set_ylabel('Values')
ax.legend()
st.pyplot(fig)



# 7. Correlation diagram: Relation between GDP per capita and Happiness Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='GDP per capita', y='Score', data=df, ax=ax,color='darkgrey')
ax.set_title('Relation between GDP per capita and Happiness Score')
ax.set_xlabel('GDP per capita',)
ax.set_ylabel('Happiness Score')
st.pyplot(fig)


# 8. Heatmap: Correlation between Social support and Score

columns_to_analyze = ['Score', 'Social support']
correlation_matrix = df[columns_to_analyze].corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, ax=ax)
ax.set_title('Correlation between Score and Social Support')
st.pyplot(fig)

