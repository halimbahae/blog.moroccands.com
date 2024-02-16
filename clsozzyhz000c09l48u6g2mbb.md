---
title: "Exploring Moroccan Real Estate: Trends and Price Predictions"
seoTitle: "Exploring Moroccan Real Estate: Trends and Price Predictions"
seoDescription: "Embark on a journey through the vibrant landscape of Moroccan real estate with Moroccan Data Scientists."
datePublished: Fri Feb 16 2024 18:43:23 GMT+0000 (Coordinated Universal Time)
cuid: clsozzyhz000c09l48u6g2mbb
slug: exploring-moroccan-real-estate-trends-and-price-predictions
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1706875172477/8ed4b88d-c625-417f-8f13-f0d4ab66ef8d.jpeg
tags: data-science, real-estate, deep-learning, data-scraping, geospatial-visualisation

---

<details>
<summary>Table of Contents:</summary>
<ul>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#data-scraping">Data Scraping</a></li>
<li><a href="#data-cleaning">Data Cleaning</a></li>
<li><a href="#exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a></li>
<li><a href="#model-development">Model Development</a></li>
<li><a href="#model-deployment">Model Deployment</a></li>
<li><a href="#conclusion">Conclusion</a></li>
<li><a href="#acknowledgments">Acknowledgments</a></li>
</ul>
</details>

# [Introduction](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-introduction)

Hello there! Are you up for an exciting journey through Morocco's real estate scene? From the bustling markets to the hidden gems of Marrakech and Chefchaouen, we're diving into the heart of it all. Join us as we decode the secrets of Moroccan property prices and unveil the magic of predictive analytics. Get ready to explore, analyze, and uncover the possibilities in this thrilling adventure! 🌟🔍

# [Data Scraping](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-datascraping)

In the realm of real estate analytics, data is king. But how do we access this treasure trove of information scattered across the vast expanse of the internet? Enter web scraping – the digital prospector's tool of choice.

**What is Web Scraping?**

Web scraping is the art of extracting data from websites, allowing us to collect valuable information for analysis and insights. In the context of our project, web scraping enables us to gather real estate listings, economic indicators, and other pertinent data from online sources.

**Why is it Important?**

In the dynamic world of real estate, having access to up-to-date and comprehensive data is crucial for making informed decisions. Web scraping empowers us to gather vast amounts of data efficiently, giving us a competitive edge in predicting real estate prices and understanding market trends.

**Scraping Data from Mubawab with BeautifulSoup**

Our journey begins with [Mubawab](https://www.mubawab.ma/), a prominent real estate website in Morocco, brimming with valuable listings and insights. Using the Python library BeautifulSoup, we embark on our quest to extract data from Mubawab with precision and finesse.

**The Process**:

*Exploration*: We start by inspecting the structure of the Mubawab website, identifying the key elements containing the data we seek – from property listings to pricing information.

*Scripting*: Armed with our knowledge of HTML and CSS, we craft Python scripts leveraging BeautifulSoup to navigate through the website's code and extract the desired data.

*Extraction*: With surgical precision, our scripts traverse through the pages of Mubawab, capturing essential details such as property names, prices, locations, area, and more.

*Aggregation:* As the data flows in, we gather and organize it into structured formats such as Excel spreadsheets, ready for further analysis and processing.

**Python Code Snippets:**

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, urljoin

# Define the base URL of Mubawab
baseurl = 'https://www.mubawab.ma/'

# Set to store unique product links
product_links = set()

# Loop through multiple pages of apartment listings
for page_num in range(1, 3):
    # Make a GET request to the page
    r = requests.get(f'https://www.mubawab.ma/fr/sc/appartements-a-vendre:p:{page_num}')
    soup = BeautifulSoup(r.content, 'html.parser')

   # Extract listings from the page
    listings = soup.find_all(class_='listingBox w100')

    # Loop through each listing and extract product links
    for listing in listings:
        links = listing.find_all('a', href=True)
        for link in links:
            product_link = link['href']

            # Check if the URL is valid
            if not urlparse(product_link).scheme:
                product_link = urljoin(baseurl, product_link)
                product_links.add(product_link)  # Add link to the set

# Initialize list to store scraped data
scraped_data = []

# Loop through each product link and scrape data
for link in product_links:
    try:
        # Make a GET request to the product page
        r = requests.get(link)
        soup = BeautifulSoup(r.content)
        # Extract relevant information from the page
        # (Code snippet continued from previous section...)
    except Exception as e:
        print('Error processing page:', link)
        print(e)
        continue

# Convert scraped data to DataFrame and save to Excel
df = pd.DataFrame(scraped_data)
df.to_excel('mubawab_data.xlsx', index=False)
```

With web scraping as our trusty pickaxe, we delve into the digital mines of Mubawab, extracting nuggets of real estate data to fuel our predictive models and illuminate the path to informed decision-making. Let the data adventure begin! 🏠💻

# [Data Cleaning](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-datascleaning)

**Why Data Cleaning Matters:**

Before we can uncover the hidden patterns and insights within our data, we must first ensure its integrity and quality. Data cleaning plays a pivotal role in this process, serving as the foundation upon which our analyses and models are built. By removing inconsistencies, handling missing values, and standardizing formats, we pave the way for accurate and reliable results.

**Steps in Data Cleaning:**

Removing Duplicates: Duplicate entries can skew our analyses and lead to erroneous conclusions. By identifying and eliminating duplicates, we ensure that each observation contributes meaningfully to our insights.

Handling Missing Values: Missing data is a common challenge in real-world datasets. Whether due to human error or system limitations, missing values must be addressed to maintain the integrity of our analyses. Strategies such as imputation or removal can be employed based on the nature and context of the missing data.

Standardizing Data Types: Inconsistent data types can impede our ability to perform meaningful calculations and comparisons. Standardizing data types ensures uniformity and facilitates seamless data manipulation and analysis.

**Python Code Snippets:**

```python

# Load the scraped data after some cleaning with VBA Macros

df = pd.read_excel('mubawab_data.xlsx')

# Remove duplicate rows

df.drop_duplicates(inplace=True)

# Drop rows with missing prices

df.dropna(subset=['price'], inplace=True)

# Fill missing values in 'city' with corresponding values from 'secteur'

df['city'].fillna(df['secteur'], inplace=True)

# Fill missing values in 'sdb' and 'chambres' with the mean

mean_sdb = df['sdb'].mean()

mean_chambres = df['chambres'].mean()

mean_surface = df['surface'].mean()

df['sdb'].fillna(mean_sdb, inplace=True)

df['chambres'].fillna(mean_chambres, inplace=True)

df['surface'].fillna(mean_surface, inplace=True)

# Fill missing values in 'pieces' with the sum of 'sdb' and 'chambres'

df['pieces'].fillna(df['sdb'] + df['chambres'], inplace=True)

# One-hot encode 'secteur' and 'city' columns

df = pd.get_dummies(df, columns=['secteur', 'city'])

# Display the cleaned DataFrame

print(df.head())
```

**Example:**

Let's say our original dataset contains the following entries:

| **name** | **price** | **secteur** | **surface** | **pieces** | **chambres** | **sdb** | **etat** | **age** | **etage** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Appartement de 65m² en vente, Complexe Résidentiel Jnane Azzahrae | 250 000 DH | Ain Atig àTemara | 65m² | 4 Pièces | 3 Chambres | 2 Salles de bains | Nouveau |  |  |
| Av grand appartement terrasse mohammedia centre | 2 300 000 DH | Centre Ville àMohammedia | 169m² | 6 Pièces | 3 Chambres | 2 Salles de bains | Bon état | 5-10 ans | 5èmeétage |
| Appartement à vendre 269 m², 3 chambres Val Fleurie Casablanca | 3 800 000 DH | Val Fleury àCasablanca | 269m² | 4 Pièces | 3 Chambres | 2 Salles de bains | Bon état | 10-20 ans | 1erétage |
| Appartement de 105 m² en vente, Rio Beach | 9 900 DH | Sidi Rahal | 105m² | 3 Pièces | 2 Chambres | 2 Salles de bains | Nouveau | Moins d'un an |  |
| Studio Meublé moderne, à vendre | 1 360 000 DH | Racine àCasablanca | 57m² | 2 Pièces | 1 Chambre | 1 Salle de bain | Nouveau |  |  |
| Appartement 99 m² a vendre M Océan | 1 336 500 DH | Mannesmann àMohammedia | 99m² | 3 Pièces | 2 Chambres | 2 Salles de bains | Nouveau | 1-5 ans |  |
| Appartement à vendre 88 m², 2 chambres Les princesses Casablanca | 1 300 000 DH | Maârif Extension àCasablanca | 88m² | 2 Chambres | 2 Salles de bains | 4èmeétage |  |  |  |

After applying data cleaning techniques, our cleaned dataset might look like this:

| **name** | **price** | **secteur** | **city** | **surface** | **pieces** | **chambres** | **sdb** | **age** | **etage** | **etat\_Bon état** | **etat\_Nouveau** | **etat\_À rénover** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Appartement de 65m² en vente, Complexe Résidentiel Jnane Azzahrae | 250000,00 | Ain Atig | Temara | 65 | 4 | 3 | 2 | 5,43647 | 1,771796 | 0 | 1 | 0 |
| Vente appartement à quartier geliz marrakech | 135000,00 | Guéliz | Marrakech | 104 | 6 | 3 | 1 | 5,43647 | 1,771796 | 1 | 0 | 0 |
| Av grand appartement terrasse mohammedia centre | 2300000,00 | Centre Ville | Mohammedia | 169 | 6 | 3 | 1 | 7,5 | 5 | 1 | 0 | 0 |
| Appartement à vendre 269 m², 3 chambres Val Fleurie Casablanca | 3800000,00 | Val Fleury | Casablanca | 269 | 4 | 3 | 1 | 15 | 1 | 1 | 0 | 0 |
| Appartement de 105 m² en vente, Rio Beach | 9900,00 | Sidi Rahal | Marrakech | 105 | 3 | 2 | 1 | 0,5 | 1,771796 | 0 | 1 | 0 |
| Très joli appartement en vente meublé | 2000000,00 | Camp Al Ghoul | Marrakech | 99 | 3 | 2 | 1 | 3,5 | 3 | 0 | 1 | 0 |
| Appartement à vendre 259 m², 4 chambres Maârif Casablanca | 4400000,00 | Maârif | Casablanca | 259 | 6 | 4 | 1 | 15 | 2 | 1 | 0 | 0 |

**Conclusion:**

Data cleaning is the cornerstone of robust data analysis. By ensuring the cleanliness and consistency of our datasets, we lay the groundwork for accurate insights and informed decision-making. With our data now polished to perfection, we are ready to embark on the next stage of our real estate journey

# [Exploratory Data Analysis (EDA)](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-eda)

**Understanding the Dataset:**

Exploratory Data Analysis (EDA) serves as our compass, guiding us through the vast landscape of real estate data. It empowers us to uncover hidden patterns, identify outliers, and gain valuable insights into the dynamics of the market.

**Significance of EDA:**

Understanding Real Estate Dynamics: EDA allows us to delve deep into the intricacies of the real estate dataset, unraveling the relationships between various factors such as property characteristics, location, area, and pricing.

Identifying Patterns and Trends: By analyzing descriptive statistics and visualizations, we can identify trends over time, seasonal fluctuations, and spatial disparities in property prices.

Informing Decision-Making: Insights gleaned from EDA serve as the cornerstone for informed decision-making, whether it be for investors, developers, or policymakers.

**Visualizations and Descriptive Statistics:**

Distribution of Real Estate Prices: Histograms and boxplots provide a snapshot of the distribution of property prices, highlighting central tendencies, variability, and potential outliers.

Relationships Between Variables: Scatter plots and correlation matrices help us explore the relationships between different variables, such as property size, number of rooms, and prices, shedding light on potential predictors of property value.

Temporal Trends: Time series plots allow us to visualize temporal trends in property prices, discerning patterns, seasonality, and long-term trends.

**Geospatial Visualizations:**

Interactive Maps: Utilizing geospatial data, we can create interactive maps to visualize property locations, hotspots, and regional disparities in prices. This allows stakeholders to explore the real estate landscape at a glance and identify areas of interest.

Heatmaps: Heatmaps offer a bird's-eye view of property density and price distribution, providing valuable insights into market saturation and demand hotspots.

**Conclusion:**

Exploratory Data Analysis serves as our compass, guiding us through the labyrinth of real estate data. By unraveling patterns, trends, and spatial dynamics, EDA equips us with the insights needed to navigate the complexities of the real estate market with confidence and clarity. With our eyes opened to the rich tapestry of real estate data, we are poised to unlock its full potential and drive informed decision-making in the ever-evolving landscape of real estate. 📊🏠

# [Model Development](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-modeldev)

Model development marks the culmination of our journey, where we harness the power of machine learning to predict real estate prices with precision and accuracy. This transformative process involves a series of meticulous steps, each contributing to the creation of robust predictive models.

**Model Development Process:**

Data Normalization: We embark on our journey by normalizing the data, ensuring that all features are on a consistent scale. This step prevents certain features from dominating the model training process and ensures optimal performance.

Feature Selection: With a myriad of features at our disposal, we carefully select the most influential ones to include in our predictive models. Through feature selection techniques, we prioritize variables that exhibit strong correlations with property prices and contribute meaningfully to the predictive power of our models.

Model Training: Armed with a curated dataset and selected features, we embark on the model training phase. Leveraging powerful machine learning libraries such as TensorFlow, we train regression models to learn from historical data and discern intricate patterns in real estate pricing dynamics.

Evaluation Metrics: As stewards of data-driven decision-making, we rely on rigorous evaluation metrics to assess the performance of our models. Metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) serve as our compass, guiding us towards models that exhibit optimal predictive accuracy and generalization.

**Harnessing TensorFlow and Beyond:**

TensorFlow: TensorFlow stands as our stalwart companion on the journey of model development, providing a versatile framework for building and training regression models. With its intuitive interface and powerful capabilities, TensorFlow empowers us to bring our predictive visions to life with elegance and efficiency.

Machine Learning Libraries: In addition to TensorFlow, we harness a diverse array of machine learning libraries such as scikit-learn and Keras to augment our model development efforts. These libraries offer a rich ecosystem of algorithms and tools, enabling us to experiment, iterate, and refine our predictive models with finesse.

**python Code Snippet**

```python

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset and preprocess features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Define and train the TensorFlow regression model

model = tf.keras.Sequential([

tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dense(1)

])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

print(f'Mean Absolute Error: {mae}')
```

**Conclusion:**

Model development serves as the cornerstone of our predictive journey, where we transform data into actionable insights and empower stakeholders to make informed decisions in the dynamic realm of real estate. Through meticulous attention to detail, rigorous experimentation, and the judicious application of machine learning techniques, we pave the way for predictive models that illuminate the path forward with clarity and confidence. 🤖🏠📈

# [Model Deployment](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-deployment)

**Deployment Process:**

As we approach the culmination of our journey, it's time to unleash the predictive prowess of our meticulously crafted models into the real world. Model deployment represents the pivotal moment when theoretical concepts seamlessly transition into practical applications, offering actionable insights and informed decision-making capabilities to stakeholders.

**Deployment Landscape:**

Real-World Integration: Our trained models are seamlessly integrated into real-world environments, where they stand ready to analyze incoming data and provide valuable predictions on real estate prices. Whether it's assisting homebuyers in making informed purchase decisions or aiding industry professionals in strategic planning, our deployed models serve as beacons of predictive wisdom.

Deployment Overview: Our deployment offers a glimpse into the intuitive interface through which users can interact with the deployed model. By inputting relevant property features such as surface area, number of rooms, and location, users can harness the predictive power of our models to obtain accurate price predictions in a matter of seconds.

User Interaction: Interacting with our deployed model is as simple as entering the desired property features into the designated input fields. With just a few clicks, users gain access to personalized price predictions tailored to their specific requirements, empowering them to navigate the intricacies of the real estate market with confidence and clarity.

Ready to experience the magic of predictive analytics firsthand? Explore our deployment and witness the transformative potential of data-driven insights in action. Click here to embark on a journey of discovery and unlock the secrets of real estate pricing dynamics with just a click of a button.

[https://huggingface.co/spaces/saaara/real\_estate\_price\_prediction](https://huggingface.co/spaces/saaara/real_estate_price_prediction)

**Conclusion:**

Model deployment represents the culmination of our journey, where theoretical concepts are transformed into tangible solutions that empower individuals and organizations to make informed decisions in the ever-evolving landscape of real estate. Through seamless integration, intuitive interfaces, and the democratization of predictive analytics, we pave the way for a future where data-driven insights drive meaningful change and innovation.

# [Conclusion](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-conclusion)

We've made our data analysis and modeling process interactive and accessible by sharing our Google Colab notebook. Dive deeper into the intricacies of our real estate analysis, run code cells, visualize data, and even experiment with our predictive models here:

[https://colab.research.google.com/drive/1sWd5QhPXL0MpLsRsYBb7JuxK8uDTYCaq?authuser=1#scrollTo=iJF6DFW618jR](https://colab.research.google.com/drive/1sWd5QhPXL0MpLsRsYBb7JuxK8uDTYCaq?authuser=1#scrollTo=iJF6DFW618jR)

As our journey through the labyrinth of Moroccan real estate draws to a close, we stand amidst a landscape adorned with insights, revelations, and transformative discoveries. Through the collective efforts of our dedicated team, we have unearthed invaluable treasures that illuminate the intricate dynamics of real estate pricing in Morocco. Our models exhibit remarkable accuracy in forecasting real estate prices, leveraging a nuanced understanding of location, amenities, and economic trends. By harnessing the power of web scraping and exploratory data analysis, we have curated a comprehensive dataset, laying the foundation for informed decision-making. With the deployment of our models, we bridge the gap between theory and practice, offering intuitive interfaces to access predictive analytics, empowering users to navigate the complexities of the real estate market with confidence. Looking ahead, we remain committed to continuous innovation, exploring novel methodologies and technologies to enhance the accuracy of our models. While reflecting on our journey, we recognize areas for improvement, acknowledging challenges faced and lessons learned.

**A Call to Action:**

Embark on your own journey of exploration and discovery within the realm of Moroccan real estate. Whether you're a seasoned professional, an aspiring enthusiast, or a curious explorer, there's a wealth of insights and opportunities waiting to be uncovered. Connect with our team to learn more about our endeavors, collaborate on future projects, or simply indulge in the fascinating world of real estate analytics. Together, let's chart a course towards a future where data-driven insights illuminate the path to prosperity and growth. 🌟🔍🏠

# [Acknowledgments](https://hashnode.com/draft/65bccc56d821d9fd24722c81#heading-acknowledgments)

I would like to express my sincere appreciation to my dedicated team members – Sara M'HAMDI, Imane KARAM, and Asmae EL-GHEZZAZ – whose expertise and commitment have been invaluable throughout this project. Your hard work, collaboration, and enthusiasm have truly made a difference. As the team leader, I am incredibly proud to have worked alongside such talented individuals.

I also want to acknowledge Bahaeddine Halim, the founder of the Moroccan Data Science (MDS) community, whose initiative, DataStart First Edition, provided the platform for our project. His dedication to fostering a supportive environment for data enthusiasts in Morocco has been instrumental in our journey. Lastly, we thank the broader data science community for their support and encouragement. Your enthusiasm and engagement have motivated us to push boundaries and continuously strive for excellence in our endeavors.

To all those who have contributed to this project – mentors, team members, and supporters – we express our heartfelt thanks. Your collective efforts have been integral to the success of this endeavor, and we look forward to continued collaboration and growth in the future.