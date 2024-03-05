---
title: "Moroccan News Aggregator:"
datePublished: Tue Mar 05 2024 03:46:49 GMT+0000 (Coordinated Universal Time)
cuid: cltdtwa7u000609l8c7p881iv
slug: moroccan-news-aggregator
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1709541854890/0640a090-9bf4-4e8c-a19b-4014c0deb9a0.webp
tags: data-science, web-scraping, streamlit, moroccan-news

---

### Introduction

This article is your gateway to understanding and utilizing news data effectively. Here, we will walk you through a straightforward yet effective approach to gather, organize, analyze, and use data from Moroccan news websites. We will use easy-to-understand methods involving popular Python tools like Streamlit, Pandas, Selenium, along with Google Drive Handle and python-dotenv. Our aim is to make this journey from collecting data to applying it as smooth as possible for you.

### Understanding Data Scraping

Data scraping is the process of collecting information from websites. This is commonly done to gather data from various sources on the internet for analysis, research, or storage.

Data scraping can be especially useful for keeping track of changes on websites, gathering large amounts of data quickly, and automating repetitive tasks of collecting information.

However, it's important to remember that while data scraping is a powerful tool, it should be used responsibly and ethically. This means respecting website terms of use, considering data privacy laws, and not overloading websites with too many requests at once.

### Setting Up

To get started with our project, you'll need to set up your computer with some specific tools. Here's a list of what you need and how to install them:

1. **Streamlit**: This tool helps us build and share web applications easily. Install it using the command: `pip install streamlit`.
    
2. **Pandas**: A library that makes handling data easier. Install it with `pip install pandas`.
    
3. **Selenium (version 4.0.0 to less than 5.0.0)**: It's crucial for web scraping. Install the correct version using `pip install selenium==4.*`.
    
4. **PyDrive**: This tool will help us work with Google Drive files. Install it using `pip install PyDrive`.
    
5. **python-dotenv**: This library will help us manage environment variables securely. Install it with `pip install python-dotenv`.
    
6. **WebDriver Manager**: It helps in managing the browser drivers needed for Selenium. Install it using `pip install webdriver_manager`.
    
7. **BeautifulSoup4**: A library that makes it easy to scrape information from web pages. Install it with `pip install beautifulsoup4`.
    

With these tools installed, you'll be ready to start working on your data scraping and analysis project!

### Data Cleaning with Pandas

After successfully scraping data from websites, the next crucial step is cleaning this data. This means making sure the data is organized, free from errors, and ready for analysis. For this, we use Pandas, a powerful Python library that simplifies the process of data manipulation.

#### Getting Started with Pandas

1. **Loading Data**: Begin by loading the scraped data into Pandas for easy manipulation.
    
2. **Removing Duplicates**: Identify and remove any duplicate entries to ensure data quality.
    
3. **Handling Missing Values**: Find and address any missing or incomplete information in the dataset.
    
4. **Formatting Data**: Adjust data formats as needed for consistency and easier analysis.
    
    ```python
    import pandas as pd
    # Load the dataset
    df = pd.read_csv('scraped_articles.csv')
    # Remove duplicates
    df.drop_duplicates(subset='article_title', inplace=True)
    # Remove any rows with missing values
    df.dropna(inplace=True)
    df.to_csv('cleaned_articles.csv', index=False)
    ```
    

### **Integrating Google Drive API for Data Storage and Sharing:**

In our project, we leverage the Google Drive API to efficiently store and share the data we've collected. This approach not only provides a secure and scalable storage solution but also simplifies the process of making our data accessible to others. Here's how we do it:

#### Setting Up the Environment

1. **Importing Libraries**: We start by importing necessary libraries including `pydrive.auth`, `GoogleDrive`, and `oauth2client.client`.
    
2. **Environment Variables**: Using `load_dotenv` from the `dotenv` package, we load our Google API credentials stored in environment variables for security. This includes `CLIENT_ID`, `CLIENT_SECRET`, and `REFRESH_TOKEN`.
    

#### Authenticating with Google Drive

* **Authentication Function**: We define a function `authenticate_google_drive` which sets up the authentication using OAuth2 credentials. This ensures a secure connection to Google Drive.
    

#### Uploading Files to Google Drive

* **Upload Function**: The `upload_file_to_drive` function takes in the `drive` object and the file path to upload the file to Google Drive. We also handle the case where the file might already exist on Google Drive, updating it instead of uploading a duplicate.
    
* **Error Handling**: The function includes error handling to manage any issues during the upload process.
    

#### Generating Downloadable Links

* **Download Link Function**: The `get_drive_download_link` function is used to generate a direct download link for the files uploaded. This function sets the necessary permissions on the file to make it accessible to anyone with the link.
    

#### Practical Example

```python
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import OAuth2Credentials
import os
    
load_dotenv()
    
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')
REDIRECT_URI = os.getenv('REDIRECT_URIS').split(',')[0]  # Access the first URI
    
    def authenticate_google_drive():
        gauth = GoogleAuth()
        gauth.credentials = OAuth2Credentials(None, CLIENT_ID, CLIENT_SECRET,REFRESH_TOKEN, None,
                                             "https://accounts.google.com/o/oauth2/token", None, "web")
        drive = GoogleDrive(gauth)
        return drive
    
    drive = authenticate_google_drive()
    
    def upload_file_to_drive(drive, file_path, folder_id=None):
        if not os.path.exists(file_path):
            print(f"Cannot upload, file does not exist at path: {file_path}")
            return None
    
        try:
            file_metadata = {'title': os.path.basename(file_path)}
            if folder_id:
                file_metadata['parents'] = [{'id': folder_id}]
    
            upload_file = drive.CreateFile(file_metadata)
    
            # Check if the file already exists on Google Drive
            existing_files = drive.ListFile({'q': f"title='{upload_file['title']}'"}).GetList()
            if existing_files:
                # File with the same name already exists, update the existing file
                upload_file = existing_files[0]
                print(f"File already exists on Drive. Updating file with ID: {upload_file['id']}")
            else:
                print("Uploading a new file to Drive.")
    
            upload_file.SetContentFile(file_path)
            upload_file.Upload()
            print(f"File uploaded successfully. File ID: {upload_file['id']}")
            return upload_file['id']
        except Exception as e:
            print(f"An error occurred during file upload: {e}")
            return None
    
    
    def get_drive_download_link(drive, file_id):
        try:
            file = drive.CreateFile({'id': file_id})
            file.Upload() # Make sure the file exists on Drive
            file.InsertPermission({
                'type': 'anyone',
                'value': 'anyone',
                'role': 'reader'})
            return "https://drive.google.com/uc?export=download&id=" + file_id
        except Exception as e:
            print(f"Error fetching download link: {e}")
            return None
```

#### Conclusion

This integration of the Google Drive API provides a robust and secure method for storing and sharing large datasets. By automating the upload and link generation process, we significantly enhance the accessibility and usability of our data.

### Streamlit Web Interface

Our Streamlit application serves as a central hub for aggregating news from various Moroccan websites. It's designed to be intuitive and user-friendly, enabling users to select news sources, languages, and categories for scraping.

#### Key Features

1. **Dynamic Configuration**: Users can choose websites and specific categories from a dynamically loaded configuration (`config.json`). This allows for a customizable scraping experience.
    
2. **Language and Category Selection**: For websites offering content in multiple languages, users can select their preferred language. Additionally, users can pick specific categories of news they are interested in.
    
3. **Control Over Data Collection**: Through a simple interface, users can specify the number of articles to scrape.
    
4. **Initiating Scraping**: A 'Start Scraping' button triggers the scraping process, with a progress bar indicating the ongoing operation.
    
5. **Real-time Updates and Data Display**: As data is scraped and uploaded, users receive real-time updates. Each successful scrape results in a download link for the data and a display of the scraped data in a tabular format within the application.
    
6. **Google Drive Integration**: Scraped data files are uploaded to Google Drive, and direct download links are provided within the Streamlit interface for easy access.
    
7. **Error Handling**: The application includes error handling for issues like failed file uploads or unsuccessful scrapes, ensuring a smooth user experience.
    
    ```python
    import streamlit as st
    import pandas as pd
    import json
    import importlib
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    import google_drive_handle as gdrive
    from dotenv import load_dotenv
    import os
    
    # Load config.json
    with open('config.json') as f:
        config = json.load(f)
    
    # Set up Chrome WebDriver with options
    options = ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('log-level=3')
    # Initialize the Chrome WebDriver
    wd = webdriver.Chrome(options=options)
    drive = gdrive.authenticate_google_drive()
    processed_files = set()
    st.markdown(
        """
        <style>
            .centered {
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 class='centered'>Moroccan News Aggregator</h1>", unsafe_allow_html=True)
    
    selected_websites = {}
    selected_categories = {}
    
    def save_file_id_mapping(file_id_mapping):
        with open("file_id_mapping.json", "w") as file:
            json.dump(file_id_mapping, file)
    
    def load_file_id_mapping():
        try:
            with open("file_id_mapping.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}  # Return an empty dictionary if the file doesn't exist
    
    file_id_mapping = load_file_id_mapping()
    
    selected_websites = {}
    
    for website, details in config.items():
        if st.checkbox(website, key=website):
            # Language selection
            languages = details.get("languages", {})
            if languages and len(languages) > 1:
                language = st.selectbox(f'Choose language for {website}', list(languages.keys()), key=f'lang_{website}')
                selected_websites[website] = f"{website}_{language}"  # like: hespress_en
            else:
                selected_websites[website] = website  # like: akhbarona
    
            # Category selection
            categories = languages.get(language, {})
            if categories:
                categories = st.multiselect(f'Select categories for {website}', list(categories.keys()), key=f'{website}_categories')
                selected_categories[website] = categories
    
    # Number of articles input
    num_articles = st.number_input('Number of Articles', min_value=1, max_value=10000, step=1)
    
    # Start scraping button
    if st.button('Start Scraping'):
        with st.spinner('Scraping in progress...'):
            progress_bar = st.progress(0)
            total_tasks = sum(len(categories) for categories in selected_categories.values())
            completed_tasks = 0
            for website, module_name in selected_websites.items():
                scraper_module = importlib.import_module(module_name)
                for category in selected_categories.get(website, []):
                    category_url = config[website]['languages'][language][category]
                    if 'category_name' in config[website]:
                        category_name = config[website]['category_name'].get(category, 'default_category_name')
                    file_path = scraper_module.scrape_category(category_url, num_articles)
    
                    if file_path:
                        if file_path not in file_id_mapping:
                            file_id = gdrive.upload_file_to_drive(drive, file_path)
                            print(f"Uploading file: {file_path}, File ID: {file_id}")
                            file_id_mapping[file_path] = file_id
                            save_file_id_mapping(file_id_mapping)
                        else:
                            file_id = file_id_mapping[file_path]
                            print(f"File already uploaded. Using existing File ID: {file_id}")
    
                        if file_id:
                            download_link = gdrive.get_drive_download_link(drive, file_id)
                            if download_link:
                                st.markdown(f"[Download {website} - {category} data]({download_link})", unsafe_allow_html=True)
    
                                df = pd.read_csv(file_path)
                                st.write(f"{website} - {category} Data:")
                                st.dataframe(df)
                            else:
                                st.error(f"Failed to retrieve download link for file ID: {file_id}")
                        else:
                            st.error(f"Failed to upload file for {website} - {category}")
                    else:
                        st.error(f"File not created for {website} - {category}")
    
            st.success('Scraping Completed!')
    ```
    

This Streamlit web interface stands as a testament to the power of Python in creating efficient, user-friendly tools for data aggregation and management. It simplifies the complex process of data collection, storage, and sharing, making it accessible even to those with minimal technical background.

### Dynamic Configuration with 'config.json'

Our Streamlit application is designed with agility and future expansion in mind. By incorporating a `config.json` file, we've created a flexible framework that allows for easy addition and modification of news sources.

#### The Role of `config.json`

* **Flexible Source Management**: The `config.json` file holds the details of various news websites, their available languages, and specific category URLs. This setup enables us to easily add new sources or modify existing ones without altering the core code of the application.
    
* **Language and Category Customization**: For each news website, multiple languages and categories are defined. Users can select their preferred language and categories, making the data scraping process highly customizable.
    

#### Implementation in Streamlit

* **Loading Configuration**: At the start of the application, `config.json` is loaded to dynamically populate the website choices, along with their respective languages and categories.
    
* **User Interactions**: Users interact with checkboxes and dropdowns generated based on the configuration. They can select websites, languages, and specific categories for scraping.
    
* **Scalability**: Adding a new website, language, or category is as simple as updating the `config.json` file, making the application scalable and easy to maintain.
    

#### Advantages

* **Maintainability**: Changes to the list of websites or their categories don't require code changes, reducing maintenance complexity.
    

**User Experience**: Provides a user-friendly interface where options are dynamically generated, offering a seamless and intuitive experience.

### Demo

%[https://huggingface.co/spaces/MoroccanDS/A8-Moroccan-News-Aggregator] 

## Conclusion

Our journey through scraping, cleaning, and deploying data from Moroccan news websites has been a testament to the power and flexibility of Python and its libraries. By leveraging tools like Selenium, Pandas, Streamlit, and Google Drive API, we've demonstrated a streamlined process that transforms raw data into accessible and actionable insights. This project not only showcases the technical capabilities of these tools but also highlights the potential for data-driven strategies in understanding and disseminating information effectively.

## Acknowledgements

First and foremost, a heartfelt thanks to my dedicated teammates - @[Tajeddine Bourhim](@ScorpionTaj) , @[Yahya NPC](@Steevie) and @marwane khadrouf. Their expertise, creativity, and commitment were pivotal in turning this concept into reality. Their contributions in various aspects of the project, from data scraping to interface design, have been invaluable.

We also extend our gratitude to the founder of MDS and the initiator of the DataStart initiative @[Bahae Eddine Halim](@bahae) . This initiative not only kick-started our project but also inspired us to delve into the realm of data handling and analysis, contributing to a range of projects including this one.

this project stands as a collaborative effort, blending individual talents and shared vision. It's a celebration of teamwork, innovation, and the endless possibilities that open-source technology and data science bring to our world.