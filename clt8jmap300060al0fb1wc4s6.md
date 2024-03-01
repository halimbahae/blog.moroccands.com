---
title: "Python based currency converter web app : A simple approach"
datePublished: Fri Mar 01 2024 11:00:16 GMT+0000 (Coordinated Universal Time)
cuid: clt8jmap300060al0fb1wc4s6
slug: python-based-currency-converter-web-app-a-simple-approach
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/jgOkEjVw-KM/upload/a00f7122862ab57ad248a495b9db197b.jpeg
tags: python, data-science, web-development, apis, python3, ui-design, scraping, beautifulsoup, currency-converter, matplotlib, streamlit, currency, huggingface, morocco, morrocan

---

## Introduction

In today's globalized world, the ability to swiftly convert currencies is a necessity for travelers, businesses, and finance enthusiasts alike. In this blog, we delve into the fascinating realm of currency conversion, showcasing a Python-based web application that brings real-time exchange rates to your fingertips.

Join us on this journey as we unveil the inner workings of our currency converter, crafted with the powerful combination of Python, Streamlit, and BeautifulSoup. We'll explore the challenges and triumphs of sourcing current exchange rates, and the seamless user experience delivered by our intuitive web interface.

So sit back, relax, and prepare to embark on a voyage through the dynamic landscape of currency conversion, guided by the ingenuity of Python's versatile ecosystem. Let's dive in! 🌐💱

## What is streamlit

Streamlit is a cutting-edge Python library that empowers developers to create interactive web applications with remarkable ease and speed. What sets Streamlit apart is its simplicity and focus on the developer experience. With just a few lines of Python code, developers can transform their data scripts into polished, user-friendly web apps. Streamlit handles all the heavy lifting behind the scenes, from data visualization to user input handling, allowing developers to focus on crafting engaging experiences for their users. Whether you're a seasoned developer or just starting out, Streamlit offers a seamless and intuitive platform for bringing your ideas to life on the web.

## What is beautifulSoup

Beautiful Soup is a Python library renowned for its ability to parse HTML and XML documents, making web scraping and data extraction a breeze. It provides a powerful yet user-friendly interface for navigating and manipulating parsed documents, allowing developers to extract relevant information with ease. Beautiful Soup's intuitive syntax and robust functionality make it a go-to choice for extracting data from websites of all complexities. Whether you're scraping a simple webpage or traversing a complex hierarchy of elements, Beautiful Soup streamlines the process, enabling developers to focus on extracting insights from the data rather than wrestling with the intricacies of web parsing. With its versatility and reliability, Beautiful Soup continues to be a cornerstone tool for web scraping tasks across various domains and industries.

* ### Code snippet (main page)
    

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

    
def rate_parser(input_curr, output_curr):
    url = f"https://www.xe.com/currencyconverter/convert/?Amount=1&From={input_curr}&To={output_curr}"
    content = requests.get(url).text
    soup = BeautifulSoup(content, 'html.parser')

    result_element = soup.find("p", class_="result__BigRate-sc-1bsijpp-1 dPdXSB")

    if result_element:
        currency_text = result_element.get_text().replace(',', '')  # Remove comma
        rate = float(currency_text.split()[0])
        return rate
    else:
        print(f"Element not found for {input_curr} to {output_curr}.")
        return None
  
    
def convert(base,dest,amount):
    rate = rate_parser(base,dest)
    new_amount = rate * amount
    return new_amount 


currencies = ["USD","EUR","CAD","MAD","GBP","AUD","JPY"]
#this is just a test list you can add any currency available at xe-currency

st.write("# Python based currency converter")

st.sidebar.write("### Currency converter:")

base = st.sidebar.selectbox("Enter a base currency:",currencies)

dest = st.sidebar.selectbox("Enter a destination currency:",currencies)

amount = st.sidebar.number_input("Enter an amount")

input = st.sidebar.button("Convert")

if input:
    current_rate = rate_parser(base,dest)
    output = convert(base,dest,amount)
    st.success("Success")
    st.write(f"## Current exchange rate between {base} and {dest} ")
    st.write(f"#### 1 {base} = ")
    st.write(f" ## :red[{current_rate}] {dest}")

    st.write("## Converted amount:")
    st.write(f" ### {amount} {base} = :red[{output}] {dest}")
```

## Why we chose scraping over API

In our pursuit of building a Python-based currency converter web app, the decision to opt for web scraping over utilizing an API was rooted in the specific requirements of our project. While APIs offer a convenient way to access data, particularly for real-time exchange rates, we encountered limitations with free API plans, which typically impose restrictions on the number of requests allowed per day. This became a significant constraint when attempting to gather historical exchange rate data, as it necessitated a considerable number of requests.

In light of this, we shifted our approach to web scraping using BeautifulSoup. While scraping provided the flexibility needed for historical data retrieval, it introduced the trade-off of slower performance compared to APIs. The notable advantage, however, lay in the ability to extract comprehensive historical datasets without the constraints imposed by API rate limits. This strategic decision allowed us to balance our data acquisition requirements, ensuring the reliability and completeness of the information while acknowledging the potential trade-offs in speed associated with web scraping.

* ### Code snippet (historical data)
    

```python
import pandas as pd
import requests
from datetime import timedelta, date
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#Create a function to return a list of each day between two dates

def daterange(start_date, end_date):
    list = []
    for n in range(int ((end_date - start_date).days)):
        list.append(start_date + timedelta(n))
    return list

#Create a function that scrapes historical data nb(all apis didn't work well)

def historical_data(start_date,end_date,base,dest):
    df = pd.DataFrame()
    times = daterange(start_date, end_date)
    for single_date in times:
        dfs = pd.read_html(f'https://www.xe.com/currencytables/?from={base}&date={single_date.strftime("%Y-%m-%d")}')[0]
        dfs['Date'] = single_date.strftime("%Y-%m-%d")
        df = pd.concat([df, dfs], ignore_index=True)

    df_curr=df.loc[df['Currency']==dest]
    df_curr = df_curr.reset_index(drop=True)
    df_curr.set_index('Date',inplace = True)
    return df_curr


#Create a list of all supported currencies

currencies = ["USD","EUR","CAD","MAD","GBP","AUD","JPY"]
#this is just a test list
#Expend the list as needed

#Create the UI using streamlit

st.write("# Historical Data:")
st.warning("This is originally a scraping webapp so choosing a large duration might cause substancial running time")
st.warning("Recommanded max duration : 1 Month")

base = st.sidebar.selectbox("Enter a base currency:",currencies)

dest = st.sidebar.selectbox("Enter a destination currency:",currencies)

start = st.sidebar.date_input("Enter start date:")

finish = st.sidebar.date_input("Enter finish date:")

input = st.sidebar.button("Confirm")

if input:
    if start == finish:
        st.error("Error: cannot process same date")
    with st.spinner('Wait for it...'):
        data = historical_data(start,finish,base,dest)
    st.success('Done!')
    st.table(data)
    st.write("## Plotting")
    
    st.write("### Static plot:")
    fig, ax = plt.subplots()
    ax.plot(data[f"{base} per unit"])
    ax.set_title(f'{base} to {dest} over Time')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    st.pyplot(fig)
```

### Example

|  | Currency | Name | Units per MAD | MAD per unit |
| --- | --- | --- | --- | --- |
| 2024-02-01 | USD | US Dollar | 0.0999 | 10.0087 |
| 2024-02-02 | USD | US Dollar | 0.0996 | 10.0422 |
| 2024-02-03 | USD | US Dollar | 0.0996 | 10.0381 |
| 2024-02-04 | USD | US Dollar | 0.0996 | 10.0381 |
| 2024-02-05 | USD | US Dollar | 0.0992 | 10.0827 |
| 2024-02-06 | USD | US Dollar | 0.0993 | 10.0722 |
| 2024-02-07 | USD | US Dollar | 0.0994 | 10.0611 |
| 2024-02-08 | USD | US Dollar | 0.0995 | 10.0458 |
| 2024-02-09 | USD | US Dollar | 0.0997 | 10.0340 |
| 2024-02-10 | USD | US Dollar | 0.0997 | 10.0327 |
| 2024-02-11 | USD | US Dollar | 0.0997 | 10.0327 |
| 2024-02-12 | USD | US Dollar | 0.0996 | 10.0396 |
| 2024-02-13 | USD | US Dollar | 0.0993 | 10.0679 |
| 2024-02-14 | USD | US Dollar | 0.0993 | 10.0744 |
| 2024-02-15 | USD | US Dollar | 0.0994 | 10.0646 |
| 2024-02-16 | USD | US Dollar | 0.0994 | 10.0643 |
| 2024-02-17 | USD | US Dollar | 0.0994 | 10.0636 |
| 2024-02-18 | USD | US Dollar | 0.0994 | 10.0642 |
| 2024-02-19 | USD | US Dollar | 0.0991 | 10.0882 |
| 2024-02-20 | USD | US Dollar | 0.0992 | 10.0763 |
| 2024-02-21 | USD | US Dollar | 0.0992 | 10.0761 |
| 2024-02-22 | USD | US Dollar | 0.0994 | 10.0556 |
| 2024-02-23 | USD | US Dollar | 0.0994 | 10.0557 |
| 2024-02-24 | USD | US Dollar | 0.0994 | 10.0555 |
| 2024-02-25 | USD | US Dollar | 0.0995 | 10.0553 |
| 2024-02-26 | USD | US Dollar | 0.0995 | 10.0497 |
| 2024-02-27 | USD | US Dollar | 0.0993 | 10.0663 |
| 2024-02-28 | USD | US Dollar | 0.0990 | 10.1015 |

### Deployment

We chose Hugging Face for deployment due to its reputation as a leading platform for deploying machine learning models with ease and efficiency. Hugging Face offers a comprehensive suite of tools and services that streamline the deployment process, enabling us to seamlessly deploy our web app.

One of the key reasons for selecting Hugging Face is its user-friendly interface and robust documentation, which simplifies the deployment workflow and reduces the learning curve for developers.

Furthermore, Hugging Face's scalability and reliability were critical factors in our decision-making process. The platform's infrastructure is designed to handle high volumes of traffic and deliver consistent performance, making it well-suited for deploying production-ready applications.

Moreover, Hugging Face offers built-in features for model versioning, monitoring, and management, which facilitate seamless updates and maintenance of deployed models.

Overall, Hugging Face emerged as the optimal choice for deployment based on its reputation for reliability, scalability, ease of use, and comprehensive feature set, enabling us to deploy our machine learning models with confidence and efficiency.

[https://huggingface.co/spaces/MoroccanDS/B7-Pyhton-based-currency-converter](https://huggingface.co/spaces/MoroccanDS/B7-Pyhton-based-currency-converter)

### Conclusion

In conclusion, embarking on this project to develop a Python-based currency converter web app has proven to be a highly rewarding endeavor, not only for the usefulness of the application but also for the invaluable skills gained throughout the process. By leveraging technologies such as Streamlit and BeautifulSoup, we've honed our abilities in web development, data extraction, and user interface design, equipping us with valuable tools for future projects.