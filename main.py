import streamlit as st
import pandas as pd
import plotly as pl
import pickle 
import requests
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from faker import Faker
import matplotlib.pyplot as plt
import nltk
import re
from langdetect import detect
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def models(x):
      with open("countVectorizer.pkl", "rb") as cvf:
          count_vectorizer = pickle.load(cvf)
          x_new_count = count_vectorizer.transform(x)
          # Convert the sparse matrix to a dense numpy array
          x_new_count_dense = x_new_count.toarray()

          # Load the scaler and transform the dense data
      with open("scaler.pkl", "rb") as sc:
              scaler = pickle.load(sc)
              x_new_scaled = scaler.transform(x_new_count_dense)
     # Load the trained Random Forest model
      with open("rn.pkl", "rb") as fr:
          model_rf = pickle.load(fr)
          # Predict using the trained model
          pred_new = model_rf.predict(x_new_scaled)   
      df_reviews["fed"]=pred_new


def amazon_data(url):
        stemmer = PorterStemmer()
        STOPWORDS = set(stopwords.words('english'))
        reviewlist = []

        def stemming(data):
            review = str(data)
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.lower()
            review= re.sub(r'\b(?:heat|bad service|bad customer|bad delivery|bakavas|sad|not|vibrate|hang|overheating|heating|bad|worst|lag|problem|missing|spam)\b', 'poor', review)
            review=review.split()
            #review = [stemmer.stem(word) for word in review]
            review = [stemmer.stem(word) for word in review if not word in STOPWORDS]

            return ' '.join(review)


        def extractReviews(reviewUrl, pageNumber, fake):
                fake_ipv4 = fake.ipv4()
                print("Fake IPv4 Address:", fake_ipv4)

                headers = {
                    'authority': 'www.amazon.in',
                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'accept-language': 'en-US,en;q=0.9',
                    'cache-control': 'max-age=0',
                    # Requests sorts cookies= alphabetically
                    # 'cookie': 'session-id=259-3113978-6678618; i18n-prefs=INR; ubid-acbin=260-8554202-6973909; lc-acbin=en_IN; csm-hit=tb:BS866TA0AKH6X86N924E+sa-7XYTQAXQHJP5ADH88228-DY27HYE0CK5V9FW24GBD|1656009294944&t:1656009294945&adb:adblk_yes; session-token=Z1j175VoYxPr2Un/9ciL3Q6lKw+QtLYYIwSQ+GLxjT06952u8vOZromD4WcFE0bs+yrUyLPy8HmIn7mTjUt8qsx3n0meC7yWKFqqwDEm5iecYedklsrNwmDrQOiaMH9lpacbdB8kgUk5IbZdg1VyhrdnY4OZrk6r350ARDEXJExuu2GZr0sV4fpbwUes/V9fDrfASeMQhVEEzmEAAHWN2g==; session-id-time=2082758401l',
                    'device-memory': '8',
                    'downlink': '10',
                    'dpr': '0.8',
                    'ect': '4g',
                    'referer': 'https://www.amazon.in/OnePlus-Nord-Black-128GB-Storage/dp/B09WQY65HN/ref=sr_1_4?crid=1D99WHM86WX80&keywords=oneplus&qid=1656009113&sprefix=onep%2Caps%2C315&sr=8-4&th=1',
                    'rtt': '0',
                    'sec-ch-device-memory': '8',
                    'sec-ch-dpr': '0.8',
                    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-ch-viewport-width': '2400',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'same-origin',
                    'sec-fetch-user': '?1',
                    'service-worker-navigation-preload': 'true',
                    'upgrade-insecure-requests': '1',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                    'viewport-width': '2400',
                }

                resp = requests.get(reviewUrl, headers=headers, proxies={"http": f"http://{fake_ipv4}"})
                soup = BeautifulSoup(resp.text, 'html.parser')
                reviews = soup.findAll('div', {'data-hook': "review"})
                for item in reviews:
                    review = {
                        'Review Title': item.find('a', {'data-hook': "review-title"}).text.strip(),
                        'Rating': item.find('i', {'data-hook': 'review-star-rating'}).text.strip(),
                        'Review Body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
                    }
                    print(review)
                    reviewlist.append(review)

        def pagenum(reviewUrl):
                fake_ipv4 = fake.ipv4()
                print("Fake IPv4 Address:", fake_ipv4)

                headers = {
                    'authority': 'www.amazon.in',
                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'accept-language': 'en-US,en;q=0.9',
                    'cache-control': 'max-age=0',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                }
                resp = requests.get(reviewUrl, headers=headers, proxies={"http": f"http://{fake_ipv4}"})
                soup = BeautifulSoup(resp.text, 'html.parser')
                reviews = soup.find('div', {'data-hook': "cr-filter-info-review-rating-count"})
                num_reviews_text = reviews.text.strip().split(", ")
                num = num_reviews_text[1].split(" ")
                out = int(num[0].replace(',', ''))
                return out

        # Main program

        fake = Faker()

        productUrl = url
        reviewUrl = productUrl.replace("dp", "product-reviews") + "?pageNumber=" + str(1)
        totalPg = pagenum(reviewUrl)
        print("Total reviews:", totalPg)

        if totalPg >= 100:
            totalPg = 10
        else:
            totalPg = 3

        # Use fake IP for scraping
        for i in range(1, totalPg + 1):
            print(f"Running page {i}")
            reviewUrl = productUrl.replace("dp", "product-reviews") + f"/ref=cm_cr_getr_d_paging_btm_{i}?ie=UTF8&pageNumber={i}&reviewerType=all_reviews&pageSize=10"
            #print(reviewUrl)
            extractReviews(reviewUrl, i, fake)

        # critical
        critical_url = productUrl.replace("dp", "product-reviews") + "/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar=critical&reviewerType=all_reviews&pageNumber=1"
        criticalPg = pagenum(critical_url)
        c=criticalPg
        print("Critical reviews:", criticalPg)


        # positive
        pos_url = productUrl.replace("dp", "product-reviews") + "/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar=positive&reviewerType=all_reviews&pageNumber=1"
        posPg = pagenum(pos_url)
        print("Positive reviews:", posPg)
        p=posPg
                    
        
                
        df_reviews = pd.DataFrame(reviewlist)
        df_reviews["test"]=df_reviews["Review Body"].apply(stemming)
        x=df_reviews["test"].values
        return x,df_reviews,c,p

if __name__=="__main__":
    # Set page configuration
  st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon=":bar_chart:",
        initial_sidebar_state="expanded",
    )
  st.title("Sentimental Analysis of Amazon Product Reviews")
  st.write("By Random Forest Algorithm")
  st.write("----------------------------------------------------")
  st.subheader("Enter Amazon Url")
  url=st.text_input("",placeholder="enter url")
  if st.button("Enter"):
    with st.spinner("Fetching data..."):
            try:
                x, df_reviews, c, p = amazon_data(url)
                models(x)
                #st.write(df_reviews)
                # Count positive and negative sentiment reviews
                sentiment_counts = df_reviews['fed'].value_counts()
                # Plot horizontal bar graph for sentiment analysis results
                st.write("-------------------------------------------------------")
                st.subheader("Sentiment Analysis Results")
                
                # Create the horizontal bar graph
                fig_sentiment = go.Figure(go.Bar(
                    y=sentiment_counts.index.map({1: 'Positive', 0: 'Negative'}),
                    x=sentiment_counts.values,
                    text=sentiment_counts.values,
                    textposition='auto',
                    marker_color=['green', 'red'] , # Colors for positive and negative bars
                    orientation='h'
                ))

                fig_sentiment.update_layout(
                    xaxis_title='Number of Reviews',
                    yaxis_title='Sentiment',
                    barmode='group',  # Display bars side by side
                    
                )


                # Display the chart in Streamlit
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                st.write("------------------------------------------------")
                st.subheader("Total Positive and Critical Reviews")
                # Display donut chart for positive and critical reviews
                labels = ['Critical', 'Positive']
                values = [c, p]
                colors = ['red', 'green']  # Define colors for each segment

                # Use `hole` to create a donut-like pie chart
                fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker=dict(colors=colors))])

                # Display the chart in Streamlit
                st.plotly_chart(fig_donut, use_container_width=True)

                
                # Display specific messages and emojis based on sentiment analysis results
                if sentiment_counts.get(1, 0) > sentiment_counts.get(0, 0):  # Positive sentiment is higher
                    st.info("👍 You can buy this product!")
                    st.balloons()
                else:  # Negative sentiment is higher or equal
                    st.info("👎 You should not buy this product.")
                st.write("---------------------------------------------------")
                # Display positive and negative reviews in columns
                col1, col2 = st.columns([1,1])
                
                with col1:
                    st.subheader("Positive Reviews")
                    positive_reviews = df_reviews[df_reviews['fed'] == 1]['Review Body']
                    for review in positive_reviews:
                        st.success(review)

                with col2:
                    st.subheader("Negative Reviews")
                    negative_reviews = df_reviews[df_reviews['fed'] == 0]['Review Body']
                    for review in negative_reviews:
                        st.warning(review)    

            except Exception as e:
                st.warning("Sorry Not able to fetch data! 😭")
               # st.write(e)
            
