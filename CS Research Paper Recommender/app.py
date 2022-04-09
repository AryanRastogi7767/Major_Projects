import streamlit as st
import numpy as np 
import pandas as pd
from time import sleep

st.cache(persist=True)
def load_data():
    data = pd.read_csv('data_with_recoms.csv')
    return data

st.set_page_config(page_title='CS Research Paper Recommender')
data = load_data()

st.title('CS Research Paper Recommender')
st.image('secondary-data-research.jpg')
st.write('In  practice,  research  paper  recommender  systems  do  not exist. However,  concepts  have  been  published  and  partly implemented  that  could  be used  for  their  realisation.\
          This project is an attempt to implement a Similarity Based Research Paper Recommender that leverages the power of practical text-processing and vectorization techniques to provide the \
              most similar recommendations to the user.')

# add key features of the project 
st.write('Key Challenges during the development of the project:')
st.write('1. Data Cleaning: Detailed Text Analysis and Regular Expressions were used to clean the data for this project.')
st.write('2. Computing Similarities: As the number of vectors and length of vectors were large, memory optimization was a major problem even with 12GB Google Colab systems.\
        It was achieved by iteratively processing each vector to find the recommendations and using non conventional math packages for performance boost.')
select_title = st.selectbox("Choose Paper: ",np.array(data['title']))
rec_title = st.button("Recommend Similar Papers")

if rec_title:
    title_index=data[data['title']==select_title].index.values[0]
    st.subheader('Selected Paper:')
    st.subheader(data['title'][title_index])
    st.write('Authors: ',data['authors'][title_index])
    st.write('Year: ',str(data['year'][title_index]))
    st.write("Abstract:")
    st.write(data['abstract'][title_index])
    st.write('Link to the Paper: ','https://arxiv.org/abs/'+str(data['id'][title_index]))
    st.write('Link to pdf file: ','https://arxiv.org/pdf/'+str(data['id'][title_index])+'.pdf')
    st.write('\n')
    st.subheader('Recommendations:')
    recoms = []
    for item in 'abcdefghijklmnopqrst':
        recoms.append(data[item][title_index])
    for i in recoms:
        st.subheader(data['title'][i])
        st.write('Authors: ',data['authors'][i])
        st.write('Year: ',str(data['year'][i]))
        st.write("Abstract:")
        st.write(data['abstract'][i])
        st.write('Link to the Paper: ','https://arxiv.org/abs/'+str(data['id'][i]))
        st.write('Link to pdf file: ','https://arxiv.org/pdf/'+str(data['id'][i])+'.pdf')