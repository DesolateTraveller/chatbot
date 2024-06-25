#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="Chatbot",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[ChatBot | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
### LLM Hyperparameters
#---------------------------------------------------------------------------------------------------------------------------------

#stats_expander = st.sidebar.expander("**:blue[LLM HyperParameters]**", expanded=False)
#with stats_expander: 
with st.sidebar.popover("**:blue[:pushpin: LLM HyperParameters]**", help="Tune the hyperparameters whenever required"):    
    llm_model = st.selectbox("**Select LLM**", ["anthropic.claude-v2:1","amazon.titan-text-express-v1","ai21.j2-ultra-v1","anthropic.claude-3-sonnet-20240229-v1:0"])
    max_tokens = st.number_input("**Max Tokens**", value=250)
    temperature= st.number_input(label="**Temperature (randomness)**",step=.1,format="%.2f", value=0.7)
    top_p= st.number_input(label="**top_p (cumulative probability)**",step=.01,format="%.2f", value=0.9)
    top_k= st.number_input(label="**top_k (top k most probable tokens)**",step=10, value=250)                                  
    chunk_size= st.number_input(label="**chunk_size (managable segments)**",step=100, value=10000) 
    chunk_overlap= st.number_input(label="**chunk_overlap (overlap between chunks)**",step=100, value=1000) 

with st.sidebar.popover("**:blue[:blue_book: Definition of LLM HyperParameters]**"):                
    st.info('''
                    
            - **LLM**           - 'Large language Model (LLM)' used for analysis.  
            - **Max Tokens**    - the maximum number of tokens that the model can process at once, the maximum length of the prompt and the output of the model.
            - **Temparature**   - a parameter that controls the randomness and creativity of a large language model's (LLM) responses. 
            - **top_p**         - it sets a threshold such that only the words with probabilities greater than or equal to the threshold will be included (sets a cumulative probability threshold).
            - **top_k**         - it is used to limit the number of choices for the next predicted word or token.         
            ''')
    
st.sidebar.divider()