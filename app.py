import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

#vectorise data 
loader = CSVLoader(file_path="legal_doc.csv")
documents = loader.load()
check=0
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

#similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query,k=1)
    page_contents_array=[doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")  
template="""
You are a a legal assistant which will provide legal documentation to small business and indiviuals
I will share a query for legal documentation and you will provide the best legal documentation you can make from the template legal document i will provide,
and you will follow ALL of the rules below:
1/ Response should be similar or identical to template in terms of length, tone of voice, logical argument and other details
2/ If the template provided is irrelevant, then try to mimic the style of the template

Below is the query for legal document i recieved:
{message}
here is the template legal documentation:
{doc}

Please write the best legal document which should be used for that case:
"""
prompt = PromptTemplate(
    input_variables=["message","doc"],
    template=template)

chain = LLMChain(llm=llm,prompt=prompt)
# retrival of ans
def generate_response(message):
    doc=retrieve_info(message)
    response=chain.run(message=message,doc=doc)
    return response



#print(response)

def main():
    st.set_page_config(page_title="Legal Documentation Agent",page_icon=":bird:")
    st.header("Legal Documentation Agent")
    message=st.text_area("enter legal documentation requirement")

    if message:
        st.write("generating document")
        result=generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()
