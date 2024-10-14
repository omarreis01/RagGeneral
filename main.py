#from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from tqdm.auto import tqdm
import gradio as gr 
import pandas as pd
import zipfile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
import boto3
from langchain_aws import BedrockEmbeddings, BedrockLLM, ChatBedrock
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re
import requests
from bs4 import BeautifulSoup
import json
from pydantic import BaseModel
from typing import Optional
import shutil
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from collections import deque
import multiprocessing


# Pydantic model for normal RAG output (no calculations)
#class RAGOutput(BaseModel):
#    scratchpad: Optional[str]
#    answer: Optional[str]
#    source: Optional[str]

# Pydantic model for calculated RAG output (with calculations)
#class CalculatedRAGOutput(RAGOutput):
#    calc_code: Optional[str]  # The code used for calculations
#    code_result: Optional[str]  # The result of executing the calculation


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

load_dotenv("base.env")

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-west-2')

bedrock_client = boto3.client("bedrock-runtime", 
                                region_name=aws_region, 
                                aws_access_key_id=aws_access_key_id, 
                                aws_secret_access_key=aws_secret_access_key)
#cohere.embed-multilingual-v3
#amazon.titan-embed-text-v2:0
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client) 

#model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_id = "meta.llama3-1-405b-instruct-v1:0" 

def get_embedding(text):
    # Use the BedrockEmbeddings model to generate the embedding
    embedding_result = bedrock_embeddings.embed_query(text)
    return embedding_result  # This should be a vector representing the text


class DataPoolLoader:
    def __init__(self, parent_folder: str, data_pool_folder: str):
        self.parent_folder = parent_folder
        self.data_pool_folder = data_pool_folder

    @property
    def pool_path(self):
        return os.path.abspath(self.data_pool_folder)

    def find_file_paths(self):
        try:
            file_paths = []
            for root, dirs, files in os.walk(self.parent_folder):
                for file in files:
                    # Skip hidden or macOS-specific files (starting with '._' or '.DS_Store')
                    if file.startswith('.') or file.startswith('._'):
                        continue
                    file_paths.append(os.path.join(root, file))
            return file_paths
        except Exception as e:
            raise RuntimeError(f"Error finding file paths: {str(e)}")

    def create_data_pool(self):
        try:
            os.makedirs(self.data_pool_folder, exist_ok=True)
            file_paths = self.find_file_paths()
            for file_path in file_paths:
                destination = os.path.join(self.data_pool_folder, os.path.basename(file_path))
                shutil.copy2(file_path, destination)
        except Exception as e:
            raise RuntimeError(f"Error creating data pool: {str(e)}")

    def load(self):
        try:
            self.create_data_pool()
            return f"Data pool created successfully!"
        except Exception as e:
            return f"Error loading data pool: {str(e)}"
        
# Initialize the target domain dynamically based on the first URL
TARGET_DOMAIN = None

# Global set to track visited URLs
seen_urls = set()
content_dict = set()

# Folder to store content
SAVE_FOLDER = 'scraped_content'

# Ensure the folder exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Generate a valid filename from URL
def generate_filename(url):
    parsed_url = urlparse(url)
    # Replace invalid characters in file names with underscores
    filename = re.sub(r'[\\/*?:"<>|]', '_', parsed_url.path.strip('/'))
    if not filename:
        filename = 'index'  # Handle root URLs (e.g., 'https://example.com')
    return f"{parsed_url.netloc}_{filename}.txt"

# Save progress to a file with the URL-based name
def save_progress(url, data):
    filename = generate_filename(url)
    file_path = os.path.join(SAVE_FOLDER, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{data}")

# Extract content only from <p> tags or specific sections
def extract_content_from_page(soup):

    #For web scraping you may add other tags
    elements = soup.find_all(['p','li' ,'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'aside', 'thead', 'tbody', 'tr', 'th', 'td'])
    extracted_content = ' '.join([element.get_text() for element in elements])
    return extracted_content

# Check if a URL is valid and belongs to the target domain
def is_valid_url(url):
    global TARGET_DOMAIN
    extensions = [".mp4",".pdf",".dmg",".jpg",".jpeg",".png"]
    for ext in extensions:
        if url.endswith(ext):
            return False
    parsed_url = urlparse(url)

    # Set the target domain dynamically based on the first URL processed
    if TARGET_DOMAIN is None:
        TARGET_DOMAIN = parsed_url.netloc

    return bool(parsed_url.scheme in ['http', 'https'] and parsed_url.netloc and TARGET_DOMAIN in parsed_url.netloc)

# Prevent duplicate extraction when the URL has a fragment (e.g., #ContentSection)
def is_duplicate_url(base_url, url_with_fragment):
    base = urlparse(base_url)
    fragment_url = urlparse(url_with_fragment)

    # Compare the base URLs (ignore fragments)
    return base.scheme == fragment_url.scheme and base.netloc == fragment_url.netloc and base.path == fragment_url.path

def extract_and_replace_url_bfs(start_url, max_depth = None):
    global seen_urls
    all_text = ""
    
    queue = deque([(start_url, 0)])

    while queue:
        url, current_depth = queue.popleft()
        
        if url in seen_urls:
            continue

        if not is_valid_url(url):
            continue

        try:
            response = requests.get(url)
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = extract_content_from_page(soup)
                combined_content = f"{main_content}"
                seen_urls.add(url)

                if main_content not in content_dict:
                    save_progress(url, main_content) 
                    content_dict.add(main_content)
                    all_text += main_content

                if max_depth is None or current_depth < max_depth:
                    sub_urls = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True) if is_valid_url(urljoin(url, link.get('href')))]
                    for sub_url in sub_urls:
                        if sub_url not in seen_urls and not is_duplicate_url(url, sub_url):
                            queue.append((sub_url, current_depth + 1))
            else:
                print(f"Failed to fetch content from {url}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from {url}: {e}")

    return all_text

def process_url_bfs(url, max_depth):
    return extract_and_replace_url_bfs(url, max_depth=max_depth)

def find_all_urls(text):
    # Regular expression to match URLs
    url_pattern = re.compile(r'https?://(?:www\.)?\S+(?:/|[^\s])')
    urls = re.findall(url_pattern, text)
    for url in urls:
        process_url_bfs(url)
    return urls


#2022 yili için A ürününün sabit modeldeki ortalama başari yüzdesi ne kadardir?
class DocumentProcessor:
    def __init__(self):
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.xml': UnstructuredXMLLoader, 
            '.csv': CSVLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader, # sheet metadata needed
            '.json': JSONLoader,  # Use for structured data
        }
        self.docs = None
        self.splits = []

    def reset_splits(self):
        self.splits = []
        return
    
    def _get_loader(self, file_path):
        try:
            extension = os.path.splitext(file_path)[1]
            return self.loaders.get(extension, TextLoader)(file_path)
        except Exception as e:
            raise ValueError(f"Error getting loader for {file_path}: {str(e)}")

    def load_single_document(self, file_path):
        try:
            # Get the file name
            file_name = os.path.basename(file_path)
            loader = self._get_loader(file_path)
            doc = loader.load()
            new_docs = []
            
            for page in doc:
                if file_name =="linkler.txt":
                    find_all_urls(page.page_content)
                else:
                    new_docs.append(page)
            return new_docs
            
        except Exception as e:
            raise RuntimeError(f"Error loading single document: {str(e)}")

    def split_single_document(self, document, chunk_size=4096, chunk_overlap=128):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            return text_splitter.split_documents(document)
        except Exception as e:
            raise RuntimeError(f"Error splitting single document: {str(e)}")

    def process_single_document(self, file_path):
        try:
            document = self.load_single_document(file_path)
            doc_splits = self.split_single_document(document)
            self.splits.extend(doc_splits)
        except Exception as e:
            raise RuntimeError(f"Error processing single document: {str(e)}")


# from langchain_huggingface import HuggingFaceEmbeddings

# bedrock_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", model_kwargs={'device': 'gpu'})

def extract_dict_from_response(llm_response):
    """
    Extracts the JSON object from a response string by matching the first
    occurrence of '{' and the last occurrence of '}'.
    """
    start_idx = llm_response.find('{')  # Find the first '{'
    end_idx = llm_response.rfind('}')   # Find the last '}'
    
    if start_idx == -1 or end_idx == -1:
        print("JSON FORMAT UYMUYOR 1")
        my_dict = {
        "scratchpad":"",
        "answer": "Bilmiyorum.",
        "is_calculations": False,
        "calc_code": ""
        }       
        my_dict_json = eval(my_dict)
        return my_dict_json

    # Extract the JSON string from the first '{' to the last '}'
    json_str = llm_response[start_idx:end_idx+1].strip()
    
    # Attempt to parse the JSON
    json_str = json_str.replace("true", "True").replace("false", "False").replace("null", "None")
    try:
        result_dict = eval(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        print("JSON FORMATA UYMUYOR 2")
        my_dict = {
        "scratchpad":"",
        "answer": "Bilmiyorum.",
        "is_calculations": False,
        "calc_code": ""
        }       
        
        my_dict_json = eval(my_dict)
        return my_dict_json
    
class FAISSDatabase:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def create_db(self, documents):
        self.db = FAISS.from_documents(documents, self.embeddings)
        return self.db

    def save_db(self, file_name="faiss_pool"):
        if self.db is None:
            raise ValueError("Database has not been created yet.")
        db_file = f"{file_name}"
        self.db.save_local(db_file)
        print(f"Database saved as {db_file}")

    def load_db(self, file_path):
        self.db = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)
        return self.db
    
    def update_db(self, new_documents):
        if self.db is None:
            raise ValueError("Database has not been loaded or created yet.")
        
        self.db.add_documents(new_documents)
        print("Database updated with new documents.")
        return self.db
    
class CreateFaissDatabase():
    def __init__(self):
        self.rag_chain = None
        self.data_loader = None
        self.retriever = None
        self.data_loader = None
        self.document_processor = DocumentProcessor()
        self.faiss_db = FAISSDatabase(bedrock_embeddings)
        self.uploaded_files_list = []  # List to store uploaded file paths
        self.uploaded_links_list = []
        self.uploaded_files_path = os.path.join(os.path.dirname(__file__), "uploaded_files.txt")  # Path to the txt file
        self.uploaded_links_path = os.path.join(os.path.dirname(__file__), "uploaded_links.txt")  

    
    def related_database(self,uploaded_files):
        try:
            # Define a dictionary to map file extensions to loaders
            file_loaders = {
                '.pdf': PyPDFLoader,
                '.xml': UnstructuredXMLLoader,
                '.csv': CSVLoader,
                '.txt': TextLoader,
                '.docx': Docx2txtLoader,
                '.xlsx': UnstructuredExcelLoader,
                '.json': JSONLoader
            }
            faiss_db_path = "faiss_pool.vdb"
            print("HERE")
            exists = False
            if os.path.exists(faiss_db_path):
                self.faiss_db.load_db(faiss_db_path)
                exists = True
                status_message = "Sistemdeki dökümanlar yüklendi."
            else:
                status_message = "Sistemde döküman bulunmadi. Yeni dökümanlarla beraber veri tabanı oluşturulacak."

            # Ensure the uploaded_files is a list
            if uploaded_files:
                print("uploaded files ==>",uploaded_files)
                for uploaded_file in uploaded_files:
                    file_path = uploaded_file  # Each file path in the list

                    if file_path.endswith('.zip'):
                        # Extract ZIP file to a temporary folder
                        extract_folder = "extracted_files"
                        data_pool_folder = "data_pool"  # Separate folder to store valid files

                        if not os.path.exists(extract_folder):
                            os.makedirs(extract_folder)

                        encoding = 'cp437'
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Decode filenames to handle special characters
                                decoded_name = member.encode(encoding).decode('utf-8', 'ignore')
                                target_path = os.path.join(extract_folder, decoded_name)

                                # Check if it's a folder
                                if member.endswith('/'):
                                    # Create directories if they don't exist
                                    os.makedirs(target_path, exist_ok=True)
                                else:
                                    # Ensure the directory structure is created before extracting files
                                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                                    # Extract the file to the target path
                                    with zip_ref.open(member) as source_file:
                                        with open(target_path, 'wb') as target_file:
                                            target_file.write(source_file.read())

                        # Now use DataPoolLoader to find all valid file paths and create the data pool
                        data_pool_folder = "data_pool"
                        data_loader = DataPoolLoader(parent_folder=extract_folder, data_pool_folder=data_pool_folder)
                        data_loader.load()

                            
                        for root, dirs, files in os.walk(data_pool_folder):
                            for file_name in files:
                                extracted_file_path = os.path.join(root, file_name)
                                
                                # Skip hidden files (starting with a dot) or macOS metadata files (e.g., '._' or '.DS_Store')
                                if file_name.startswith('.') or file_name.startswith('._'):
                                    continue
                                # Ensure it's a file (not a directory or invalid file)
                                if os.path.isfile(extracted_file_path):
                                    self.document_processor.process_single_document(extracted_file_path)
                                    self.uploaded_files_list.append(os.path.basename(extracted_file_path))


                    else:
                        self.document_processor.process_single_document(file_path)
                        self.uploaded_files_list.append(os.path.basename(file_path))
                # Specify the folder name
                folder_name = 'scraped_content'
        
                # Check if the folder exists and is a directory
                if os.path.exists(folder_name) and os.path.isdir(folder_name):
                    # Loop through all files in the folder
                    for file_name in os.listdir(folder_name):
                        # Get the full path to the file
                        extracted_file_path = os.path.join(folder_name, file_name)
                        # Ensure it's a text file and not a directory
                        if os.path.isfile(extracted_file_path) and file_name.endswith('.txt'):
                            # Send the file to the document processor
                            self.document_processor.process_single_document(extracted_file_path)
                            
                            modified_file_name = os.path.basename(extracted_file_path).replace('_', '/').rsplit('.', 1)[0]
                            # Append the modified file name to the uploaded files list
                            self.uploaded_links_list.append(modified_file_name)
                            
                    #print("All text files have been processed.")
                    shutil.rmtree(folder_name)
                    #print(f"The folder '{folder_name}' and its contents have been deleted.")
                else:
                    print(f"The folder '{folder_name}' does not exist.")


                # Step 3: If FAISS database does not exist, create it and save it
                if not exists:
                    self.faiss_db.create_db(self.document_processor.splits)  # Use the splits from document processing
                    self.faiss_db.save_db(faiss_db_path)
                    status_message += " Veri tabanı oluşturuldu ve dökümanlar yüklendi."
                else:
                    # Step 4: If FAISS database exists, update it with new documents
                    self.faiss_db.update_db(self.document_processor.splits)  # Update with new document chunks
                    self.faiss_db.save_db(faiss_db_path)
                    status_message += "Döküman başarıyla yüklendi."

                self._write_to_file()
                self.document_processor.reset_splits()
                self.uploaded_files_list = []

            llm = ChatBedrock(model_id=model_id, client=bedrock_client, model_kwargs={"temperature": 0.3})

            retriever = self.faiss_db.db.as_retriever(search_kwargs={"k":8})

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            template = """

                You are tasked with analyzing the following question based on the provided context. The questions are in Turkish, and you should provide the answer in Turkish. Your response must be in **JSON format** with the following fields, and all fields must exist without including any additional text, explanations, or comments:

                1. **scratchpad**: Think and reason step by step, you can output your thoughts into the stratchpad field.

                2. **answer**: This is the factual answer based on the information you find. If the answer cannot be determined or is unavailable, explicitly state: `"Bilmiyorum"`. You should never attempt to fabricate an answer when none exists.If is_calculations field is True, the answer field must only contain "Hesaplamalar yapılıyor!" as the response.
                3. **is_calculations**: This is a boolean value.
                - Set it to `True` if the question requires analytical operations, such as calculating maximum, minimum, average, summation, or identifying specific dates or patterns within a dataset, or operations involving rates, percentages, growth, or any form of numeric analysis (e.g., from a table or list).
                - Set it to `False` if no such calculations are required and the answer can be derived directly from the information given.

                4. **calc_code**:
                - If `is_calculations` is `False`, this field must be an **empty string** (`""`), and no code should be generated.
                - If `is_calculations` is `True`, provide Python code in this field, which performs the necessary calculation.

                The `calc_code` must follow these rules:

                1. **Function Definition**:
                - The code should define a function named `func` that accepts the required input data as parameters.
                - Clearly specify the input variables as arguments to the function to ensure that the input data is captured and used correctly in the calculation.

                2. **Calculation and Output**:
                - Perform the required calculation within the `func` function.
                - Store the result in a local variable named `result`.
                - Ensure that the function returns `result`.

                3. **Calling the Function**:
                - After defining the `func` function, call it using the appropriate input values and store its returned value in a variable named `result`.

                4. **Structure Example**:
                Here is a template for the code that must be followed:
                ```python
                def func(input1, input2, ...):  # Replace with appropriate input names
                    # Perform the calculation
                    result = ...  # Store the calculated result
                    return result

                result = func(input1, input2, ...)  # Call the function with the correct inputs
                5. **Important**:
                - You MUST write your code in triple quotes \"\"\", otherwise we will encounter an error.
            
                - You will get a tip of 200$ dollars, if you answer correctly.
                - **DO NOT** include any additional text, explanations, or comments outside the required JSON format.
                - Your response should follow **exactly** the JSON format and must **only** contain the JSON block without anything extra. 
                - Think and reason step by step
                - Never leave fields empty, always fill them unless calc_code is False
                Here is the expected format:
                <start>
                    {{
                    "scratchpad":"<your_thoughts_go_here_string>",
                    "answer": "<answer_string>", 
                    "is_calculations": <boolean>, 
                    "calc_code": "<code_string>" 
                    }}
                <end>
                
                Context: {context}
                Question:{question}

                <IMPORTANT_NOTES>
                1. ALWAYS *ONLY* OUTPUT JSON
                2. FOR CALCULATIONS OUTPUT *ONLY* RELEVANT VALUES.
                3. ALWAYS THINK STEP BY STEP, USE SCRATCHPAD FOR THAT PURPOSE.
                </IMPORTANT_NOTES>
            """
          
            custom_rag_prompt = PromptTemplate.from_template(template)

            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
            compressor = CrossEncoderReranker(model=model, top_n=8)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            rag_chain = (
                {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )
        
            self.retriever = retriever
            self.rag_chain = rag_chain
            return status_message
        
        except Exception as e:
            return f"Error creating FAISS database: {str(e)}"
    
    def _write_to_file(self):
        """Write the uploaded file name to the uploaded_files.txt."""
        with open(self.uploaded_files_path, 'a') as file:
            for file_name in self.uploaded_files_list:
                file.write(file_name + '\n')
        with open(self.uploaded_links_path, 'a') as f:
            for link_name in self.uploaded_links_list:
                f.write(link_name + "\n")
        
        
    def view_uploaded_documents(self):
        """Read and return the content of the uploaded_files.txt in a structured format."""
        if not os.path.exists(self.uploaded_files_path):
            return "Herhangi bir döküman yüklenmemiş."

        with open(self.uploaded_files_path, 'r', encoding='utf-8') as file:
            file_names = file.readlines()

        if not file_names:
            return "Herhangi bir döküman yüklenmemiş."
        
        # Return file names as a structured list (each file on a new line)
        return "\n".join([f"{index + 1}. {file_name.strip()}" for index, file_name in enumerate(file_names)])

    def answer_question(self,question): 

        docs_rt = self.retriever.invoke(question)
        print("STARTING THE QUESTION ==",question)
        #question_embedding = get_embedding(question)
        # Initialize a list to store the cosine similarities
        #similarity_scores = []

        # Iterate over the retrieved chunks
        #for doc in docs_rt:
            # Get the text of the chunk
        #    chunk_text = doc.page_content  # Assuming page_content contains the chunk text
        #    chunk_embedding = get_embedding(chunk_text)
            
            # Compute the cosine similarity between the question and the chunk
        #    similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
            
            # Store the similarity along with the chunk for sorting later
        #    similarity_scores.append((similarity, doc))
        # Sort the chunks by their cosine similarity scores in descending order
        #ranked_docs = sorted(similarity_scores, key=lambda x: x[0], reverse=True)

        # Extract the ranked list of sources
        #ranked_sources = [os.path.basename(doc.metadata["source"]) for _, doc in ranked_docs]
        # Sort the chunks by their cosine similarity scores in descending order

        # Example of how to use the ranked sources
        #for score, doc in ranked_docs:
        #    print(f"Source: {os.path.basename(doc.metadata['source'])}, Similarity: {score:.4f}")

        full_source_path = docs_rt[0].metadata["source"]
        source = os.path.basename(full_source_path) 
        #if source.startswith('www.'):
            #source = source.rsplit('.txt', 1)[0]
            #source = source.replace('_', '/')
        try:
            result = ""
            for chunk in self.rag_chain.stream(question):
                result += chunk
            result_json = extract_dict_from_response(result)
            scratchpad = result_json.get("scratchpad")
            answer = result_json.get("answer")
            is_calculations = result_json.get("is_calculations")
            if type(is_calculations) == str:
                if is_calculations.lower().strip() == "true":
                    is_calculations = True
                else:
                    is_calculations = False

            calc_code = result_json.get("calc_code","")
            code_result = None
            if is_calculations:
                local_vars = {}
                try:
                    exec(calc_code, {}, local_vars)
                    code_result = local_vars.get('result')
                except ZeroDivisionError:
                    return question,"","Bilmiyorum.",None,None,None
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return question,"","Bilmiyorum.",None,None,None
                print("Calculation result:", result)

            if scratchpad =="":
                source = None
            return question,scratchpad,answer,calc_code,code_result,source

        except json.JSONDecodeError:
            return "Error: Could not parse the result as JSON."
        except Exception as e:
            return f"Error(answer_question): {str(e)}"

    def answer_question_gradio(self, question):
        _, *ans = self.answer_question(question)
        return ans
            
    def process_excel(self, file):
        try:
            df = pd.read_excel(file.name)
            
            if 'Sorular' not in df.columns or 'Cevaplar' not in df.columns or 'Model_Düsüncesi' not in df.columns or 'Kod' not in df.columns or 'Kod_Sonucu' not in df.columns or 'Kaynak' not in df.columns:
                return "Excel dosyası bu kolonları içermeli: 'Model_Düsüncesi', 'Cevaplar', 'Kod', 'Kod_Sonucu', ve 'Kaynak' kolonları."
            
            sorular_list = df["Sorular"].to_list()
            sorular_list = [s for s in sorular_list if s is not None and s != ""]

            output_path = "excel_output.xlsx"

            results = []

            pool = ThreadPoolExecutor(multiprocessing.cpu_count()/2)
            #pool = ThreadPool(5)

            results = list(pool.map(self.answer_question, sorular_list))

            # Construct the new DataFrame after all tasks are completed
            new_df = {
                "Sorular": [],
                "Model_Düsüncesi": [],
                "Cevaplar": [],
                "Kod": [],
                "Kod_Sonucu": [],
                "Kaynak": []
            }

            for result_a in results:
                question, scratchpad, answer, calc_code, code_result, source = result_a
                new_df["Sorular"].append(question)
                new_df["Model_Düsüncesi"].append(scratchpad)
                new_df["Cevaplar"].append(answer)
                new_df["Kod"].append(calc_code)
                new_df["Kod_Sonucu"].append(code_result)
                new_df["Kaynak"].append(source)

            yeni_df = pd.DataFrame(new_df)
            yeni_df.to_excel(output_path, index=False)

            return f"Excel dosyası başarıyla yüklendi. {output_path}"
        
        except Exception as e:
            return f"Error processing Excel file: {str(e)}"



# Initialize CreateFaissDatabase with the data_loader
initialDB = CreateFaissDatabase()

# Call related_database with None if no specific file needs to be uploaded
initialDB.related_database(None)
WITH_TAB = True
if WITH_TAB:
    # Interface 1: Raw text question input and answer
    qa_interface = gr.Interface(
        fn=initialDB.answer_question_gradio,
        inputs=gr.Textbox(label="Soru"),
        outputs=[
            gr.Textbox(label = "Modelin Düşüncesi"),
            gr.Textbox(label = "Cevap"),
            gr.Textbox(label = "Kod"),
            gr.Textbox(label = "Kod sonucu"),
            gr.Textbox(label="Kaynak"),  # Use Markdown for clickable links
            #gr.Textbox(label="Sayfa numarası")
        ],
        title="Ziraat Assistant'a Sor"
    )

    # Interface 2: Excel file upload for question-answer processing
    excel_interface = gr.Interface(
        fn=initialDB.process_excel,
        inputs=gr.File(label="Excel Dosyasını Yükle"),
        outputs=gr.Textbox(label="Statü"),
        title="Excel Dosyasını yükle"
    )

    with gr.Blocks() as creating_initial_database:
        with gr.Row():
            gr.Markdown("Dil modelimiz: meta.llama3-1-405b-instruct-v1:0") 
        with gr.Row():
            file_input = gr.File(label="Dosya yükle ", file_count="multiple")
            upload_output = gr.Textbox(label="Statü", lines=8, interactive=False)
        with gr.Row():
            upload_button = gr.Button("Dosya yükle")
            view_files_button = gr.Button("Yüklü dosyaları gör")
            #reset_button = gr.Button("Reset Database")  # Reset button added here
        with gr.Row():
            file_list_output = gr.Textbox(label="Yüklü Dosyalar", lines=5, interactive=False)

        # Button click actions
        upload_button.click(initialDB.related_database, inputs=file_input, outputs=upload_output)  # For uploading files
        view_files_button.click(initialDB.view_uploaded_documents, outputs=file_list_output)  # For viewing uploaded files
        #reset_button.click(initialDB.reset_database, outputs=upload_output)  # For resetting the database

   # Combine all interfaces into a tabbed interface
    demo = gr.TabbedInterface(
        [creating_initial_database,qa_interface, excel_interface],
        [ "Dokumanlari Yükle","Soru sor", "Excel yükle"]
    )
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
demo.launch()






