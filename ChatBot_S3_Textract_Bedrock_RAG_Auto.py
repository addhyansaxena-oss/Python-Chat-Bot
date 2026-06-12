import streamlit as st
import boto3
import uuid
import os
import json
import time
import io
from botocore.exceptions import ClientError, BotoCoreError

# ===== AWS Configuration =====
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
S3_BUCKET = st.secrets["S3_BUCKET"]
S3_PREFIX = st.secrets.get("S3_PREFIX", "bedrock-ingestion/")
KB_ID = st.secrets["KB_ID"]
MODEL_ARN = st.secrets["MODEL_ARN"]
DATA_SOURCE_ID = st.secrets["DATA_SOURCE_ID"]
MANIFEST_KEY = os.path.join(S3_PREFIX, "manifest.jsonl")
ALLOWED_EXTENSIONS = ['.pdf', '.jpeg', '.jpg', '.png', '.tiff', '.tif']

@st.cache_resource
def get_aws_clients():
    return {
        'textract': boto3.client('textract', region_name=AWS_REGION,
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        'bedrock-agent': boto3.client('bedrock-agent', region_name=AWS_REGION,
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        'bedrock-agent-runtime': boto3.client('bedrock-agent-runtime', region_name=AWS_REGION,
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        's3': boto3.client('s3', region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    }

clients = get_aws_clients()

def s3_key_for_doc(filename):
    return os.path.join(S3_PREFIX, filename)

def s3_key_for_txt(filename):
    base, _ = os.path.splitext(filename)
    return os.path.join(S3_PREFIX, base + '.txt')

def download_manifest():
    try:
        obj = clients['s3'].get_object(Bucket=S3_BUCKET, Key=MANIFEST_KEY)
        lines = obj['Body'].read().decode('utf-8').splitlines()
        return [json.loads(line) for line in lines]
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            return []
        else:
            st.error(f"Error downloading manifest: {e}")
            return []
    except Exception as e:
        st.error(f"Unexpected error downloading manifest: {e}")
        return []

def upload_manifest(manifest):
    try:
        body = "\n".join(json.dumps(entry) for entry in manifest)
        clients['s3'].put_object(Bucket=S3_BUCKET, Key=MANIFEST_KEY, Body=body.encode('utf-8'))
        st.info(f"Manifest uploaded to s3://{S3_BUCKET}/{MANIFEST_KEY}")
    except Exception as e:
        st.error(f"Error uploading manifest: {e}")

def is_in_manifest(filename, manifest):
    return any(entry.get("filename") == filename for entry in manifest)

def add_to_manifest(filename, txt_s3_uri, manifest):
    manifest.append({"filename": filename, "txt_s3_uri": txt_s3_uri, "ingested": True})
    upload_manifest(manifest)

def upload_to_s3(file, bucket, key):
    try:
        clients['s3'].upload_fileobj(file, bucket, key)
        st.info(f"Uploaded file to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"
    except Exception as e:
        st.error(f"Error uploading {key} to S3: {e}")
        return None

def save_txt_to_s3(text, bucket, key):
    try:
        clients['s3'].put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
        st.info(f"Saved .txt to s3://{bucket}/{key}")
        time.sleep(2)  # S3 consistency wait
        return f"s3://{bucket}/{key}"
    except Exception as e:
        st.error(f"Error saving text to S3: {e}")
        return None

# === UPDATED FUNCTION: Handles both images and PDFs ===
def process_doc_with_textract(file_bytes, ext, filename):
    try:
        if ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tif']:
            # For images, use the synchronous API
            response = clients['textract'].analyze_document(
                Document={'Bytes': file_bytes},
                FeatureTypes=['FORMS', 'TABLES']
            )
            return '\n'.join(block['Text'] for block in response['Blocks'] if block['BlockType'] in ['LINE', 'WORD'])

        elif ext == '.pdf':
            # For PDFs, upload to S3 and use the async API
            temp_pdf_key = f"textract-temp/{uuid.uuid4()}-{filename}"
            clients['s3'].put_object(Bucket=S3_BUCKET, Key=temp_pdf_key, Body=file_bytes)
            s3_obj = {'S3Object': {'Bucket': S3_BUCKET, 'Name': temp_pdf_key}}

            start_response = clients['textract'].start_document_analysis(
                DocumentLocation=s3_obj,
                FeatureTypes=['FORMS', 'TABLES']
            )
            job_id = start_response['JobId']

            # Wait for job to complete (polling)
            while True:
                job_status = clients['textract'].get_document_analysis(JobId=job_id)
                status = job_status['JobStatus']
                if status in ['SUCCEEDED', 'FAILED']:
                    break
                time.sleep(5)
            if status == 'FAILED':
                st.error(f"Textract PDF analysis failed: {job_status.get('StatusMessage', '')}")
                return None

            # Gather results (pagination)
            blocks = []
            next_token = None
            while True:
                if next_token:
                    job_status = clients['textract'].get_document_analysis(JobId=job_id, NextToken=next_token)
                blocks.extend(job_status['Blocks'])
                next_token = job_status.get('NextToken')
                if not next_token:
                    break
            return '\n'.join(block['Text'] for block in blocks if block['BlockType'] in ['LINE', 'WORD'])

        else:
            st.error("Unsupported file type for Textract.")
            return None

    except (BotoCoreError, ClientError) as e:
        st.error(f"Textract error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during Textract OCR: {e}")
        return None

def start_bedrock_kb_ingestion():
    try:
        response = clients['bedrock-agent'].start_ingestion_job(
            knowledgeBaseId=KB_ID,
            dataSourceId=DATA_SOURCE_ID
        )
        job_id = response['ingestionJob']['ingestionJobId']
        st.info(f"Started ingestion job with ID: {job_id}")
        return job_id
    except Exception as e:
        st.error(f"Error starting Bedrock KB ingestion: {e}")
        return None

def wait_for_bedrock_ingestion(job_id, timeout=600):
    try:
        start = time.time()
        while True:
            resp = clients['bedrock-agent'].get_ingestion_job(
                knowledgeBaseId=KB_ID,
                dataSourceId=DATA_SOURCE_ID,
                ingestionJobId=job_id
            )
            status = resp['ingestionJob']['status']
            st.info(f"Ingestion job {job_id} status: {status}")
            if status in ("COMPLETED", "COMPLETE"):
                return True
            if status in ("FAILED", "STOPPED"):
                st.error(f"Ingestion job {job_id} failed or stopped.")
                st.error(f"Details: {resp['ingestionJob']}")
                return False
            if time.time() - start > timeout:
                st.error(f"Ingestion job {job_id} timed out.")
                return False
            time.sleep(5)
    except Exception as e:
        st.error(f"Error checking ingestion job status: {e}")
        return False

def get_rag_response(query, session_id=None):
    prompt_template = (
        "You are an oncology specialist assistant.\n"
        "Use only trusted Oncology resources like papers, journals, published articles, and newsletters from the internet, along with the information in the knowledge base, to answer the question.\n\n"
        "Context:\n{search_results}\n$search_results$\n\n"
        "Question: " + query + "\n\n"
        "Answer with clinical accuracy. If uncertain, state \"I need to consult medical records.\""
    )
    st.info(f"Prompt template being sent to Bedrock:\n{prompt_template}")
    try:
        kwargs = {
            "input": {'text': query},
            "retrieveAndGenerateConfiguration": {
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KB_ID,
                    'modelArn': MODEL_ARN,
                    'generationConfiguration': {
                        'promptTemplate': {'textPromptTemplate': prompt_template}
                    }
                }
            }
        }
        if session_id:
            kwargs["sessionId"] = session_id

        response = clients['bedrock-agent-runtime'].retrieve_and_generate(**kwargs)
        # Always update session_id with what Bedrock returns
        return response['output']['text'], response.get('sessionId')
    except Exception as e:
        st.error(f"Clinical error: {str(e)}")
        # If session error, clear session_id so a new one is started next time
        if "Session with Id" in str(e) and "is not valid" in str(e):
            return "Session expired or invalid. Please try again.", None
        return "Please consult your physician for immediate concerns.", session_id

# ===== Streamlit UI =====
st.title("Clinical Oncology Virtual Assistant 🩺")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

with st.expander("Upload Patient Records (PDF, JPEG, PNG, TIFF)"):
    uploaded_files = st.file_uploader(
        "Upload medical documents (PDF, JPEG, PNG, TIFF)",
        type=['pdf', 'jpeg', 'jpg', 'png', 'tiff', 'tif'],
        accept_multiple_files=True
    )

    if uploaded_files:
        manifest = download_manifest()
        for file in uploaded_files:
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                st.error(f"{file.name}: Unsupported file type.")
                continue

            if is_in_manifest(file.name, manifest):
                st.success(f"{file.name} is already present in Bedrock Knowledge Base.")
                continue

            file_bytes = file.read()
            # Pre-validation
            if len(file_bytes) == 0:
                st.error(f"{file.name}: Uploaded file is empty.")
                continue
            if ext == '.pdf' and len(file_bytes) > 500 * 1024 * 1024:
                st.error(f"{file.name}: PDF is larger than 500MB. Please upload a smaller file.")
                continue
            if ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tif'] and len(file_bytes) > 500 * 1024 * 1024:
                st.error(f"{file.name}: Image file is larger than 500MB. Please upload a smaller image.")
                continue

            doc_key = s3_key_for_doc(file.name)
            txt_key = s3_key_for_txt(file.name)
            txt_s3_uri = f"s3://{S3_BUCKET}/{txt_key}"

            # (Optionally upload original file to S3 for record-keeping)
            # s3_doc_uri = upload_to_s3(io.BytesIO(file_bytes), S3_BUCKET, doc_key)

            # Run Textract OCR
            with st.spinner(f"Running Textract OCR on {file.name}..."):
                text = process_doc_with_textract(file_bytes, ext, file.name)
            if not text:
                st.error(f"OCR failed for {file.name}. Skipping.")
                continue

            # Save extracted text to S3
            s3_txt_uri = save_txt_to_s3(text, S3_BUCKET, txt_key)
            if not s3_txt_uri:
                continue
            st.success(f"Extracted text saved as {os.path.basename(txt_key)} in S3.")

            # Wait for S3 consistency
            time.sleep(2)

            # Trigger Bedrock KB ingestion
            st.info(f"Triggering ingestion for {s3_txt_uri}")
            with st.spinner(f"Ingesting {os.path.basename(txt_key)} into Bedrock Knowledge Base..."):
                job_id = start_bedrock_kb_ingestion()
                if not job_id:
                    continue
                success = wait_for_bedrock_ingestion(job_id)
                if success:
                    add_to_manifest(file.name, txt_s3_uri, manifest)
                    st.success(f"{file.name} successfully ingested into Bedrock Knowledge Base.")
                else:
                    st.error(f"Ingestion failed or timed out for {file.name}.")
                    st.info("Check Bedrock Console → Knowledge Base → Data Source → Sync History for details.")

st.divider()

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about patient records..."):
    st.chat_message("user").write(user_input)
    with st.spinner("Consulting medical knowledge..."):
        response, new_session_id = get_rag_response(user_input, st.session_state.session_id)
        # If Bedrock returns a new session_id, store it
        if new_session_id:
            st.session_state.session_id = new_session_id
        else:
            # If session invalid, clear it so next query starts a new session
            st.session_state.session_id = None

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])
