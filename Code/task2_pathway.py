# Classify test data of the publishable papers identified as publishable in the first task
# Test + Infer the model on the test data

# python3 -m http.server 8668
# ngrok http 8668
# ngrok config add-authtoken 2rY2SvWOeUOE3HS7r2QT2Utj94h_7eXhxDL6W2oZPxjqWf8eE
# run the above line on the terminal before running this code

# killall ngrok
# run the above line on the terminal after running this code

import pathway as pw
import pandas as pd
import tempfile
import os
import json
from PyPDF2 import PdfReader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.vectorstores import PathwayVectorClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch

# Step 1: Google Drive Integration
def extract_pdf_text(binary_data):
    """Extract text from PDF binary data."""
    try:
        pdf_file = io.BytesIO(binary_data)
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

class PDFProcessor:
    def __init__(self, credentials_dict):
        self.credentials_dict = credentials_dict

    def process_pdfs(self, folder_id):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(self.credentials_dict, temp_file)
            temp_credentials_path = temp_file.name

            try:
                input_table = pw.io.gdrive.read(
                    folder_id, service_user_credentials_file=temp_credentials_path
                )
                processed_table = input_table.select(
                    file_name=input_table.id,
                    content=pw.apply(extract_pdf_text, input_table.data),
                )
                pw.io.csv.write(processed_table, "extracted_texts.csv")
            finally:
                if os.path.exists(temp_credentials_path):
                    os.remove(temp_credentials_path)

# Step 2: Pathway Vector Store Integration
def store_embeddings(file_path, vector_store_data):
    df = pd.read_csv(file_path)
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    embeddings = SentenceTransformerEmbeddings(
        model_name="allenai/scibert_scivocab_uncased"
    )

    for _, row in df.iterrows():
        chunks = splitter.split_text(row['content'])
        for chunk in chunks:
            embedding = embeddings.embed(chunk)
            vector_store_data.write({"text": chunk, "label": "unknown", "embedding": embedding})

# Step 3: Model Training
class ConferenceDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])

def train_model(vector_store_client, label_map):
    embeddings, labels = [], []
    for record in vector_store_client.all_records():
        embeddings.append(record['embedding'])
        labels.append(label_map.get(record['label'], -1))  # Default label as -1 for "unknown"

    train_embs, test_embs, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    train_dataset = ConferenceDataset(train_embs, train_labels)
    test_dataset = ConferenceDataset(test_embs, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        'allenai/scibert_scivocab_uncased', num_labels=len(label_map)
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# Step 4: Test Data Classification
def classify_test_data(test_file_path, vector_store_client, output_file_path):
    df = pd.read_csv(test_file_path)
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    classified_data = []

    for _, row in df.iterrows():
        chunks = splitter.split_text(row['content'])
        for chunk in chunks:
            result = vector_store_client.similarity_search(chunk)
            if result:
                classified_data.append(result[0]['text'])  # Adjust based on Pathway output
            else:
                classified_data.append("Unclassified")

    df['classified_conference'] = classified_data
    df.to_csv(output_file_path, index=False)

# Main Workflow
def main():
    # Google Drive configuration
    creds_dict = {  # Fill in your service account credentials
        "type": "service_account",
        # Add all required fields from your JSON
    }
    folder_id = "your-google-drive-folder-id"  # Replace with your folder ID

    processor = PDFProcessor(creds_dict)
    processor.process_pdfs(folder_id)

    # Initialize Pathway Vector Store
    pw.set_license_key("your-pathway-license-key")
    host, port = "127.0.0.1", 8667
    vector_store_data = pw.io.fs.read(
        "./data", format="binary", mode="streaming", with_metadata=True
    )
    server = VectorStoreServer.from_langchain_components(
        vector_store_data,
        embedder=SentenceTransformerEmbeddings(model_name="allenai/scibert_scivocab_uncased"),
        splitter=CharacterTextSplitter(chunk_size=512, chunk_overlap=0),
    )
    server.run_server(host, port)

    # Store embeddings
    store_embeddings("extracted_texts.csv", vector_store_data)

    # Train model
    label_map = {
        "KDD": 0, "NeurIPS": 1, "CVPR": 2, "EMNLP": 3, "TMLR": 4
    }
    vector_store_client = PathwayVectorClient(url=f"http://{host}:{port}")
    train_model(vector_store_client, label_map)

    classify_test_data("publishable_papers.csv", vector_store_client, "classified_test_data.csv")

if __name__ == "__main__":
    main()
