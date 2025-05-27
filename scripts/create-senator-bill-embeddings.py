import json
import os
import random
import re
import voyageai
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Voyage AI client
vo = voyageai.Client()

def clean_text(text):
    """Clean text by removing extra whitespace, tabs, and normalizing spaces."""
    if not text:
        return ""
    
    # Replace tabs with single spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single spaces
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with single newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def read_bill_document(bill_id):
    """Read and clean a bill document from the senate-bills directory."""
    bill_path = Path(f"../data/senate-bills/{bill_id}.txt")
    try:
        with open(bill_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            return clean_text(raw_text)
    except FileNotFoundError:
        logger.warning(f"Bill document not found: {bill_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading {bill_path}: {e}")
        return None

def sample_and_combine_documents(bill_ids, max_documents=50, max_chars=80000):
    """Sample from bill IDs and combine documents efficiently."""
    if not bill_ids:
        return ""
    
    # Sample up to max_documents bills
    sampled_bills = random.sample(bill_ids, min(len(bill_ids), max_documents))
    logger.info(f"Sampling {len(sampled_bills)} documents from {len(bill_ids)} available")
    
    # Read sampled documents
    documents = []
    for bill_id in sampled_bills:
        doc = read_bill_document(bill_id)
        if doc:
            documents.append(doc)
    
    if not documents:
        return ""
    
    # Calculate how much text to take from each document
    chars_per_doc = max_chars // len(documents)
    
    # Take equal portions from each document and combine
    combined_parts = []
    for doc in documents:
        if len(doc) <= chars_per_doc:
            combined_parts.append(doc)
        else:
            # Take from the beginning of the document
            combined_parts.append(doc[:chars_per_doc])
    
    combined_text = " ".join(combined_parts)
    
    # Final cleanup and truncation if needed
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
    
    logger.info(f"Combined {len(documents)} documents into {len(combined_text)} characters")
    return combined_text

def create_senator_embeddings(senator_name, votes_data):
    """Create embeddings for a senator's Yea and Nay votes."""
    logger.info(f"Processing senator: {senator_name}")
    
    yea_votes = votes_data.get("Yea", [])
    nay_votes = votes_data.get("Nay", [])
    
    logger.info(f"Senator {senator_name} has {len(yea_votes)} Yea votes and {len(nay_votes)} Nay votes")
    
    # Sample and combine documents for each vote type
    yea_text = sample_and_combine_documents(yea_votes, max_documents=50, max_chars=80000)
    nay_text = sample_and_combine_documents(nay_votes, max_documents=50, max_chars=80000)
    
    logger.info(f"Final text lengths - Yea: {len(yea_text)} chars, Nay: {len(nay_text)} chars")
    
    # Create embeddings
    texts_to_embed = []
    labels = []
    
    if yea_text:
        texts_to_embed.append(yea_text)
        labels.append("yea")
    
    if nay_text:
        texts_to_embed.append(nay_text)
        labels.append("nay")
    
    if not texts_to_embed:
        logger.warning(f"No text found for {senator_name}")
        return None
    
    try:
        # Get embeddings from Voyage AI
        result = vo.embed(texts_to_embed, model="voyage-3-large", truncation=True)
        embeddings = result.embeddings
        
        # Structure the result
        embedding_data = {}
        for i, label in enumerate(labels):
            embedding_data[label] = {
                "embedding": embeddings[i],
                "original_document_count": len(yea_votes) if label == "yea" else len(nay_votes),
                "sampled_document_count": min(len(yea_votes), 50) if label == "yea" else min(len(nay_votes), 50),
                "text_length": len(yea_text) if label == "yea" else len(nay_text)
            }
        
        return embedding_data
        
    except Exception as e:
        logger.error(f"Error creating embeddings for {senator_name}: {e}")
        return None

def main():
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Read senator votes data
    votes_file = Path("../data/senator-votes/senator_votes.json")
    
    try:
        with open(votes_file, 'r', encoding='utf-8') as f:
            senator_votes = json.load(f)
        logger.info(f"Successfully loaded senator votes from: {votes_file}")
    except FileNotFoundError:
        logger.error(f"Senator votes file not found: {votes_file}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Looking for file at absolute path: {votes_file.absolute()}")
        return
    except Exception as e:
        logger.error(f"Error reading senator votes file: {e}")
        return
    
    # Create output directory
    output_dir = Path("../data/senator-bill-embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each senator
    total_senators = len(senator_votes)
    processed_count = 0
    skipped_count = 0
    
    for i, (senator_name, votes_data) in enumerate(senator_votes.items(), 1):
        output_file = output_dir / f"{senator_name}.json"
        
        # Skip if embeddings already exist
        if output_file.exists():
            logger.info(f"Skipping {i}/{total_senators}: {senator_name} (embeddings already exist)")
            skipped_count += 1
            continue
            
        logger.info(f"Processing {i}/{total_senators}: {senator_name}")
        
        # Create embeddings for this senator
        embeddings = create_senator_embeddings(senator_name, votes_data)
        
        if embeddings:
            # Save embeddings to file
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings, f, indent=2)
                logger.info(f"Saved embeddings for {senator_name} to {output_file}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error saving embeddings for {senator_name}: {e}")
        else:
            logger.warning(f"No embeddings created for {senator_name}")
    
    logger.info(f"Processing complete! Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_senators}")

if __name__ == "__main__":
    main()