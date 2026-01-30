import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
import shutil
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VDB_Creator")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def create_continental_vdb():
    # Use current working directory
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    logger.info(f"Base Directory: {base_dir}")
    logger.info(f"Data Directory: {data_dir}")

    # --- Step 0: Backup ---
    vdb_path = os.path.join(data_dir, "continental_embeddings.pt")
    if os.path.exists(vdb_path):
        backup_path = vdb_path + ".bak"
        logger.info(f"Found existing VDB. Backing up to {backup_path}")
        try:
            shutil.copy2(vdb_path, backup_path)
            logger.info("Backup complete.")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    # --- Step 1: Loading source datasets ---
    logger.info("Loading source datasets...")
    csv1_name = "FINAL_ACCURATE_FOOD_DATASET_WITH_CUISINE (1).csv"
    csv2_name = "food_nutrition_with_serving_category (1).csv"
    
    csv1_path = os.path.join(data_dir, csv1_name)
    csv2_path = os.path.join(data_dir, csv2_name)
    
    if not os.path.exists(csv1_path):
        logger.error(f"Missing {csv1_name} at {csv1_path}")
        return
    if not os.path.exists(csv2_path):
        logger.error(f"Missing {csv2_name} at {csv2_path}")
        return

    # Dataset 1: Explicitly filtered
    df1 = pd.read_csv(csv1_path)
    cont1 = df1[df1['Cuisine'] == 'Continental'].copy()
    
    # Dataset 2: Hand-picked continental items
    df2 = pd.read_csv(csv2_path)
    cont_keywords = ['Steak', 'Pizza', 'Burger', 'Pasta', 'Salad', 'Lasagna', 'Sandwich', 'Pancake', 'Omelette', 'Continental']
    cont2 = df2[df2['food name'].str.contains('|'.join(cont_keywords), case=False, na=False)].copy()
    
    logger.info(f"Extracted {len(cont1)} items from Dataset 1 and {len(cont2)} from Dataset 2.")
    
    # Prepare metadata and text
    final_data = []
    seen_names = set()

    for _, row in cont1.iterrows():
        name = str(row['Dish Name']).strip()
        if name.lower() in seen_names: continue
        seen_names.add(name.lower())
        group = str(row.get('Food_Group', 'Continental'))
        final_data.append({
            "name": name,
            "category": "Continental",
            "ingredients": "N/A", 
            "text": f"Food Item: {name}. Category: {group}. Cuisine: Continental."
        })
        
    for _, row in cont2.iterrows():
        name = str(row['food name']).strip()
        if name.lower() in seen_names: continue
        seen_names.add(name.lower())
        cat = str(row.get('Category', 'Continental'))
        final_data.append({
            "name": name,
            "category": "Continental",
            "ingredients": "N/A",
            "text": f"Food Item: {name}. Category: {cat}. Cuisine: Continental."
        })

    logger.info(f"Total unique continental items for Vector DB: {len(final_data)}")

    # --- Step 2: Loading Model ---
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    logger.info(f"Loading Embedding Model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    texts = [item['text'] for item in final_data]
    metadata = [{k: v for k, v in item.items() if k != 'text'} for item in final_data]
    
    # --- Step 3: Generating Embeddings ---
    logger.info("Generating and normalizing embeddings (this may take a few minutes)...")
    all_embeddings = []
    batch_size = 16 # Smaller batch size for stability
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**inputs)
        
        batch_emb = mean_pooling(output, inputs['attention_mask'])
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
        all_embeddings.append(batch_emb)
        if (i // batch_size) % 5 == 0:
             logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} items")
        
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # --- Step 4: Saving ---
    logger.info(f"Saving new VDB to {vdb_path}...")
    save_data = {
        "embeddings": final_embeddings,
        "metadata": metadata,
        "model_name": model_name
    }
    
    try:
        torch.save(save_data, vdb_path)
        logger.info(f"✓ Vector DB successfully written to {vdb_path}")
        logger.info(f"Summary: {len(metadata)} items, {final_embeddings.shape[1]} dimensions.")
    except Exception as e:
        logger.error(f"Failed to save VDB: {e}")

if __name__ == "__main__":
    create_continental_vdb()
