import json
import random
import pandas as pd

def main():
    random.seed(42)
    
    # Load the audit report V9
    with open('dataset/v2/audit_labels_v9_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
        
    flagged = report.get('flagged', [])
    mislabeled = [item for item in flagged if item['verdict'] == 'MISLABELED']
    weak_bias = [item for item in flagged if item['verdict'] == 'WEAK_BIAS']
    
    # Sample 100 or as many as available
    sample_mislabeled = random.sample(mislabeled, min(100, len(mislabeled)))
    sample_weak_bias = random.sample(weak_bias, min(100, len(weak_bias)))
    
    # Get IDs of flagged items so we don't pick them as CORRECT
    flagged_ids = {item['id'] for item in flagged}
    
    # Load the full dataset to get CORRECT sentences
    with open('dataset/v2/bias_sentences_v9.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    all_sentences = dataset.get('entries', dataset) if isinstance(dataset, dict) else dataset
    
    # A sentence is CORRECT if it's not in flagged_ids but was audited.
    # We pick 100 random unflagged sentences.
    unflagged = [s for s in all_sentences if s['id'] not in flagged_ids]
    sample_correct = random.sample(unflagged, min(100, len(unflagged)))
    
    # Prepare rows for the CSV
    rows = []
    
    for item in sample_mislabeled:
        rows.append({
            'id': item['id'],
            'text': item['text'],
            'original_label': item['labeled_bias'],
            'prompt1_verdict': 'MISLABELED',
            'prompt1_reason': item.get('reason', ''),
            'human_verdict': '',  # To be filled by Ana
            'human_notes': ''
        })
        
    for item in sample_weak_bias:
        rows.append({
            'id': item['id'],
            'text': item['text'],
            'original_label': item['labeled_bias'],
            'prompt1_verdict': 'WEAK_BIAS',
            'prompt1_reason': item.get('reason', ''),
            'human_verdict': '',  # To be filled by Ana
            'human_notes': ''
        })
        
    for item in sample_correct:
        rows.append({
            'id': item['id'],
            'text': item['text'],
            'original_label': item.get('has_bias', ''),
            'prompt1_verdict': 'CORRECT',
            'prompt1_reason': 'Passed audit without flags',
            'human_verdict': '',  # To be filled by Ana
            'human_notes': ''
        })
        
    # Shuffle the rows so they are mixed
    random.shuffle(rows)
    
    df = pd.DataFrame(rows)
    output_file = 'dataset/v2/human_audit_sample_v9_300.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Generated {len(df)} samples: {len(sample_mislabeled)} MISLABELED, {len(sample_weak_bias)} WEAK_BIAS, {len(sample_correct)} CORRECT.")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
