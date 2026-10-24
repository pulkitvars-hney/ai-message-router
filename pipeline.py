from dotenv import load_dotenv
import pandas as pd
import time
import os
from context import build_context
from media import get_media_description
from classifier import classify_message

load_dotenv()

messages = pd.read_csv('dataset/messages.csv')
print(f"Total messages: {len(messages)}")

results = []

for i, row in messages.iterrows():
    msg_id = row['message_id']
    media_type = str(row['media_type'])

    print(f"[{i+1}/{len(messages)}] {msg_id} | {row['conversation_type']} | media={media_type}")

    try:
        ctx = build_context(row)

        media_desc = ''
        if media_type in ('image', 'voice') and str(row['media_id']) != 'nan':
            media_desc = get_media_description(media_type, ctx.get('media_path', ''))

        result = classify_message(ctx, media_desc)

        results.append({
            'message_id':           msg_id,
            'action':               result['action'],
            'message_type':         result['message_type'],
            'reason':               result['reason'],
            'confidence':           result['confidence'],
            'evidence_message_ids': result['evidence_message_ids']
        })

        print(f"   → {result['action']} | {result['message_type']} | {result['confidence']}")

    except Exception as e:
        print(f"   ERROR: {e}")
        results.append({
            'message_id':           msg_id,
            'action':               'digest',
            'message_type':         'unknown',
            'reason':               f'Processing error: {str(e)[:100]}',
            'confidence':           0.5,
            'evidence_message_ids': 'none'
        })

    time.sleep(0.3)

output_df = pd.DataFrame(results)
output_df.to_csv('dataset/output.csv', index=False)

print(f"\n{'='*60}")
print(f"DONE. {len(output_df)} messages processed.")
print(f"\nAction distribution:")
print(output_df['action'].value_counts())
print(f"\nMessage type distribution:")
print(output_df['message_type'].value_counts())