import pandas as pd

sample = pd.read_csv('dataset/sample_messages.csv')
ids = ['sample_msg_005','sample_msg_042','sample_msg_046',
       'sample_msg_049','sample_msg_013','sample_msg_014','sample_msg_015']

for mid in ids:
    row = sample[sample['message_id'] == mid].iloc[0]
    print(f"--- {mid} ---")
    print(f"Expected action: {row['action']} | type: {row['message_type']}")
    print(f"Media: {row['media_type']} | {row['media_id']}")
    print(f"Text: {str(row['message_text'])[:200]}")
    print()