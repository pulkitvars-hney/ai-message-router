import pandas as pd
from context import build_context
from media import get_media_description
from classifier import classify_message

# Load sample messages
sample = pd.read_csv('dataset/sample_messages.csv')

print(f"Running pipeline on {len(sample)} sample messages...")
print()

predictions = []

for i, row in sample.iterrows():
    msg_id = row['message_id']
    media_type = str(row['media_type'])

    print(f"[{i+1}/{len(sample)}] {msg_id}")

    try:
        ctx = build_context(row)

        media_desc = ''
        if media_type in ('image', 'voice') and str(row['media_id']) != 'nan':
            media_desc = get_media_description(
                media_type, ctx.get('media_path', '')
            )

        result = classify_message(ctx, media_desc)
        predictions.append({
            'message_id':    msg_id,
            'action_pred':   result['action'],
            'type_pred':     result['message_type'],
            'reason_pred':   result['reason'],
            'confidence':    result['confidence'],
        })

        print(f"   predicted: {result['action']} | {result['message_type']}")
        print(f"   expected:  {row['action']} | {row['message_type']}")
        print()

    except Exception as e:
        print(f"   ERROR: {e}")
        predictions.append({
            'message_id':  msg_id,
            'action_pred': 'digest',
            'type_pred':   'unknown',
            'reason_pred': 'error',
            'confidence':  0.5,
        })

# ── Score ─────────────────────────────────────────────────────────────
pred_df = pd.DataFrame(predictions)
merged = sample.merge(pred_df, on='message_id')
total = len(merged)

action_correct = (merged['action'] == merged['action_pred']).sum()
type_correct   = (merged['message_type'] == merged['type_pred']).sum()

print("=" * 50)
print(f"EVALUATED: {total} messages")
print(f"ACTION accuracy:       {action_correct}/{total} = {action_correct/total*100:.1f}%")
print(f"MESSAGE TYPE accuracy: {type_correct}/{total} = {type_correct/total*100:.1f}%")
print("=" * 50)

print("\nACTION MISMATCHES:")
wrong = merged[merged['action'] != merged['action_pred']]
if wrong.empty:
    print("None!")
else:
    print(wrong[['message_id','action','action_pred','message_type']].to_string(index=False))

print("\nTYPE MISMATCHES:")
wrong2 = merged[merged['message_type'] != merged['type_pred']]
if wrong2.empty:
    print("None!")
else:
    print(wrong2[['message_id','message_type','type_pred','action']].to_string(index=False))

print("\nCONFUSION MATRIX (actions):")
print(pd.crosstab(
    merged['action'],
    merged['action_pred'],
    rownames=['Expected'],
    colnames=['Predicted']
))