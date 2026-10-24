import pandas as pd

messages = pd.read_csv('dataset/messages.csv')
sample = pd.read_csv('dataset/sample_messages.csv')
users = pd.read_csv('dataset/users.csv')
groups = pd.read_csv('dataset/groups.csv')
business = pd.read_csv('dataset/business_accounts.csv')
msg_history = pd.read_csv('dataset/message_history.csv')
msg_events = pd.read_csv('dataset/message_events.csv')
images = pd.read_csv('dataset/images.csv')
voice_notes = pd.read_csv('dataset/voice_notes.csv')

print("GROUPS - columns:", groups.columns.tolist())
print(groups.head(2))
print()

print("BUSINESS - columns:", business.columns.tolist())
print(business.head(2))
print()

print("MESSAGE HISTORY - columns:", msg_history.columns.tolist())
print(msg_history.head(2))
print()

print("MESSAGE EVENTS - columns:", msg_events.columns.tolist())
print(msg_events.head(2))
print()

print("IMAGES - columns:", images.columns.tolist())
print(images.head(2))
print()

print("VOICE NOTES - columns:", voice_notes.columns.tolist())
print(voice_notes.head(2))
print()

# Most important: see what action types exist in sample
print("ACTIONS in sample:", sample['action'].value_counts().to_dict())
print("MESSAGE TYPES in sample:", sample['message_type'].value_counts().to_dict())