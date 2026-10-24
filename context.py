import pandas as pd
import json

# ── Load all CSVs once ──────────────────────────────────────────────
messages        = pd.read_csv('dataset/messages.csv')
users           = pd.read_csv('dataset/users.csv')
groups          = pd.read_csv('dataset/groups.csv')
group_members   = pd.read_csv('dataset/group_members.csv')
business        = pd.read_csv('dataset/business_accounts.csv')
user_business   = pd.read_csv('dataset/user_business_history.csv')
msg_history     = pd.read_csv('dataset/message_history.csv')
msg_events      = pd.read_csv('dataset/message_events.csv')
images          = pd.read_csv('dataset/images.csv')
voice_notes     = pd.read_csv('dataset/voice_notes.csv')
daily_summary   = pd.read_csv('dataset/daily_notification_summary.csv')

# ── Safe value helpers ───────────────────────────────────────────────
def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except:
        return default

def safe_bool(value):
    if pd.isna(value):
        return None
    return bool(value)

# ── Main function ────────────────────────────────────────────────────
def build_context(message_row):
    ctx = {}

    # ── 1. The message itself ────────────────────────────────────────
    ctx['message'] = {
        'message_id':        safe_str(message_row['message_id']),
        'conversation_type': safe_str(message_row['conversation_type']),
        'message_text':      safe_str(message_row['message_text']),
        'media_type':        safe_str(message_row['media_type']),
        'media_id':          safe_str(message_row['media_id']),
        'forwarded_count':   safe_int(message_row['forwarded_count']),
        'created_at':        safe_str(message_row['created_at']),
        'group_id':          safe_str(message_row['group_id']),
        'business_id':       safe_str(message_row['business_id']),
        'sender_user_id':    safe_str(message_row['sender_user_id']),
    }

    # ── 2. User profile ──────────────────────────────────────────────
    user = users[users['user_id'] == message_row['user_id']]
    ctx['user'] = user.iloc[0].to_dict() if not user.empty else {}

    # ── 3. Daily notification load ───────────────────────────────────
    date_str = safe_str(message_row['created_at'])[:10]
    daily = daily_summary[
        (daily_summary['user_id'] == message_row['user_id']) &
        (daily_summary['date'].astype(str).str.startswith(date_str))
    ]
    ctx['daily_load'] = daily.iloc[0].to_dict() if not daily.empty else {}

    # ── 4. Group info ────────────────────────────────────────────────
    group_id = message_row['group_id']
    if pd.notna(group_id) and safe_str(group_id) != '':
        group = groups[groups['group_id'] == group_id]
        ctx['group'] = group.iloc[0].to_dict() if not group.empty else {}

        member = group_members[
            (group_members['group_id'] == group_id) &
            (group_members['user_id'] == message_row['user_id'])
        ]
        if not member.empty:
            m = member.iloc[0].to_dict()
            # Fix: use correct field name group_muted_by_user
            m['is_muted'] = m.get('group_muted_by_user', False)
            ctx['user_group_role'] = m
        else:
            ctx['user_group_role'] = {}
    else:
        ctx['group'] = {}
        ctx['user_group_role'] = {}

    # ── 5. Business info ─────────────────────────────────────────────
    business_id = message_row['business_id']
    if pd.notna(business_id) and safe_str(business_id) != '':
        biz = business[business['business_id'] == business_id]
        ctx['business'] = biz.iloc[0].to_dict() if not biz.empty else {}

        ub = user_business[
            (user_business['user_id'] == message_row['user_id']) &
            (user_business['business_id'] == business_id)
        ]
        if not ub.empty:
            ctx['user_business_history'] = ub.iloc[0].to_dict()
        else:
            # No history — user is unknown to this business
            ctx['user_business_history'] = {'no_history': True}
    else:
        ctx['business'] = {}
        ctx['user_business_history'] = {}

    # ── 6. Past messages + reactions ────────────────────────────────
    user_hist = msg_history[
        msg_history['user_id'] == message_row['user_id']
    ].copy()

    # Sort chronologically — most recent first
    user_hist['created_at'] = pd.to_datetime(
        user_hist['created_at'], errors='coerce'
    )
    user_hist = user_hist.sort_values('created_at', ascending=False)

    # Narrow to same source
    if pd.notna(group_id) and safe_str(group_id) != '':
        relevant = user_hist[user_hist['group_id'] == group_id]
    elif pd.notna(business_id) and safe_str(business_id) != '':
        relevant = user_hist[user_hist['business_id'] == business_id]
    elif pd.notna(message_row.get('sender_user_id')) and safe_str(message_row.get('sender_user_id')) != '':
        relevant = user_hist[
            user_hist['sender_user_id'] == message_row['sender_user_id']
        ]
    else:
        relevant = pd.DataFrame()

    # Take 5 most recent
    past = relevant.head(5)

    if not past.empty:
        past_ids = past['message_id'].tolist()
        events = msg_events[msg_events['message_id'].isin(past_ids)]

        # Only show 3 in prompt but evidence = all 5
        past_for_prompt = past.head(3)
        ctx['past_messages'] = past_for_prompt[
            ['message_id', 'message_text', 'created_at']
        ].to_dict('records')
        ctx['past_events']          = events.to_dict('records')
        ctx['evidence_message_ids'] = past_ids  # all 5 as evidence
    else:
        ctx['past_messages']        = []
        ctx['past_events']          = []
        ctx['evidence_message_ids'] = []

    # ── 7. Media path ────────────────────────────────────────────────
    media_type = safe_str(message_row['media_type'])
    media_id   = safe_str(message_row['media_id'])

    if media_type == 'image' and media_id:
        img = images[images['image_id'] == media_id]
        ctx['media_path'] = safe_str(img.iloc[0]['file_path']) if not img.empty else ''
    elif media_type == 'voice' and media_id:
        vn = voice_notes[voice_notes['voice_note_id'] == media_id]
        ctx['media_path'] = safe_str(vn.iloc[0]['file_path']) if not vn.empty else ''
    else:
        ctx['media_path'] = ''

    return ctx


# ── TEST ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for i in range(3):
        row = messages.iloc[i]
        ctx = build_context(row)
        print("=" * 60)
        print(f"MESSAGE: {ctx['message']['message_id']}")
        print(f"Text: {ctx['message']['message_text'][:80]}")
        print(f"User muted group: {ctx['user_group_role'].get('is_muted')}")
        print(f"Business no_history: {ctx['user_business_history'].get('no_history')}")
        print(f"Past messages: {len(ctx['past_messages'])}")
        print(f"Evidence IDs: {ctx['evidence_message_ids']}")
        print()