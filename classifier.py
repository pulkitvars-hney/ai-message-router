from dotenv import load_dotenv
from groq import Groq
import os
import json
import re

load_dotenv()

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
MODEL = 'qwen/qwen3.8-27b'
VALID_ACTIONS = {'notify', 'digest', 'mute'}
VALID_TYPES = {
    'personal', 'urgent', 'event', 'payment',
    'business_update', 'promotion', 'greeting',
    'forward', 'spam', 'scam', 'unknown'
}

def classify_message(context: dict, media_description: str = '') -> dict:

    past_behavior = ''
    if context.get('past_events'):
        events = context['past_events']
        opened    = sum(e.get('message_opened', 0) for e in events)
        replied   = sum(e.get('message_replied', 0) for e in events)
        muted     = sum(e.get('muted_after_message', 0) for e in events)
        reported  = sum(e.get('message_reported', 0) for e in events)
        dismissed = sum(e.get('notification_dismissed', 0) for e in events)
        past_behavior = f"opened={opened}, replied={replied}, muted={muted}, reported={reported}, dismissed={dismissed}"

    biz_trust = ''
    if context.get('business'):
        b = context['business']
        biz_trust = f"""
Business: {b.get('display_name')}
Verified: {b.get('verified')}
Account age days: {b.get('account_age_days')}
Reports in 30d: {b.get('user_reports_30d')}
Domain match: {b.get('official_domain') == b.get('domain_used_by_sender')}
"""

    user = context.get('user', {})
    user_summary = f"""
DND window: {user.get('do_not_disturb_window')}
Messages opened 30d: {user.get('messages_opened_30d')}
Messages replied 30d: {user.get('messages_replied_30d')}
Notifications dismissed 30d: {user.get('notifications_dismissed_30d')}
Messages reported 30d: {user.get('messages_reported_30d')}
"""

    group_summary = ''
    if context.get('group'):
        g = context['group']
        role = context.get('user_group_role', {})
        group_summary = f"""
Group: {g.get('group_name')} ({g.get('group_type')})
Members: {g.get('member_count')}
User role: {role.get('role')}
User muted group: {role.get('is_muted')}
"""

    biz_history = ''
    if context.get('user_business_history'):
        uh = context['user_business_history']
        biz_history = f"""
Why user knows business: {uh.get('why_user_knows_account')}
Allows promotions: {uh.get('allows_promotions')}
Activity count 180d: {uh.get('activity_count_180d')}
Messages opened 30d: {uh.get('messages_opened_30d')}
"""

    past_msgs = ''
    if context.get('past_messages'):
        for pm in context['past_messages'][:3]:
            past_msgs += f"- [{pm['message_id']}] {str(pm['message_text'])[:80]}\n"

    evidence = context.get('evidence_message_ids', [])
    evidence_str = ';'.join(evidence) if evidence else 'none'
    prompt = f"""You are a WhatsApp message notification router. Decide how to handle this message for this specific user.

ROUTING OPTIONS:
- notify: urgent, personal, time-sensitive — interrupt the user now
- digest: useful but not urgent — show later in a summary  
- mute: spam, scam, unwanted, repetitive, suspicious — hide it

MESSAGE:
Type: {context['message']['conversation_type']}
Text: {context['message']['message_text'][:300]}
Media type: {context['message']['media_type']}
Forwarded count: {context['message']['forwarded_count']}
{f"Media content: {media_description}" if media_description else ""}

USER PROFILE:
{user_summary}
{f"GROUP INFO:{group_summary}" if group_summary else ""}
{f"BUSINESS INFO:{biz_trust}" if biz_trust else ""}
{f"USER-BUSINESS RELATIONSHIP:{biz_history}" if biz_history else ""}

PAST MESSAGES FROM SAME SOURCE:
{past_msgs if past_msgs else "No past messages found"}

USER BEHAVIOR ON PAST MESSAGES:
{past_behavior if past_behavior else "No past behavior data"}

SSTRICT DECISION RULES:

MUTE only when clearly one of these:
- Phishing, OTP theft, fake prize, impersonation → scam
- Unsolicited bulk forwards with no personal value → spam
- User explicitly opted out of this business → promotion + mute
- Unverified business with multiple reports sending promos → spam

NOTIFY when any of these:
- Message contains a deadline TODAY or words like "urgent", "immediately", "right now", "today only"
- Direct personal message from a known contact needing a response
- Society/building emergency or maintenance alert
- Payment due, failed, or pending requiring action
- Work message requiring immediate response
- Business update about an active order, delivery arriving today
- ANY message where delay would cause the user to miss something
- Group admin sending time-sensitive announcement
- Personal message that asks a direct question

DIGEST for everything else useful:
- Event announcements (meetings, festivals, school events, walkathons)
- Business updates from verified businesses (delivery scheduled, order placed)
- Promotions from businesses user has transacted with before
- Group announcements without same-day urgency
- Greeting messages → type must be greeting
- Forwarded content (news articles, jokes, tips) → type must be forward
- Useful but non-urgent personal messages

TYPE RULES — be very precise:
- event = scheduled activity with a future date (meeting, festival, school event, sports)
- urgent = action needed TODAY with real consequence if missed
- business_update = delivery/order/account status from verified business
- promotion = sale, offer, discount, coupon
- greeting = good morning, hello, festival wishes, friendly opener WITH NO other content
- forward = message that was forwarded (news, jokes, tips, inspirational content)
- personal = direct conversation between two real people
- spam = truly unwanted bulk unsolicited messages
- scam = fraud, phishing, fake prizes, OTP theft
- payment = money transaction, UPI, bank related
- unknown = genuinely unclear after reading all context

NEVER:
- Call a greeting message spam or forward
- Call a forwarded news/joke spam  
- Call an event message urgent unless it happens today
- Return unknown if you can make a reasonable guess


KEY DISTINCTIONS:
- event = scheduled activity (meeting, walkathon, festival, school event)
- urgent = needs immediate action (emergency, expiry today, access revoked)
- business_update = delivery, order, account status from verified business
- promotion = sale, discount, offer
- greeting = hello, good morning, festive wishes
- forward = forwarded message (news, jokes, inspirational content)
- personal = direct message between people
- spam = unsolicited bulk, repetitive, low quality
- scam = phishing, fraud, fake prizes, OTP requests

IMPORTANT:
- Do NOT mute legitimate promotions just because they are promotional
- Do NOT classify event messages as urgent
- Do NOT classify greetings as spam
- Prefer digest over mute when unsure
- Prefer notify over digest for time-sensitive personal messages

ALLOWED message_type values ONLY:
personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown

Respond with ONLY valid JSON, no markdown, no extra text:
{{"action": "notify|digest|mute", "message_type": "one of the allowed types", "reason": "one clear sentence", "confidence": 0.0}}"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    result = json.loads(raw)

    if result.get('action') not in VALID_ACTIONS:
        result['action'] = 'digest'
    if result.get('message_type') not in VALID_TYPES:
        result['message_type'] = 'unknown'

    result['evidence_message_ids'] = evidence_str

    return result


if __name__ == '__main__':
    import pandas as pd
    from context import build_context

    messages = pd.read_csv('dataset/messages.csv')

    for i in range(3):
        row = messages.iloc[i]
        print(f"\n{'='*60}")
        print(f"MESSAGE: {row['message_id']}")
        print(f"TEXT: {str(row['message_text'])[:80]}")
        ctx = build_context(row)
        result = classify_message(ctx)
        print(f"ACTION:     {result['action']}")
        print(f"TYPE:       {result['message_type']}")
        print(f"REASON:     {result['reason']}")
        print(f"CONFIDENCE: {result['confidence']}")
        print(f"EVIDENCE:   {result['evidence_message_ids']}")