"""Held-out evaluation set for the Center Desk RAG system.

These queries are deliberately phrased DIFFERENTLY from the canonical knowledge
base questions (paraphrases, colloquial wording) so the eval measures whether
retrieval generalizes — not whether it can echo back an exact string it already
stored. This is the point of a held-out set: never test on your training data.

Each in-scope item lists the canonical KB question(s) that *should* be
retrieved. Some real questions have more than one valid target (e.g. the seed
set and the expanded set both cover room numbers), so `expected` is a list and a
hit counts if ANY listed target appears.

Out-of-scope items (in_scope=False) test the grounding guardrail: the retriever
should return nothing above the score threshold, so the assistant refuses.
"""

# in-scope: query is a paraphrase; `expected` = acceptable canonical KB questions
IN_SCOPE = [
    {
        "query": "How do I set up call forwarding on the desk phone?",
        "expected": ["How do I forward the desk phone?"],
        "reference": "Touch the FC icon, go to Settings > Calling > Call Forwarding, pick the destination, and make a test call.",
    },
    {
        "query": "What are the steps to log a package in the mailroom?",
        "expected": ["How do I log a package into the mailroom system?"],
        "reference": "Scan it, enter the recipient name, verify room number, assign a storage location, write the number on three sides, and shelve it.",
    },
    {
        "query": "A parent is asking for their kid's room number, can I share it?",
        "expected": [
            "Can I give out a resident's room number to a parent?",
            "Can I give a resident's room number to a parent?",
        ],
        "reference": "No. Sharing a resident's room number is against policy/FERPA, even with parents.",
    },
    {
        "query": "How do I check a resident in?",
        "expected": ["How do I check in a resident?"],
        "reference": "Log into Nelson, swipe the ID, verify their info, and click Check-In.",
    },
    {
        "query": "What should I do when the fire alarm goes off?",
        "expected": [
            "What do I do in case of a fire alarm?",
            "What do I do during a fire alarm?",
        ],
        "reference": "Evacuate everyone, leave by the nearest exit, inform emergency services, and don't re-enter until cleared.",
    },
    {
        "query": "What's the closing procedure for the desk?",
        "expected": ["How do I close the Center Desk?"],
        "reference": "Check the check-out sheet, call residents for items, close the gate, log out, clean up, and clock out.",
    },
    {
        "query": "What does RTS mean?",
        "expected": ["What is RTS?"],
        "reference": "Return To Sender — used for packages for someone who no longer lives here with no forwarding address.",
    },
    {
        "query": "How does a student collect their package?",
        "expected": ["How does a resident pick up a package?"],
        "reference": "Verify identity, locate the package by number/location, mark it picked up in the system, and hand it over.",
    },
    {
        "query": "A resident is locked out of their room, what do I do?",
        "expected": [
            "What is the procedure for a room lockout?",
            "What do I do if a resident locks themselves out of their room?",
        ],
        "reference": "Verify they're assigned to the room, unlock the key vault, escort them and let them in, avoid issuing a temp card, and log it.",
    },
    {
        "query": "How many radios should I grab at each desk?",
        "expected": ["How many radios are at each desk?"],
        "reference": "Desk 1 has 1 radio, Desk 2 has 3 radios.",
    },
    {
        "query": "What's the duty chain?",
        "expected": ["What is the duty chain?"],
        "reference": "The escalation path for issues beyond your role: RA on duty, then on-call professional staff.",
    },
    {
        "query": "Why does FERPA matter at the front desk?",
        "expected": ["What is FERPA and why does it matter at the desk?"],
        "reference": "It protects students' records/personal info, so you can't share details like room number or whether someone lives here.",
    },
    {
        "query": "How do I submit a normal, non-urgent maintenance request?",
        "expected": ["How do I report a routine maintenance issue?"],
        "reference": "Submit it through ASKrps (or the RA) with location and description — not Emergency Maintenance.",
    },
    {
        "query": "What kinds of issues count as a maintenance emergency?",
        "expected": [
            "What counts as an emergency maintenance issue?",
            "When should I call Emergency Maintenance?",
        ],
        "reference": "Bodily fluids, flooding, no heat, icy sidewalks, bathroom supply shortages, AC above 80F, gas smells, etc.",
    },
    {
        "query": "How do I sign a guest in?",
        "expected": ["What is the guest sign-in procedure?"],
        "reference": "Take a photo ID, record the required info in the guest log, confirm the host, and explain the guest policy.",
    },
    {
        "query": "How much do I charge for a replacement ID card?",
        "expected": ["How much does a replacement card cost?"],
        "reference": "The first replacement each semester is free; additional ones are charged.",
    },
    {
        "query": "What's Nelson for?",
        "expected": ["What is Nelson used for?"],
        "reference": "Checking residents in and out.",
    },
    {
        "query": "What is ResCard?",
        "expected": ["What is ResCard used for?"],
        "reference": "Managing resident access cards — deactivating lost/old cards and issuing replacements.",
    },
    {
        "query": "What do we use Arrivals for?",
        "expected": ["What is Arrivals used for?"],
        "reference": "Managing room assignments and room changes.",
    },
    {
        "query": "After forwarding the phone, how do I confirm it actually worked?",
        "expected": ["How do I make a test call after forwarding the phone?"],
        "reference": "Call the desk line from another phone and confirm it routes to the chosen destination.",
    },
    {
        "query": "Someone handed me a lost item they found, what do I do?",
        "expected": ["What do I do when someone turns in a found item?"],
        "reference": "Log it with description/date/location and store it securely; deactivate or escalate sensitive items.",
    },
    {
        "query": "A resident says a dryer is broken, what do I do?",
        "expected": ["What do I do if a resident reports a broken washer or dryer?"],
        "reference": "Note the machine number/location, post an out-of-order notice, and submit a service request.",
    },
    {
        "query": "When should I file an incident report?",
        "expected": ["When do I need to write an incident report?"],
        "reference": "For anything out of the ordinary — policy violations, injuries, conflicts, damage, anything escalated.",
    },
    {
        "query": "How do I deal with a noise complaint during quiet hours?",
        "expected": ["What are quiet hours and how do I handle a noise complaint?"],
        "reference": "Log it and notify the RA on duty, who handles the confrontation; the desk doesn't enforce conduct.",
    },
    {
        "query": "How should I take a cash payment?",
        "expected": ["How do I handle cash payments at the desk?"],
        "reference": "Follow the cash-handling procedure: count in front of the resident, give a receipt, log it, secure the cash.",
    },
    {
        "query": "What if the check-in system is down?",
        "expected": ["What do I do if Nelson, ResCard, or Arrivals is down?"],
        "reference": "Note the error, retry once, use the manual backup for time-sensitive tasks, report it, and log to enter later.",
    },
    {
        "query": "My replacement didn't show up for the next shift, now what?",
        "expected": ["What should I do if my relief does not show up?"],
        "reference": "Don't leave the desk unattended; call the next CDA, then notify your supervisor/RA on duty.",
    },
    {
        "query": "A resident is having a medical emergency, what do I do?",
        "expected": ["What do I do if a resident reports a medical emergency?"],
        "reference": "Call 911, give the location, notify the duty chain, stay on the line; don't give care beyond your training.",
    },
    {
        "query": "There's a tornado warning, what do I do?",
        "expected": ["What do I do during a severe weather warning (tornado)?"],
        "reference": "Move people to the designated shelter area away from windows and stay until an official all-clear.",
    },
    {
        "query": "How do I lend a vacuum or other item to a resident?",
        "expected": ["How do I check out an item to a resident?"],
        "reference": "Verify identity, record the item/resident/time on the check-out sheet, and tell them when it's due back.",
    },
]

OUT_OF_SCOPE = [
    {"query": "What is the capital of France?"},
    {"query": "Write me a poem about the ocean."},
    {"query": "What's the weather forecast for tomorrow?"},
    {"query": "How do I bake chocolate chip cookies?"},
    {"query": "Who is the current president of the United States?"},
    {"query": "Recommend a good action movie to watch tonight."},
]
