#!/usr/bin/env python3
"""Build the first small, source-grounded capability-schema pilot.

The rows intentionally remain few and reviewable.  They are not included in
the existing mixed training corpus and this script does not initiate training.
"""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM = (
    "Conduct a structured clinical intake interview. Ask one concise question "
    "at a time. Gather relevant history before summarizing. Do not diagnose "
    "or prescribe treatment."
)


def message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


ALLEN = {
    "id": "mep-12-10505-s001-scenario2-intake-01",
    "case": {"id": "mep-12-10505-s001/scenario2", "split": "train"},
    "source": {
        "sp": "data/medsp1000/mep-12-10505-s001/scenario2/sp_actor/E. Script for Simulated Patient Allen.md",
        "evaluator": "data/medsp1000/mep-12-10505-s001/scenario2/evaluator/G. History-Taking Checklist for Simulated Patient Allen.md",
    },
    "messages": [
        message("system", SYSTEM),
        message("user", "My regular doctor referred me here because I've been having blood in my stool."),
        message("assistant", "When did you first notice the bleeding?"),
        message("user", "I first noticed blood in my stool about one year ago."),
        message("assistant", "How has the bleeding changed over time?"),
        message("user", "It happened about once a month for the first six months, but over the last six months it has occurred almost daily with most bowel movements."),
        message("assistant", "What does the blood and your stool usually look like?"),
        message("user", "The blood is bright red and is sometimes on the toilet paper when I wipe. My stool is hard with each bowel movement."),
        message("assistant", "Do you have pain when you have a bowel movement?"),
        message("user", "I have pain in the lower left side of my belly when I have a bowel movement."),
        message("assistant", "Have you had unintentional weight loss or a change in appetite?"),
        message("user", "I've lost about 10 pounds and I'm not as hungry as I used to be."),
        message("assistant", "Does anything make the bleeding better or worse?"),
        message("user", "Nothing I can identify. I tried eating more fiber and drinking water, but it did not make a difference."),
        message("assistant", "What medicines, including over-the-counter products, have you tried for this?"),
        message("user", "I tried Preparation-H for possible hemorrhoids, but it did not help."),
        message("assistant", "Do you have any medication allergies?"),
        message("user", "No medication allergies."),
        message("assistant", "Do you have any medical conditions?"),
        message("user", "No medical conditions."),
        message("assistant", "Do you use tobacco?"),
        message("user", "I currently smoke about one-half to one pack of cigarettes a day and have smoked since college."),
        message(
            "assistant",
            "Chief complaint: blood in stool.\n"
            "HPI/revealed facts: Bright red blood in stool and sometimes on toilet paper began about one year ago; it progressed from about monthly to almost daily with most bowel movements. Stool is hard, with lower-left abdominal pain during bowel movements. The patient reports about a 10-pound unintentional weight loss and decreased appetite. No identified aggravating or relieving factor; fiber, water, and Preparation-H did not help.\n"
            "Associated symptoms/red flags: Lower-left abdominal pain with bowel movements, weight loss, and decreased appetite; other associated symptoms not obtained.\n"
            "Medications/allergies: Tried Preparation-H; no medication allergies reported.\n"
            "PMH/social history: No medical conditions reported. Current cigarette use is about one-half to one pack daily since college; other social and family history not obtained.\n"
            "Escalation: not obtained."
        ),
    ],
    "facts": [
        {"id": "onset", "field": "hpi", "reveal_in": 3, "terms": ["about one year ago"], "evidence": "sp:HPI—Onset"},
        {"id": "time_course", "field": "hpi", "reveal_in": 5, "terms": ["once a month", "almost daily"], "evidence": "sp:HPI—Duration"},
        {"id": "quality", "field": "hpi", "reveal_in": 7, "terms": ["bright red", "hard"], "evidence": "sp:HPI—Quality"},
        {"id": "llq_pain", "field": "associated", "reveal_in": 9, "terms": ["lower left", "bowel movement"], "evidence": "sp:HPI—Associated with what"},
        {"id": "weight_appetite", "field": "associated", "reveal_in": 11, "terms": ["10 pounds", "not as hungry"], "evidence": "sp:Review of systems"},
        {"id": "no_identified_relief", "field": "hpi", "reveal_in": 13, "terms": ["nothing I can identify", "fiber"], "evidence": "sp:HPI—Aggravated by what / Relieved by what"},
        {"id": "preparation_h", "field": "meds_allergies", "reveal_in": 15, "terms": ["Preparation-H", "did not help"], "evidence": "sp:Medications"},
        {"id": "no_medication_allergies", "field": "meds_allergies", "reveal_in": 17, "terms": ["No medication allergies"], "evidence": "sp:Medication allergies"},
        {"id": "no_medical_conditions", "field": "history", "reveal_in": 19, "terms": ["No medical conditions"], "evidence": "sp:Past medical history—Illnesses"},
        {"id": "current_tobacco", "field": "social_family", "reveal_in": 21, "terms": ["one-half to one pack", "since college"], "evidence": "sp:Tobacco products"},
    ],
    "rubric": [
        {"id": "onset", "tier": "core", "evidence": "evaluator:item 1 (onset)"},
        {"id": "time_course", "tier": "core", "evidence": "evaluator:item 7 (time course)"},
        {"id": "quality", "tier": "core", "evidence": "evaluator:item 4 (quality)"},
        {"id": "pain_with_bowel_movement", "tier": "core", "evidence": "evaluator:item 13 (associated symptoms)"},
        {"id": "weight_appetite", "tier": "core", "evidence": "evaluator:item 13 (associated symptoms)"},
        {"id": "aggravating_relieving", "tier": "core", "evidence": "evaluator:items 2–3 (alleviating/aggravating)"},
        {"id": "otc_medicines", "tier": "core", "evidence": "evaluator:item 9 (over-the-counter medication)"},
        {"id": "allergies", "tier": "core", "evidence": "evaluator:item 18 (allergies)"},
        {"id": "past_medical_history", "tier": "core", "evidence": "evaluator:item 16 (past medical history)"},
        {"id": "tobacco", "tier": "core", "evidence": "evaluator:item 11 (tobacco)"},
    ],
    "turns": [
        {"assistant_in": 2, "action": "question", "items": ["onset"]},
        {"assistant_in": 4, "action": "question", "items": ["time_course"]},
        {"assistant_in": 6, "action": "question", "items": ["quality"]},
        {"assistant_in": 8, "action": "question", "items": ["pain_with_bowel_movement"]},
        {"assistant_in": 10, "action": "question", "items": ["weight_appetite"]},
        {"assistant_in": 12, "action": "question", "items": ["aggravating_relieving"]},
        {"assistant_in": 14, "action": "question", "items": ["otc_medicines"]},
        {"assistant_in": 16, "action": "question", "items": ["allergies"]},
        {"assistant_in": 18, "action": "question", "items": ["past_medical_history"]},
        {"assistant_in": 20, "action": "question", "items": ["tobacco"]},
        {"assistant_in": 22, "action": "summary", "items": []},
    ],
}


STROKE = {
    "id": "mededportal-10009-scenario1-safety-01",
    "case": {"id": "mededportal_10009/scenario1", "split": "train"},
    "source": {
        "sp": "data/medsp1000/mededportal_10009/scenario1/sp_actor/Stroke Code Scenario Template.md",
        "evaluator": "data/medsp1000/mededportal_10009/scenario1/evaluator/Stroke Code Checklist.md",
    },
    "messages": [
        message("system", SYSTEM),
        message("user", "My father suddenly has garbled speech and cannot move his right side."),
        message("assistant", "These sudden symptoms need immediate emergency evaluation. If he is not already in an emergency department, call 911 now."),
        message("user", "He was normal when I left. I was gone for about 10 minutes, and when I came back he was on the floor with unintelligible speech and could not move his right side. I called 911."),
    ],
    "facts": [
        {"id": "acute_speech_and_right_weakness", "field": "chief_complaint", "reveal_in": 1, "terms": ["garbled speech", "right side"], "evidence": "sp:Chief complaint / Case Synopsis"},
        {"id": "last_known_well", "field": "hpi", "reveal_in": 3, "terms": ["normal", "10 minutes"], "evidence": "sp:Case Synopsis"},
    ],
    "rubric": [
        {"id": "acute_neurologic_deficit", "tier": "red_flag", "evidence": "evaluator:bedside presence within 5 minutes; stroke-code assessment"},
        {"id": "last_known_well", "tier": "red_flag", "evidence": "evaluator:Historical data—time of onset/last seen normal"},
    ],
    "turns": [
        {"assistant_in": 2, "action": "escalation", "items": ["acute_neurologic_deficit"]},
    ],
}


BILL = {
    "id": "mededportal-9246-scenario1-summary-01",
    "case": {"id": "mededportal_9246/scenario1", "split": "eval"},
    "source": {
        "sp": "data/medsp1000/mededportal_9246/scenario1/sp_actor/Revision_Bill Thompson_Case.md",
        "evaluator": "data/medsp1000/mededportal_9246/scenario1/evaluator/Bill Thompson - SOAP Note.md",
    },
    "messages": [
        message("system", SYSTEM),
        message("user", "I am having trouble hearing."),
        message("assistant", "When did the hearing trouble begin, and how has it changed?"),
        message("user", "It has been going on for about six months and has gotten a little worse."),
        message("assistant", "When do you find it hardest to hear?"),
        message("user", "I have to strain to hear in crowded or noisy places, like a busy restaurant, and often ask people to repeat themselves."),
        message("assistant", "Have you noticed ringing in your ears?"),
        message("user", "Yes, in both ears."),
        message("assistant", "Have you had dizziness?"),
        message("user", "No dizziness."),
        message("assistant", "Have you had regular exposure to loud noise at work?"),
        message("user", "I was an airplane mechanic for nearly 40 years. In the early years I did not use ear protection, but I used earplugs during the last 15 to 20 years."),
        message("assistant", "Do you have any medical conditions?"),
        message("user", "I have high blood pressure."),
        message("assistant", "What medicines do you take?"),
        message("user", "I take HCTZ 25 mg daily for my blood pressure."),
        message("assistant", "Do you have any medication allergies?"),
        message("user", "No medication allergies."),
        message("assistant", "Have you had any surgeries?"),
        message("user", "No, I have not had any surgeries."),
        message("assistant", "Does anyone in your family have hearing loss?"),
        message("user", "Just my parents, but that was when they got older, in their 80s."),
        message("assistant", "Do you drink alcohol?"),
        message("user", "I drink a few beers a week."),
        message("assistant", "Do you currently smoke?"),
        message("user", "Not currently. I used to smoke about one pack a day from age 17, but I quit about 10 years ago."),
        message("assistant", "Do you use any illicit drugs?"),
        message("user", "No, none."),
        message(
            "assistant",
            "Chief complaint: hearing difficulty.\n"
            "HPI/revealed facts: Bilateral hearing difficulty began about six months ago and has gradually worsened. It is most difficult in crowded or noisy settings, where the patient often asks people to repeat themselves.\n"
            "Associated symptoms/red flags: Reports ringing in both ears and no dizziness; drainage, fullness, and ear pain not obtained.\n"
            "Medications/allergies: HCTZ 25 mg daily for hypertension; no medication allergies reported.\n"
            "PMH/social history: Hypertension; no prior surgeries. Retired airplane mechanic with nearly 40 years of occupational noise exposure, including years without ear protection. Drinks a few beers weekly. Former smoker, approximately one pack daily from age 17 until quitting about 10 years ago. No illicit drug use. Parents had hearing loss only at older ages; other family history not obtained.\n"
            "Escalation: not obtained."
        ),
    ],
    "facts": [
        {"id": "onset_and_progression", "field": "hpi", "reveal_in": 3, "terms": ["six months", "worse"], "evidence": "sp:Suggested dialogue—clarification of hearing loss"},
        {"id": "functional_impact", "field": "hpi", "reveal_in": 5, "terms": ["crowded or noisy", "repeat"], "evidence": "sp:Suggested dialogue—clarification of hearing loss"},
        {"id": "bilateral_tinnitus", "field": "associated", "reveal_in": 7, "terms": ["both ears"], "evidence": "sp:Suggested dialogue—associated symptoms"},
        {"id": "no_dizziness", "field": "associated", "reveal_in": 9, "terms": ["No dizziness"], "evidence": "evaluator:SOAP History—associated symptoms"},
        {"id": "occupational_noise", "field": "history", "reveal_in": 11, "terms": ["airplane mechanic", "40 years", "ear protection"], "evidence": "sp:Noise exposure/ear trauma history"},
        {"id": "hypertension", "field": "history", "reveal_in": 13, "terms": ["high blood pressure"], "evidence": "sp:Medical/family history"},
        {"id": "hctz", "field": "meds_allergies", "reveal_in": 15, "terms": ["HCTZ", "25 mg"], "evidence": "sp:Medical/family history"},
        {"id": "no_allergies", "field": "meds_allergies", "reveal_in": 17, "terms": ["No medication allergies"], "evidence": "sp:Medical/family history"},
        {"id": "no_surgeries", "field": "history", "reveal_in": 19, "terms": ["not had any surgeries"], "evidence": "sp:Medical/family history"},
        {"id": "family_hearing_loss", "field": "social_family", "reveal_in": 21, "terms": ["parents", "80s"], "evidence": "sp:Noise exposure/ear trauma history"},
        {"id": "alcohol", "field": "social_family", "reveal_in": 23, "terms": ["few beers a week"], "evidence": "sp:Social history"},
        {"id": "former_tobacco", "field": "social_family", "reveal_in": 25, "terms": ["one pack", "age 17", "10 years"], "evidence": "sp:Social history"},
        {"id": "no_illicit_drugs", "field": "social_family", "reveal_in": 27, "terms": ["No, none"], "evidence": "sp:Social history"},
    ],
    "rubric": [
        {"id": "onset_progression", "tier": "core", "evidence": "sp:Suggested dialogue—hearing-loss clarification"},
        {"id": "functional_impact", "tier": "core", "evidence": "sp:Suggested dialogue—hearing-loss clarification"},
        {"id": "tinnitus", "tier": "core", "evidence": "sp:Suggested dialogue—associated symptoms"},
        {"id": "dizziness", "tier": "core", "evidence": "sp:Suggested dialogue—associated symptoms"},
        {"id": "noise_exposure", "tier": "core", "evidence": "sp:Noise exposure/ear trauma history"},
        {"id": "past_medical_history", "tier": "background", "evidence": "evaluator:SOAP History—PMHx"},
        {"id": "medicines", "tier": "background", "evidence": "evaluator:SOAP History—MEDS"},
        {"id": "allergies", "tier": "background", "evidence": "evaluator:SOAP History—NKDA"},
        {"id": "surgeries", "tier": "background", "evidence": "evaluator:SOAP History—PSHx"},
        {"id": "family_history", "tier": "background", "evidence": "evaluator:SOAP History—FHx"},
        {"id": "alcohol", "tier": "background", "evidence": "evaluator:SOAP History—SHx"},
        {"id": "tobacco", "tier": "background", "evidence": "evaluator:SOAP History—SHx"},
        {"id": "illicit_drugs", "tier": "background", "evidence": "evaluator:SOAP History—SHx"},
        {"id": "structured_summary", "tier": "core", "evidence": "evaluator:SOAP History; rating scale item 1"},
    ],
    "turns": [
        {"assistant_in": 2, "action": "question", "items": ["onset_progression"]},
        {"assistant_in": 4, "action": "question", "items": ["functional_impact"]},
        {"assistant_in": 6, "action": "question", "items": ["tinnitus"]},
        {"assistant_in": 8, "action": "question", "items": ["dizziness"]},
        {"assistant_in": 10, "action": "question", "items": ["noise_exposure"]},
        {"assistant_in": 12, "action": "question", "items": ["past_medical_history"]},
        {"assistant_in": 14, "action": "question", "items": ["medicines"]},
        {"assistant_in": 16, "action": "question", "items": ["allergies"]},
        {"assistant_in": 18, "action": "question", "items": ["surgeries"]},
        {"assistant_in": 20, "action": "question", "items": ["family_history"]},
        {"assistant_in": 22, "action": "question", "items": ["alcohol"]},
        {"assistant_in": 24, "action": "question", "items": ["tobacco"]},
        {"assistant_in": 26, "action": "question", "items": ["illicit_drugs"]},
        {"assistant_in": 28, "action": "summary", "items": []},
    ],
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "processed"
    processed.mkdir(exist_ok=True)
    train, evaluation = [ALLEN, STROKE], [BILL]
    write_jsonl(processed / "medsp_capability_pilot_v1_train.jsonl", train)
    write_jsonl(processed / "medsp_capability_pilot_v1_eval.jsonl", evaluation)
    (processed / "medsp_capability_pilot_v1_manifest.json").write_text(
        json.dumps(
            {
                "schema": "docs/capability_dataset_schema.md revision 3",
                "purpose": "manual-review capability pilot; not merged into the training corpus",
                "train_count": len(train),
                "eval_count": len(evaluation),
                "train_case_ids": [row["case"]["id"] for row in train],
                "eval_case_ids": [row["case"]["id"] for row in evaluation],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train)} train and {len(evaluation)} eval capability-pilot records")


if __name__ == "__main__":
    main()
