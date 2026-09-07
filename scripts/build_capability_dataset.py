#!/usr/bin/env python3
"""Build capability dataset v1 by extending the validated three-case pilot."""
from __future__ import annotations

import json
from pathlib import Path

from build_capability_pilot import ALLEN, BILL, STROKE, SYSTEM, message


ELSIE = {
    "id": "mededportal-9507-scenario1-falls-01",
    "case": {"id": "mededportal_9507/scenario1", "split": "train"},
    "source": {"sp": "data/medsp1000/mededportal_9507/scenario1/sp_actor/Standardized Patient Training Materials.md", "evaluator": "data/medsp1000/mededportal_9507/scenario1/evaluator/Standardized Patient Training Materials.md"},
    "messages": [
        message("system", SYSTEM),
        message("user", "Two weeks ago I went to the Emergency Room after I fell at home and hit my knee and jaw. I'm scared of falling again because I've been falling too much."),
        message("assistant", "Could you tell me what happened during the fall that led to the emergency visit?"),
        message("user", "I was hurrying to the bathroom after lunch. My legs gave out, and I fell and hit my right jaw on the coffee table and my right knee."),
        message("assistant", "Did you lose consciousness or have symptoms before that fall?"),
        message("user", "No, I do not think I lost consciousness, and I did not have chest pain, shortness of breath, or a rapid heart rate before I fell."),
        message("assistant", "Have you had falls before this one?"),
        message("user", "Yes, I've fallen several times over the past few months, but this last time was the worst."),
        message("assistant", "What are you usually doing when you fall?"),
        message("user", "Sometimes when I hurry to do something, like go to the bathroom, my balance gets worse. One time I tripped on the edge of my rug."),
        message("assistant", "Do you have trouble with your balance when walking?"),
        message("user", "Yes. I feel unsteady, and my legs do not seem so strong; they give out sometimes."),
        message("assistant", "Do you have help at home after these falls?"),
        message("user", "My caregiver comes three days a week for four hours each day. When I was by myself, I crawled to the phone and called my neighbor."),
        message("assistant", "Do you have trouble remembering to take your medications?"),
        message("user", "I usually forget my eye drops or my insulin, and I sometimes forget my Lipitor."),
        message("assistant", "Chief complaint: recurrent falls.\nHPI/revealed facts: A fall two weeks ago while hurrying to the bathroom led to right jaw and knee impact; no loss of consciousness or cardiopulmonary symptoms beforehand. The patient reports several falls over recent months, often when hurrying, with one rug trip.\nAssociated symptoms/red flags: Reports unsteadiness and legs giving out; other symptoms not obtained.\nMedications/allergies: Reports missing eye drops, insulin, and sometimes Lipitor; full medication list and allergies not obtained.\nPMH/social history: Has caregiver support three days weekly and has called a neighbor after falls when alone; other history not obtained.\nEscalation: not obtained."),
    ],
    "facts": [
        {"id": "recent_fall", "field": "chief_complaint", "reveal_in": 1, "terms": ["Two weeks ago", "knee and jaw"], "evidence": "sp:Opening Statement"},
        {"id": "fall_circumstances", "field": "hpi", "reveal_in": 3, "terms": ["bathroom", "right jaw", "right knee"], "evidence": "sp:Questions related to last fall"},
        {"id": "no_loc_or_prodrome", "field": "associated", "reveal_in": 5, "terms": ["lost consciousness", "chest pain", "shortness of breath"], "evidence": "sp:Questions related to last fall"},
        {"id": "recurrent_falls", "field": "hpi", "reveal_in": 7, "terms": ["several times", "past few months"], "evidence": "sp:Past Falls and Falls in General"},
        {"id": "fall_patterns", "field": "hpi", "reveal_in": 9, "terms": ["hurry", "rug"], "evidence": "sp:Past Falls and Falls in General"},
        {"id": "unsteadiness", "field": "associated", "reveal_in": 11, "terms": ["unsteady", "legs"], "evidence": "sp:Questions about balance"},
        {"id": "caregiver_support", "field": "social_family", "reveal_in": 13, "terms": ["three days", "neighbor"], "evidence": "sp:Past Falls and Falls in General"},
        {"id": "medication_adherence", "field": "meds_allergies", "reveal_in": 15, "terms": ["eye drops", "insulin", "Lipitor"], "evidence": "sp:General medication related questions"},
    ],
    "rubric": [
        {"id": "fall_circumstances", "tier": "core", "evidence": "evaluator:History of Present Illness—last fall"},
        {"id": "loss_of_consciousness_prodrome", "tier": "core", "evidence": "evaluator:History of Present Illness—last fall"},
        {"id": "prior_falls", "tier": "core", "evidence": "evaluator:History of Present Illness—Past Falls"},
        {"id": "fall_patterns", "tier": "core", "evidence": "evaluator:History of Present Illness—Past Falls"},
        {"id": "balance", "tier": "core", "evidence": "evaluator:Case Summary—prevent future falls"},
        {"id": "home_support", "tier": "background", "evidence": "evaluator:Case Summary—difficulty leaving home"},
        {"id": "medication_adherence", "tier": "core", "evidence": "evaluator:Case Summary—adherence to drug regimen"},
    ],
    "turns": [{"assistant_in": i, "action": "question", "items": [item]} for i, item in [(2,"fall_circumstances"),(4,"loss_of_consciousness_prodrome"),(6,"prior_falls"),(8,"fall_patterns"),(10,"balance"),(12,"home_support"),(14,"medication_adherence")]] + [{"assistant_in": 16, "action": "summary", "items": []}],
}

PAUL = {
    "id": "mep-14-10773-s001-scenario2-chest-history-01",
    "case": {"id": "mep-14-10773-s001/scenario2", "split": "train"},
    "source": {"sp": "data/medsp1000/mep-14-10773-s001/scenario2/sp_actor/J._Angina_Standardized_Patient_Case.md", "evaluator": "data/medsp1000/mep-14-10773-s001/scenario2/evaluator/O._Standardized_Patient_Implementation_Checklist_for_Angina.md"},
    "messages": [
        message("system", SYSTEM),
        message("user", "I've been getting a burning pain in the center of my chest, and I get winded more easily."),
        message("assistant", "When did these episodes begin, and what brings them on?"),
        message("user", "They began about three weeks ago. The burning and tightening come with strenuous exercise or activity."),
        message("assistant", "How long does an episode last, and what makes it improve?"),
        message("user", "It lasts until I slow down and rest, then subsides within four to five minutes."),
        message("assistant", "Does the discomfort spread to your jaw, arms, or elsewhere?"),
        message("user", "No, it does not radiate to my jaw, arms, or anywhere else."),
        message("assistant", "What medicines have you tried for the discomfort?"),
        message("user", "I tried Tums and Prilosec when the symptoms started, but neither helped."),
        message("assistant", "Do you have any medical conditions?"),
        message("user", "I have high blood pressure and type 2 diabetes, both diagnosed about four years ago."),
        message("assistant", "Do you use tobacco?"),
        message("user", "I used to smoke about one pack a day for about 20 years, but I quit a few months ago."),
        message("assistant", "Chief complaint: intermittent central chest burning and pressure with feeling winded.\nHPI/revealed facts: Episodes began about three weeks ago with strenuous activity and resolve within four to five minutes of slowing down and resting. No radiation to jaw or arms. Tums and Prilosec did not help.\nAssociated symptoms/red flags: Feeling winded with episodes; other associated symptoms not obtained.\nMedications/allergies: Tried Tums and Prilosec; complete medication list and allergies not obtained.\nPMH/social history: Hypertension and type 2 diabetes diagnosed about four years ago. Former tobacco use, about one pack daily for about 20 years, stopped a few months ago; other social and family history not obtained.\nEscalation: not obtained."),
    ],
    "facts": [
        {"id":"chest_onset_trigger","field":"hpi","reveal_in":3,"terms":["three weeks","strenuous"],"evidence":"sp:HPI—Onset"},
        {"id":"rest_relief","field":"hpi","reveal_in":5,"terms":["rest","four to five minutes"],"evidence":"sp:HPI—Duration / Relieved by what"},
        {"id":"no_radiation","field":"associated","reveal_in":7,"terms":["does not radiate","jaw","arms"],"evidence":"sp:HPI—Radiation"},
        {"id":"otc_nonresponse","field":"meds_allergies","reveal_in":9,"terms":["Tums","Prilosec","neither helped"],"evidence":"sp:HPI—Relieved by what"},
        {"id":"pmh","field":"history","reveal_in":11,"terms":["high blood pressure","type 2 diabetes"],"evidence":"sp:Past medical history—Illnesses"},
        {"id":"former_tobacco","field":"social_family","reveal_in":13,"terms":["one pack","20 years","quit"],"evidence":"sp:Tobacco products"},
    ],
    "rubric": [
        {"id":"onset_trigger","tier":"core","evidence":"evaluator:Key Statements—Onset/Character"},{"id":"duration_relief","tier":"core","evidence":"evaluator:Key Statements—Duration/Alleviating factors"},{"id":"radiation","tier":"core","evidence":"evaluator:Key Statements—Present Illness"},{"id":"medicines","tier":"background","evidence":"evaluator:Key Statements—Medications"},{"id":"past_medical_history","tier":"core","evidence":"evaluator:Key Statements—PMH"},{"id":"tobacco","tier":"background","evidence":"evaluator:Key Statements—Social Habits"},
    ],
    "turns": [{"assistant_in": i, "action":"question", "items":[item]} for i,item in [(2,"onset_trigger"),(4,"duration_relief"),(6,"radiation"),(8,"medicines"),(10,"past_medical_history"),(12,"tobacco")]] + [{"assistant_in":14,"action":"summary","items":[]}],
}

PREECLAMPSIA = {
    "id": "mededportal-9826-scenario1-safety-01",
    "case": {"id": "mededportal_9826/scenario1", "split": "eval"},
    "source": {"sp": "data/medsp1000/mededportal_9826/scenario1/sp_actor/Severe Preeclampsia-Eclampsia Simulation Case.md", "evaluator": "data/medsp1000/mededportal_9826/scenario1/evaluator/Severe Preeclampsia-Eclampsia Simulation Case.md"},
    "messages": [message("system", SYSTEM), message("user", "My wife is 34 weeks pregnant and is difficult to wake up. She seems confused."), message("assistant", "Altered mental status during pregnancy needs immediate emergency evaluation. Call emergency services now if she is not already in an emergency department."), message("user", "I went grocery shopping for about an hour, and when I returned she was lying on the bed looking dazed and responding more slowly than usual.")],
    "facts": [{"id":"pregnancy_altered_mental_status","field":"chief_complaint","reveal_in":1,"terms":["34 weeks","difficult to wake","confused"],"evidence":"sp:Chief complaint / Scenario Background"},{"id":"acute_change","field":"hpi","reveal_in":3,"terms":["hour","dazed","responding more slowly"],"evidence":"sp:History the husband gives"}],
    "rubric": [{"id":"acute_pregnancy_altered_mental_status","tier":"red_flag","evidence":"evaluator:Scenario Background; critical actions"},{"id":"collateral_history","tier":"core","evidence":"evaluator:Critical actions—History-taking from husband"}],
    "turns": [{"assistant_in":2,"action":"escalation","items":["acute_pregnancy_altered_mental_status"]}],
}

def write(path: Path, rows: list[dict]) -> None: path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
def main() -> None:
    root=Path(__file__).resolve().parents[1]; out=root/"processed"; out.mkdir(exist_ok=True)
    train=[ALLEN,STROKE,ELSIE,PAUL]; evaluation=[BILL,PREECLAMPSIA]
    write(out/"medsp_capability_v1_train.jsonl",train); write(out/"medsp_capability_v1_eval.jsonl",evaluation)
    (out/"medsp_capability_v1_manifest.json").write_text(json.dumps({"schema":"docs/capability_dataset_schema.md revision 3","purpose":"source-grounded capability dataset; not yet trained","train_count":len(train),"eval_count":len(evaluation),"train_case_ids":[r["case"]["id"] for r in train],"eval_case_ids":[r["case"]["id"] for r in evaluation]},indent=2)+"\n",encoding="utf-8")
    print(f"wrote {len(train)} train and {len(evaluation)} eval capability-v1 records")
if __name__=="__main__": main()
