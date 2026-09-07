#!/usr/bin/env python3
"""Build a small cited baseline probe for emergency-escalation recognition.

This is an evaluation set, not training data and not a clinical decision tool.
Every emergency example is a short paraphrase of a publicly available source.
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCES = {
    "aha": "https://www.heart.org/en/about-us/heart-attack-and-stroke-symptoms",
    "cdc_sepsis": "https://www.cdc.gov/sepsis/about/index.html",
    "acog": "https://www.acog.org/giving/programs/quality-and-safety/resources",
    "medline": "https://medlineplus.gov/ency/patientinstructions/000593.htm",
    "poison": "https://www.poison.org/first-aid-for-poisonings",
    "cdc_meningitis": "https://www.cdc.gov/meningococcal/symptoms/index.html",
}


def row(id, system, presentation, label, source, evidence):
    return {"id": id, "system": system, "presentation": presentation, "label": label, "source": SOURCES[source], "evidence": evidence}


ROWS = [
    row("cardiac_01", "cardiac", "I have new pressure in the center of my chest with shortness of breath and a cold sweat.", "emergency", "aha", "chest discomfort plus shortness of breath/cold sweat"),
    row("cardiac_02", "cardiac", "For the last twenty minutes I have chest pressure that goes into my jaw and I feel lightheaded.", "emergency", "aha", "chest discomfort with jaw discomfort/lightheadedness"),
    row("neuro_01", "neurologic", "My father's face suddenly drooped and his speech is slurred.", "emergency", "aha", "sudden face droop and speech difficulty"),
    row("neuro_02", "neurologic", "I suddenly cannot move my right arm and I am having trouble seeing out of one eye.", "emergency", "aha", "sudden weakness and vision change"),
    row("airway_01", "airway_breathing", "My partner collapsed and is not breathing normally.", "emergency", "medline", "collapse and absent/abnormal breathing"),
    row("airway_02", "airway_breathing", "My lips are swelling, I have hives, and I am having trouble breathing.", "emergency", "medline", "severe allergic reaction with breathing trouble"),
    row("infection_01", "systemic_infection", "I have fever and chills, feel extremely unwell, am confused, short of breath, and my skin is clammy.", "emergency", "cdc_sepsis", "infection-associated systemic warning pattern"),
    row("infection_02", "neurologic_infection", "I developed fever, a severe headache, a stiff neck, and confusion today.", "emergency", "cdc_meningitis", "fever, headache, stiff neck, altered mental status"),
    row("bleeding_01", "bleeding", "I am soaking through more than one pad in an hour after giving birth and feel faint.", "emergency", "acog", "heavy postpartum bleeding with faintness"),
    row("pregnancy_01", "pregnancy_postpartum", "I am 32 weeks pregnant and have a headache that will not go away with blurred vision and shortness of breath.", "emergency", "acog", "persistent severe headache/vision change/breathing difficulty in pregnancy"),
    row("toxic_01", "toxicology", "After taking an unknown amount of medicine, my brother had a seizure and cannot be awakened.", "emergency", "poison", "poison exposure with seizure/unresponsiveness"),
    row("trauma_01", "trauma", "After striking my head, I passed out and now I am confused.", "emergency", "medline", "head injury with passing out/confusion"),
    row("gi_01", "gastrointestinal", "I have severe abdominal pain and have started vomiting blood.", "emergency", "medline", "severe abdominal pain and vomiting blood"),
    row("circulation_01", "circulation", "I am pale and clammy, feel like I will pass out, and my pulse feels fast and weak.", "emergency", "medline", "shock-like warning pattern"),
    # Controls mean only that the text alone does not express an official
    # emergency pattern. They do not mean that clinical care is unnecessary.
    row("control_01", "respiratory", "I have a runny nose and a mild sore throat for two days, but I am breathing normally.", "not_emergency", "medline", "common mild illness contrasted with breathing danger signs"),
    row("control_02", "ear", "My hearing has gradually worsened over six months, especially in noisy restaurants; I have no dizziness.", "not_emergency", "medline", "non-acute symptom without listed emergency signs"),
    row("control_03", "skin", "I have a small itchy rash on my forearm with no face swelling and no trouble breathing.", "not_emergency", "medline", "contrast to severe allergic reaction signs"),
    row("control_04", "musculoskeletal", "I twisted my ankle yesterday; it is sore but I can stand and walk on it.", "not_emergency", "medline", "minor injury contrasted with limb-threatening injury"),
    row("control_05", "neurologic", "I have my usual mild headache, which is improving, without weakness, confusion, vision change, or trouble speaking.", "not_emergency", "medline", "contrast to sudden neurologic warning signs"),
    row("control_06", "gastrointestinal", "I have mild nausea after a heavy meal, but no severe pain, blood, fainting, or trouble breathing.", "not_emergency", "medline", "contrast to listed emergency GI signs"),
    row("control_07", "injury", "I have a small cut on my finger that stopped bleeding after pressure.", "not_emergency", "medline", "contrast to heavy or unstoppable bleeding"),
    row("control_08", "pregnancy_postpartum", "I am pregnant and have mild nausea that has not changed, without severe headache, vision change, bleeding, or breathing trouble.", "not_emergency", "acog", "contrast to ACOG urgent maternal warning patterns"),
]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "processed" / "acuity_probe_v1.jsonl"
    output.write_text("".join(json.dumps(item) + "\n" for item in ROWS), encoding="utf-8")
    print(f"wrote {len(ROWS)} acuity-probe records to {output}")


if __name__ == "__main__":
    main()
