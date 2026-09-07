#!/usr/bin/env python3
"""Build 20 source-grounded single-turn intake examples from diverse MedSP cases.

Unlike the direct-pair pilot, these turns are concise authored renderings of
scenario facts and rubric targets. Every example retains a source citation and
is explicitly labeled as source-grounded rather than verbatim dialogue.
"""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM = (
    "Conduct a structured clinical intake interview. Ask one concise, relevant "
    "question at a time. Gather relevant history before summarizing. Do not diagnose "
    "or prescribe treatment."
)

# Each answer is limited to a fact explicitly present in the cited source.
CASES = [
    ("mededportal_10137/scenario1", "train", "The diarrhea is back. We have to figure out what is going on.", "When did the diarrhea return?", "I have been having recurrent diarrhea with abdominal pain.", "sp_actor/Dolores Pena - 58 yo female with Diarrhea.md"),
    ("mededportal_10020/scenario1", "train", "He's not right.", "When did you first notice that he was not acting like himself?", "For the last few weeks.", "sp_actor/Appendix F-Actor's Script.md"),
    ("mededportal_10055/scenario1", "train", "My mother was brought here with shortness of breath and fever.", "What medical problems does your mother have?", "She has progressive dementia and lives in an assisted-living residence.", "sp_actor/Medication allergy patient profile for standardized patient.md"),
    ("mededportal_10120/scenario1", "train", "My low back pain has gotten much worse.", "What makes the back pain worse?", "It is worst after I walk for more than five minutes.", "sp_actor/Musculoskeletal Workshop Series - Case A Mrs. Daniels - Spinal stenosis with myelopathy.md"),
    ("mededportal_10120/scenario2", "train", "I have had lower back pain for two days.", "What were you doing before the pain began?", "I had started a new Pilates workout routine a week earlier.", "sp_actor/Musculoskeletal Workshop Series - Case B Ms. Smith - Musculoskeletal Lumbar Pain.md"),
    ("mededportal_10120/scenario4", "train", "I have terrible right shoulder and neck pain.", "When did the pain begin?", "I noticed it when I woke up two days ago.", "sp_actor/Musculoskeletal Workshop Series - Case D Mr. Gupta - Neck Strain.md"),
    ("mededportal_10140/scenario1", "train", "I am having increasing pain after delivering my baby.", "Where is the pain located?", "It is on the left side of my labia.", "sp_actor/Four Obstetrics Cases.md"),
    ("mededportal_10140/scenario4", "train", "My IV has gone under the skin after my C-section.", "How are you feeling otherwise?", "I am doing well and tolerating a regular diet.", "sp_actor/Four Obstetrics Cases.md"),
    ("mededportal_10140/scenario5", "train", "My urine output has been low after surgery.", "How much urine have you made recently?", "Only about 40 cc over the last two hours.", "sp_actor/Four Gynecology Cases.md"),
    ("mededportal_10140/scenario7", "train", "I have low urine output after surgery for an ectopic pregnancy.", "Have you been feeling lightheaded or unusually sleepy?", "I am tired and it is difficult to stay awake.", "sp_actor/Four Gynecology Cases.md"),
    ("mededportal_10166/scenario1", "train", "I am here urgently because of chest pain.", "Where do you feel the pain?", "It is under my breastbone toward the left side.", "sp_actor/Roy Jones Case and SP Training Materials.md"),
    ("mededportal_10173/scenario1", "train", "My four-week-old has been eating less and is fussy.", "When did you first notice these changes?", "He was found to be hypothermic and bradycardic at his one-month visit today.", "sp_actor/History Physical.md"),
    ("mededportal_10248/scenario1", "train", "My husband is confused and his blood pressure is low.", "What medicines does he take?", "He takes metoprolol, Norvasc, and prednisone.", "sp_actor/Adrenal Crisis Simulation Guide.md"),
    ("mededportal_10256/scenario1", "train", "I am here to discuss a pelvic exam, but I am nervous about it.", "What has made pelvic examinations difficult for you in the past?", "I had a terrible experience during my last pelvic examination and have avoided gynecological care for years.", "sp_actor/Challenging Pelvic Exam Complete Case.md"),
    ("mededportal_10140/scenario8", "eval", "I have gas pain after my hysterectomy.", "Are you passing gas?", "Yes, I am passing flatus.", "sp_actor/Four Gynecology Cases.md"),
    ("mededportal_10166/scenario4", "eval", "I am here for follow-up after my heart attack.", "Have you had any chest pain since leaving the hospital?", "No, I have not had any more chest pain over the past three months.", "sp_actor/Roy Jones Case and SP Training Materials.md"),
    # Keep both gender variants in train: they share the same source document,
    # so using one as held-out evaluation would leak near-duplicate content.
    ("mededportal_10256/scenario2", "train", "I am here to discuss a pelvic exam, but I am nervous about it.", "What has made pelvic examinations difficult for you in the past?", "I had a terrible experience during my last pelvic examination and have avoided gynecological care for years.", "sp_actor/Challenging Pelvic Exam Complete Case.md"),
    ("mededportal_10258/scenario3", "eval", "I am worried that my other leg is getting darker.", "When did you first notice the skin changes?", "My legs started getting darker more than ten years ago.", "sp_actor/Case 3 Observation and Feedback (Derm Exam).md"),
    ("mededportal_10277/scenario1", "eval", "My eight-day-old is having trouble breathing and feeding less.", "When did the feeding and breathing problems begin?", "He is eight days old and has respiratory distress with decreased feeding.", "sp_actor/Simulation Template.md"),
    ("mededportal_10296/scenario1", "eval", "I feel weak and am bleeding heavily after my procedure.", "When did the bleeding start?", "It started after the uterine aspiration procedure for my miscarriage.", "sp_actor/Clinic Simulation.md"),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "data" / "medsp1000"
    processed = root / "processed"
    records = []
    for number, (case_id, split, opening, question, answer, source_file) in enumerate(CASES, start=1):
        source = source_root / case_id / source_file
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")
        assert question.count("?") == 1
        records.append(
            {
                "id": f"{case_id.replace('/', '__')}__source_grounded_{number:02d}",
                "source_case_id": case_id,
                "source_quality": "manually authored source-grounded single-turn pair",
                "source_file": source_file,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": opening},
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": answer},
                ],
            }
        )
    for split in ("train", "eval"):
        rows = [row for row, spec in zip(records, CASES) if spec[1] == split]
        (processed / f"medsp_expanded_v1_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    train_cases = {row["source_case_id"] for row, spec in zip(records, CASES) if spec[1] == "train"}
    eval_cases = {row["source_case_id"] for row, spec in zip(records, CASES) if spec[1] == "eval"}
    assert not train_cases & eval_cases
    (processed / "medsp_expanded_v1_manifest.json").write_text(
        json.dumps(
            {
                "method": "20 manually authored source-grounded single-turn pairs from downloaded MedSP scenarios",
                "train_count": len(train_cases),
                "eval_count": len(eval_cases),
                "train_case_ids": sorted(train_cases),
                "eval_case_ids": sorted(eval_cases),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train_cases)} train and {len(eval_cases)} eval expanded cases")


if __name__ == "__main__":
    main()
