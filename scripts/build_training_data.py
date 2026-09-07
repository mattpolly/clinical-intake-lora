#!/usr/bin/env python3
"""Build a small, source-grounded intake dataset from inspected MedSP1000 cases.

This intentionally does not use an LLM to invent conversations. Each patient
answer and each question target is manually traced to an inspected SP script or
evaluator checklist. Facts are revealed only in the patient turn that follows
the corresponding clinician question.
"""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM = (
    "Conduct a structured clinical intake interview. Ask one concise, relevant "
    "question at a time. Gather relevant history before summarizing. Do not diagnose "
    "or prescribe treatment. When concerning symptoms are disclosed, recommend an "
    "appropriate level of urgent evaluation."
)


def case(
    case_id: str,
    split: str,
    source_files: list[str],
    opening: str,
    pairs: list[tuple[str, str]],
    summary: str,
    fixed_prefix: int,
) -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "source_files": source_files,
        "opening": opening,
        "pairs": pairs,
        "summary": summary,
        "fixed_prefix": fixed_prefix,
    }


CASES = [
    case(
        "mededportal_9045/scenario1",
        "eval",
        [
            "sp_actor/Pat Andares Headache Resource.md",
            "evaluator/Pat Andares Headache Resource.md",
        ],
        "I have a really bad headache.",
        [
            ("When did this new headache begin?", "It started gradually yesterday afternoon after a very stressful day at work."),
            ("Where do you feel the headache most?", "It is worst in the back of my head and neck, though it feels like it involves my whole head."),
            ("How would you describe the pain?", "It feels like something is banging inside my head when I move it."),
            ("How severe is it on a scale from 0 to 10?", "It is a 10 out of 10."),
            ("How is this different from your prior headaches?", "I have had milder headaches after stressful workdays before, but I have never had one like this."),
            ("Have you had any changes in your vision?", "Last night I lost sight in both eyes for about 10 to 15 seconds. Everything went black, then it resolved."),
            ("Because this is a severe new headache with brief vision loss, please seek emergency evaluation now. Is someone available to take you to the emergency department?", "My son drove me here this morning because I did not feel well."),
            ("Have you noticed neck stiffness?", "Yes, my neck feels stiff."),
            ("Have you had a fever or chills?", "No fever or chills."),
            ("Have you had recent head trauma?", "No."),
            ("Do you have high blood pressure or other medical conditions?", "I have high blood pressure and high cholesterol. I do not see my primary doctor regularly."),
            ("What medicines do you take?", "I take captopril and hydrochlorothiazide only when I think I need them. I sometimes take Lasix too."),
            ("Do you have any medication allergies?", "No allergies."),
            ("Do you smoke or drink alcohol?", "I do not smoke. I usually have one beer most evenings."),
            ("Are there important health conditions in your family?", "Both of my parents had high blood pressure, and my mother died from complications of diabetes."),
        ],
        "Summary: You reported a new, severe 10/10 headache that began yesterday and is worst in the back of your head and neck. You described neck stiffness and a brief episode of vision loss. You also reported hypertension with inconsistent medication use, no allergies, no fever or chills, and a family history of hypertension and diabetes. Because of the severe new headache and vision change, urgent emergency evaluation is appropriate.",
        8,
    ),
    case(
        "mededportal_9118/scenario1",
        "train",
        [
            "sp_actor/Victor Billar - Acute Abdominal Pain - Resource.md",
            "evaluator/Victor Billar - Acute Abdominal Pain - Resource.md",
        ],
        "I have had this awful pain in my abdomen for several hours now.",
        [
            ("When did the pain start?", "It started about four hours ago."),
            ("Where is the pain located?", "It is in my right upper abdomen and the upper middle part of my abdomen."),
            ("Does the pain move anywhere?", "It travels to my right shoulder."),
            ("How severe is the pain on a scale from 0 to 10?", "It is a 9 or 10 out of 10."),
            ("Is the pain constant or does it come and go?", "It is steady and constant now."),
            ("What was happening before the pain started?", "It began about an hour after I ate a rich, fatty meal."),
            ("What makes the pain worse?", "Walking or moving makes it worse."),
            ("Have you tried anything for it?", "I took Aleve 220 milligrams, but it did not help."),
            ("Have you had nausea or vomiting?", "I feel nauseated and vomited once about an hour ago."),
            ("Have you had fever or chills?", "The nurse said I had a fever, but I have not felt feverish or had chills."),
            ("Because you have severe abdominal pain with vomiting and a reported fever, you need urgent in-person evaluation. Are you able to get to an emergency department now?", "Yes, I can get help getting there."),
            ("Have you had similar pain before?", "I have had milder cramping pain in the same area after fatty foods for about three years, but it usually went away within an hour."),
            ("Have you had abdominal surgery or hospitalizations?", "No surgeries and no hospitalizations."),
            ("Do you have any relevant family history?", "My father has high blood pressure, and I do not know of gallbladder disease in my family."),
            ("Do you smoke, drink alcohol, or use recreational drugs?", "I have never smoked, have several beers during the week, and do not use recreational drugs."),
        ],
        "Summary: You reported constant 9–10/10 right-upper and upper-central abdominal pain that started four hours ago after a fatty meal and radiates to your right shoulder. Movement worsens it and Aleve did not help. You reported nausea, one episode of vomiting, and a reported fever, with prior milder meal-related episodes. You should receive urgent in-person evaluation.",
        11,
    ),
    case(
        "mededportal_10373/scenario1",
        "train",
        [
            "sp_actor/A. Cough Case.md",
            "evaluator/B. Cough Case Standardized Patient Master Encounter Checklist.md",
        ],
        "I can't seem to get rid of this cough.",
        [
            ("Can you tell me more about the cough?", "It is a deep, hacking cough that started about three days ago and has been getting worse."),
            ("Are you bringing up any phlegm?", "Yes, yellow phlegm, but only with a lot of effort."),
            ("Did you have symptoms before the cough began?", "I had a terrible shaking chill about four days ago and intermittent chills since then."),
            ("Do you get short of breath?", "I get short of breath with activity, like climbing stairs, but not at rest."),
            ("Do you have chest discomfort?", "The right lower side of my chest hurts when I cough or take a big breath."),
            ("Have you measured a fever?", "I feel feverish but have not checked my temperature."),
            ("Have you been around anyone who was ill?", "I volunteer at a rescue mission where people have been coughing and sneezing, and my grandson recently had strep throat."),
            ("Because you have worsening cough, feverishness, chest pain with breathing, and shortness of breath, you should have prompt in-person evaluation. Are you having trouble breathing at rest right now?", "No, not at rest."),
            ("Do you have congestion or a runny nose?", "No congestion or runny nose."),
            ("Have you had wheezing?", "No wheezing."),
            ("What have you taken for the cough?", "I have been taking pseudoephedrine and guaifenesin, but they have not helped."),
            ("Do you have any medical conditions or regular medicines?", "No regular medicines or medical conditions. I have environmental allergies in the spring."),
            ("Do you smoke or use nicotine?", "I used to smoke about a pack a day, but quit two weeks ago and am using a nicotine patch."),
        ],
        "Summary: You reported a worsening deep cough for three days with yellow phlegm, preceded by chills. You have exertional shortness of breath, right-sided pain when coughing or taking a deep breath, and possible fever. You also reported exposure to sick contacts and recent smoking cessation. Prompt in-person evaluation is appropriate.",
        8,
    ),
    case(
        "mep_2374-8265.10866-s001/scenario1",
        "eval",
        [
            "sp_actor/E. Chest Pain.md",
            "evaluator/E. Chest Pain.md",
        ],
        "I have a heavy feeling in my chest and I feel short of breath.",
        [
            ("When did the chest discomfort start?", "It started about 15 minutes ago while I was sitting in a chair watching television."),
            ("Where do you feel the discomfort?", "It is in the center of my chest."),
            ("How would you describe it?", "It feels like heaviness or pressure."),
            ("How severe is it on a scale from 0 to 10?", "It is about a 5 out of 10."),
            ("Does it spread anywhere else?", "No, it does not radiate."),
            ("Do you have nausea, sweating, or shortness of breath?", "I am mildly nauseated and short of breath, and I feel sweaty."),
            ("Chest pressure with shortness of breath and sweating can be an emergency. Please call emergency services now rather than driving yourself. Are you alone?", "No, I am in the hospital ward."),
            ("Does it change when you take a deep breath?", "No."),
            ("Does it change with position?", "No."),
            ("Have you had a similar feeling before?", "I have had similar heaviness and shortness of breath with climbing one flight of stairs, but it usually lasts only a few minutes."),
            ("What heart or vascular conditions do you have?", "I have diabetes, coronary artery disease with prior stents, heart failure, and a prior abdominal aortic aneurysm repair."),
        ],
        "Summary: You reported new central chest heaviness at rest for about 15 minutes, rated 5/10, with shortness of breath, sweating, and mild nausea. It does not radiate or change with breathing or position. You have diabetes, known coronary artery disease with prior stents, and heart failure. This needs emergency evaluation now.",
        7,
    ),
    case(
        "mep-12-10488-s001/scenario1",
        "train",
        [
            "sp_actor/G. Dyspnea Case Standardized Patient.md",
            "sp_actor/F. Dyspnea Case Storyboard.md",
        ],
        "I'm really short of breath. I could barely make it back from the bathroom.",
        [
            ("When did the shortness of breath begin?", "I was feeling pretty good when I went to sleep at 9:30, but when I woke at 11:30 I could barely make it to the bathroom."),
            ("What makes the shortness of breath worse?", "Doing anything makes it worse."),
            ("What makes it better?", "Nothing has helped."),
            ("Have you had shortness of breath like this before?", "No, I have not had shortness of breath like this before."),
            ("Do you have chest pain or pressure?", "No chest pain or pressure."),
            ("Do you have cough or wheezing?", "No cough or wheezing."),
            ("Severe worsening shortness of breath needs immediate bedside assessment. Are you able to speak in full sentences right now?", "Not really. It is getting harder to breathe."),
            ("Have you had fever or chills?", "No fever or chills."),
            ("Do you have a history of heart problems?", "Not that I know of. I do not see a doctor much."),
            ("What medicines do you normally take?", "I take aspirin or ibuprofen a couple of times a week for knee pain."),
            ("Do you have any allergies?", "No known allergies, including to latex or IV contrast."),
            ("Do you smoke or drink alcohol?", "I do not smoke now. I smoked for a few years in my twenties and have a beer or two on weekends."),
        ],
        "Summary: You reported sudden severe shortness of breath that woke you from sleep and worsens with any activity, with no relief. You reported no prior similar episodes, no chest pain, cough, wheezing, fever, or chills. You also reported intermittent NSAID use and no known allergies. Because you are struggling to speak in full sentences, you need immediate bedside assessment.",
        7,
    ),
]


def ordered_pairs(spec: dict, variant: int) -> list[tuple[str, str]]:
    """Vary only non-critical, already source-grounded history order."""
    pairs = spec["pairs"]
    prefix = pairs[: spec["fixed_prefix"]]
    tail = pairs[spec["fixed_prefix"] :]
    if not tail:
        return prefix
    shift = variant % len(tail)
    return prefix + tail[shift:] + tail[:shift]


def make_conversation(spec: dict, variant: int) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": spec["opening"]},
    ]
    for question, answer in ordered_pairs(spec, variant):
        messages.extend(
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": answer},
            ]
        )
    messages.append({"role": "assistant", "content": spec["summary"]})
    return {
        "id": f"{spec['case_id'].replace('/', '__')}__v{variant}",
        "source_case_id": spec["case_id"],
        "source_files": spec["source_files"],
        "messages": messages,
    }


def validate(record: dict) -> None:
    messages = record["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    for index, message in enumerate(messages[2:], start=2):
        expected = "assistant" if index % 2 == 0 else "user"
        assert message["role"] == expected, (record["id"], index, message)
        if message["role"] == "assistant" and not message["content"].startswith("Summary:"):
            assert message["content"].count("?") == 1, message["content"]
        assert "diagnosis" not in message["content"].lower()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [make_conversation(spec, variant) for spec in CASES for variant in range(5)]
    for record in records:
        validate(record)

    train = [record for record, spec in zip(records, [spec for spec in CASES for _ in range(5)]) if spec["split"] == "train"]
    evaluation = [record for record, spec in zip(records, [spec for spec in CASES for _ in range(5)]) if spec["split"] == "eval"]
    train_cases = {record["source_case_id"] for record in train}
    evaluation_cases = {record["source_case_id"] for record in evaluation}
    assert not train_cases & evaluation_cases
    assert 20 <= len(records) <= 100

    write_jsonl(output_dir / "medsp_intake_train.jsonl", train)
    write_jsonl(output_dir / "medsp_intake_eval.jsonl", evaluation)
    (output_dir / "medsp_intake_manifest.json").write_text(
        json.dumps(
            {
                "dataset_repo": "byrLLCC/MedSP1000",
                "dataset_revision": "55e3e55efd08c73baab912ba0c5b42637114fbc8",
                "conversation_count": len(records),
                "train_count": len(train),
                "eval_count": len(evaluation),
                "train_case_ids": sorted(train_cases),
                "eval_case_ids": sorted(evaluation_cases),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train)} train and {len(evaluation)} evaluation conversations")


if __name__ == "__main__":
    main()
