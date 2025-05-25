import requests, os, pandas as pd, tqdm
import xml.etree.ElementTree as ET

congress, session = 118, 1
start_roll, end_roll = 1, 352     
output_dir = "./data/senate-votes"

os.makedirs(output_dir, exist_ok=True)
all_votes = []
vote_summaries = []

def get_int(elem, tag, default=0):
    if elem is None:
        return default
    child = elem.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return int(child.text.strip())
    except ValueError:
        return default


for roll in tqdm.tqdm(range(start_roll, end_roll+1)):
    url = (
        f"https://www.senate.gov/legislative/LIS/roll_call_votes/"
        f"vote{congress}{session}/vote_{congress}_{session}_{roll:05d}.xml"
    )
    try:
        r = requests.get(url, timeout=5)
    except requests.RequestException:
        continue

    if r.status_code != 200 or "<roll_call_vote" not in r.text:
        continue

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        continue

    def get_text(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    count = root.find("count")
    summary = {
        "roll_number":      get_text("vote_number"),
        "vote_date":        get_text("vote_date"),
        "title":            get_text("vote_title"),
        "question":         get_text("vote_question_text"),
        "result":           get_text("vote_result_text"),
        "yeas":             get_int(count, "yeas"),
        "nays":             get_int(count, "nays"),
        "present":          get_int(count, "present"),
        "absent":           get_int(count, "absent"),
    }
    vote_summaries.append(summary)

    for m in root.findall("./members/member"):
        all_votes.append({
            "roll_number":   summary["roll_number"],
            "member_full":   (m.findtext("member_full")  or "").strip(),
            "first_name":    (m.findtext("first_name")    or "").strip(),
            "last_name":     (m.findtext("last_name")     or "").strip(),
            "party":         (m.findtext("party")         or "").strip(),
            "state":         (m.findtext("state")         or "").strip(),
            "lis_member_id": (m.findtext("lis_member_id") or "").strip(),
            "vote_cast":     (m.findtext("vote_cast")     or "").strip(),
        })

pd.DataFrame(vote_summaries).to_csv(
    os.path.join(output_dir,"senate_vote_summaries.csv"), index=False
)
pd.DataFrame(all_votes).to_csv(
    os.path.join(output_dir,"senate_roll_call_votes.csv"), index=False
)

print("Done.")
