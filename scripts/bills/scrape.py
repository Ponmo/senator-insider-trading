import requests
import os
import pandas as pd
import re
import tqdm

start, end = 100, 724

def get_url(roll):
    return f"https://clerk.house.gov/evs/2023/roll{roll}.xml"

def fetch(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.text
    except:
        return None

def parse_tag(block, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    return m.group(1).strip() if m else None

def normalize_vote_label(vote):
    vote = vote.strip()
    if vote in ("Aye", "Yea", "Yes"):
        return "Yes"
    if vote in ("No", "Nay"):
        return "No"
    if vote == "Present":
        return "Present"
    if vote == "Not Voting":
        return "Not Voting"
    return vote 

if __name__ == "__main__":
    os.makedirs("../../data/bills", exist_ok=True)

    all_votes   = []
    bill_rows   = []

    totals_pat  = re.compile(r"<totals-by-party>(.*?)</totals-by-party>", re.DOTALL)
    rec_pat     = re.compile(r"<recorded-vote>(.*?)</recorded-vote>", re.DOTALL)
    leg_pat     = re.compile(r'name-id="([^"]+)".*?party="([^"]+)".*?state="([^"]+)".*?>([^<]+)</legislator>')
    vote_pat    = re.compile(r"<vote>([^<]+)</vote>")

    for roll in tqdm.tqdm(range(start, end+1)):

        xml = fetch(get_url(roll))
        if not xml:
            print(f"✖ could not fetch roll {roll}")
            continue

        md = {}
        mdb = re.search(r"<vote-metadata>(.*?)</vote-metadata>", xml, re.DOTALL)
        if mdb:
            md_block = mdb.group(1)
            for t in ("majority","congress","session","chamber",
                      "rollcall-num","legis-num","vote-question",
                      "amendment-num","amendment-author","vote-type",
                      "vote-result","action-date","vote-desc"):
                md[t.replace("-","_")] = parse_tag(md_block, t)
            m = re.search(r'<action-time time-etz="([^"]+)">([^<]+)</action-time>', md_block)
            md["action_time_etz"] = m.group(1) if m else None
            md["action_time"]     = m.group(2) if m else None
        else:
            for k in ("majority","congress","session","chamber",
                      "rollcall_num","legis_num","vote_question",
                      "amendment_num","amendment_author","vote_type",
                      "vote_result","action_date","vote_desc",
                      "action_time_etz","action_time"):
                md[k] = None

        counts = { "Republican":{}, "Democratic":{}, "Independent":{} }
        vt = re.search(r"<vote-totals>(.*?)</vote-totals>", xml, re.DOTALL)
        if vt:
            for part_block in totals_pat.finditer(vt.group(1)):
                b = part_block.group(1)
                party = parse_tag(b, "party")
                if party in counts:
                    counts[party] = {
                        "yea":        int(parse_tag(b, "yea-total")     or 0),
                        "nay":        int(parse_tag(b, "nay-total")     or 0),
                        "present":    int(parse_tag(b, "present-total") or 0),
                        "not_voting": int(parse_tag(b, "not-voting-total") or 0),
                    }

        vd = re.search(r"<vote-data>(.*?)</vote-data>", xml, re.DOTALL)
        if vd:
            for rec in rec_pat.finditer(vd.group(1)):
                chunk = rec.group(1)
                lm = leg_pat.search(chunk)
                vm = vote_pat.search(chunk)
                if lm and vm:
                    label = normalize_vote_label(vm.group(1))
                    all_votes.append({
                        "roll_number":     roll,
                        "name_id":         lm.group(1),
                        "party":           lm.group(2),
                        "state":           lm.group(3),
                        "legislator_name": lm.group(4).strip(),
                        "vote":            label
                    })

        row = {"roll_number": roll}
        row.update(md)
        for p in ("Republican","Democratic","Independent"):
            c = counts[p]
            key = p.lower()
            row[f"{key}_yeas"]       = c.get("yea",0)
            row[f"{key}_nays"]       = c.get("nay",0)
            row[f"{key}_present"]    = c.get("present",0)
            row[f"{key}_not_voting"] = c.get("not_voting",0)

        bill_rows.append(row)

    pd.DataFrame(all_votes)  \
      .to_csv("../../data/bills/congress_roll_call_votes.csv", index=False)
    pd.DataFrame(bill_rows)  \
      .to_csv("../../data/bills/congress_roll_call_bill_details.csv", index=False)

    print("Done")
