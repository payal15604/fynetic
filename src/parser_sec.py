import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup

ITEM_TITLE_MAP = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis (MD&A)", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary"
}

def clean_text(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    txt = soup.get_text(separator="\n")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in txt.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def _clean_section_content(content: str) -> str:
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\.{3,}\s*\d+', '', content)  # toc dots
    content = re.sub(r'^(FORM 10-K|10-K|ANNUAL REPORT)\s*', '', content, flags=re.IGNORECASE)
    return content.strip()

def split_sections(text: str) -> List[Dict[str, str]]:
    # robust item detection incl. NBSP
    patterns = [
        r'(?:^|\n)\s*ITEM[\s\u00A0]+([0-9]+[A-Z]?)[.\s:\-]',
        r'(?:^|\n)\s*Item[\s\u00A0]+([0-9]+[A-Z]?)[.\s:\-]',
        r'(?:^|\n)\s*item[\s\u00A0]+([0-9]+[A-Z]?)[.\s:\-]'
    ]
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            matches.append({"item_num": m.group(1).upper(), "start": m.start(), "end": m.end()})

    # unique by earliest occurrence, then sort
    best = {}
    for m in matches:
        k = m["item_num"]
        if k not in best or m["start"] < best[k]["start"]:
            best[k] = m
    ordered = sorted(best.values(), key=lambda x: x["start"])

    if not ordered:
        # fallback: simple split
        parts = re.split(r"(Item\s+[0-9A-Za-z\.]+)", text, flags=re.IGNORECASE)
        sections = []
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            if len(content) >= 100:
                sections.append({"section": header, "text": _clean_section_content(content)})
        return sections

    sections = []
    for i, m in enumerate(ordered):
        start = m["end"]
        end = ordered[i+1]["start"] if i+1 < len(ordered) else len(text)
        content = text[start:end].strip()
        if len(content) < 100:
            continue
        content = _clean_section_content(content)
        item = m["item_num"]
        name = ITEM_TITLE_MAP.get(item, "Unknown Section")
        sections.append({"section": f"Item {item} - {name}", "item_number": item, "text": content})
    return sections

def parse_filing_record(raw: dict) -> Dict[str, Any]:
    """
    raw: { cik, accession, company, filing_type, filing_date, text }
    returns: { meta, sections }
    """
    cleaned = clean_text(raw.get("text", ""))
    sections = split_sections(cleaned)
    return {
        "meta": {
            "cik": raw.get("cik", ""),
            "accession": raw.get("accession", ""),
            "company": raw.get("company", ""),
            "filing_type": raw.get("filing_type", ""),
            "filing_date": raw.get("filing_date", "")
        },
        "sections": sections
    }
