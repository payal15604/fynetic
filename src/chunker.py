import re
from typing import List

def sentence_split(text: str) -> List[str]:
    # simple sentence split; preserves abbreviations decently
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]

def chunk_by_words_preserving_sentences(
    text: str,
    chunk_words: int = 180,
    overlap_words: int = 40
) -> List[str]:
    if not text or len(text.strip()) < 50:
        return []
    sents = sentence_split(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_len + w > chunk_words:
            chunks.append(" ".join(cur))
            # overlap: keep last N words ≈ overlap_words by sentence units
            tail = []
            tlen = 0
            for t in reversed(cur):
                tw = len(t.split())
                if tlen + tw >= overlap_words: 
                    tail.insert(0, t); break
                tail.insert(0, t); tlen += tw
            cur, cur_len = tail, sum(len(t.split()) for t in tail)
        cur.append(s); cur_len += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks
