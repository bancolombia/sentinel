import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_uses_resolvable_citations():
    paper_text = (ROOT / "paper" / "paper.md").read_text(encoding="utf-8")
    bib_text = (ROOT / "paper" / "paper.bib").read_text(encoding="utf-8")

    citation_keys = re.findall(r"@\w+\{([^,]+),", bib_text)
    keys_in_paper = set(re.findall(r"@\[(.*?)\]", paper_text))

    # The paper may contain multiple citations in a single bracketed group.
    resolved = set()
    for group in keys_in_paper:
        for key in re.split(r"\s*,\s*", group.strip()):
            if key:
                resolved.add(key)

    missing = sorted(resolved - set(citation_keys))
    assert not missing, f"Missing citation keys in paper.bib: {missing}"
