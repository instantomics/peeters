"""Archive a release of peeters on Zenodo: a new version of the same record, so the concept
DOI (in CITATION.cff) stays the same and every version gets a DOI of its own.

    python scripts/zenodo_release.py 0.1.2              # publish (dist.sh does this)
    python scripts/zenodo_release.py 0.1.2 --draft      # everything but the publish
    python scripts/zenodo_release.py --discard <id>     # delete an unpublished draft

The token is ZENODO_TOKEN (dist.sh reads ~/.secrets/zenodo.env); ZENODO_URL points at
another Zenodo (https://sandbox.zenodo.org). The first release, while CITATION.cff has no
`doi:`, makes a new record and prints its concept DOI: put that in CITATION.cff and the
README. A published version cannot be deleted, only followed by a newer one.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/") + "/api"


def call(method: str, url: str, body=None, raw: bytes | None = None):
    if not url.startswith("http"):
        url = API + url
    headers = {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if raw is not None:
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            text = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo {method} {url}: {e.code} {e.read().decode(errors='replace')[:800]}")
    return json.loads(text) if text else None


def concept_doi() -> str | None:
    m = re.search(r'^doi:\s*"?(10\.5281/zenodo\.\d+)"?\s*$',
                  (ROOT / "CITATION.cff").read_text(), flags=re.M)
    return m.group(1) if m else None


def metadata(version: str) -> dict:
    meta = json.loads((ROOT / ".zenodo.json").read_text())
    meta["license"] = meta.get("license", "mit").lower()  # the API's licence ids
    meta["version"] = version
    meta["publication_date"] = datetime.date.today().isoformat()
    meta["related_identifiers"] = [
        {"identifier": f"https://pypi.org/project/peeters/{version}/",
         "relation": "isIdenticalTo", "scheme": "url"},
        {"identifier": f"https://github.com/instantomics/peeters/tree/v{version}",
         "relation": "isSupplementTo", "scheme": "url"},
    ] + [r for r in meta.get("related_identifiers", []) if r["scheme"] == "doi"]
    return meta


def draft_for(version: str) -> dict:
    """A draft: a new version of the concept's latest record, or a new record."""
    doi = concept_doi()
    if doi is None:
        print("CITATION.cff has no doi: a new record (its concept DOI comes with the publish)")
        return call("POST", "/deposit/depositions", {})
    concept = doi.rsplit(".", 1)[1]
    latest = call("GET", f"/records/{concept}/versions/latest")
    made = call("POST", f"/deposit/depositions/{latest['id']}/actions/newversion")
    draft = call("GET", made["links"]["latest_draft"])
    for f in draft.get("files", []):  # a new version starts with the last one's files
        call("DELETE", f["links"]["self"])
    return call("GET", draft["links"]["self"])


def main(argv: list[str]) -> None:
    if argv[:1] == ["--discard"]:
        call("DELETE", f"/deposit/depositions/{argv[1]}")
        print(f"draft {argv[1]} deleted")
        return
    version, publish = argv[0].lstrip("v"), "--draft" not in argv
    files = sorted((ROOT / "dist").glob(f"peeters-{version}*"))
    if not files:
        sys.exit(f"no dist/peeters-{version}* to archive: build first")
    draft = draft_for(version)
    for f in files:
        call("PUT", f"{draft['links']['bucket']}/{f.name}", raw=f.read_bytes())
        print(f"uploaded {f.name}")
    call("PUT", draft["links"]["self"], {"metadata": metadata(version)})
    if not publish:
        print(f"draft {draft['id']} ready, not published: {draft['links']['html']}\n"
              f"delete it with: python scripts/zenodo_release.py --discard {draft['id']}")
        return
    done = call("POST", draft["links"]["publish"])
    print(f"published peeters {version}: doi {done['doi']} (concept {done['conceptdoi']})")


if __name__ == "__main__":
    main(sys.argv[1:])
