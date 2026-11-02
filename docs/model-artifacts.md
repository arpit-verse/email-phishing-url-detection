# Model artifact manifest

The public release omits binary model files. The API downloads only these exact public files from the Sentinel Hugging Face repository and validates the full file bytes against the listed SHA-256 before loading.

Repository: https://huggingface.co/arpit-verse/sentinel-phishing-models

| Local runtime path | Immutable file URL | Revision | SHA-256 | Verified local size |
|---|---|---|---|---:|
| models/email_pipeline.pkl | https://huggingface.co/arpit-verse/sentinel-phishing-models/resolve/88693cab826ef3c9041908e2f9d14d25a824ec52/email_pipeline.pkl | 88693cab826ef3c9041908e2f9d14d25a824ec52 | 8f36987f042631ee6831486ef94d16d942f53e70302efc37b3a366e28317884b | 717,124 bytes |
| models/url_random_forest.pkl | https://huggingface.co/arpit-verse/sentinel-phishing-models/resolve/de19ff7841af7ce3f96d2324fa25cc863a4cc3e8/url_random_forest.pkl | de19ff7841af7ce3f96d2324fa25cc863a4cc3e8 | db3729b9ad3c4b2d069e0f34348b736b00a525b807f838c7fd1c1f9924ee62d3 | 98,534,153 bytes |

The hashes and byte sizes were recomputed from the local artifacts on 2026-11-02. The pinned revisions and expected hashes match api/main.py. Startup streams downloads to temporary files in models/, rejects a Git LFS pointer and any hash mismatch, stages and verifies both files before replacing either model path, then calls the existing predictor loaders.

These are pickle/joblib artifacts. Load only the pinned, hash-verified objects. A matching hash establishes byte identity, not dataset provenance, legal rights, or model quality. Public model availability does not establish training-data redistribution rights.