# Archaeo-Resonance-Pipeline

**Multisensor archaeological detection and resonance analysis**

## Overview
This project combines geophysical, LIDAR, magnetic, EM, spectral, and symbolic data to identify likely archaeological sites using feature fusion, harmonic analysis, and machine learning. Designed for non-invasive site detection.

## Quick Start
```bash
git clone <repo_url>
cd archaeo-resonance-pipeline
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

Repository Structure

See /src for core processing modules and /streamlit_app for interactive visualization.

Data

Small example dataset in /examples/sample_tile

Known sites in /examples/known_sites.geojson

Custom user data can be uploaded through the Streamlit app (obey ethical permissions).


Ethics & Permissions

All methods are non-invasive until explicit consent (EarthConsent = True)

Respect local communities and indigenous custodians

Track permissions and chain-of-custody for all site data


Contributing

Follow [CODE_OF_CONDUCT.md]

Submit PRs for improvements

Run tests before committing (pytest tests/)


License

MIT — see LICENSE file.

---

## LICENSE (MIT template)
```text
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy...
