# Image Pipeline V2 (E2E, 5-class @ 512x512)

Implements the updated pipeline requirements including:
- YAML config, logging, CSV→class mapping
- Preprocessing (resize to 512×512, CLAHE contrast, normalize), save pipeline `.pkl`
- Save **processed images to class folders**
- **simpleModel** CNN (5 classes) and training/eval with plots
- Save model wrapper to `.pkl`
- Bulk/single run to CSV (filename, predicted_class, severity, confidence)
- Streamlit app (single + ZIP), scrollable results + CSV download
