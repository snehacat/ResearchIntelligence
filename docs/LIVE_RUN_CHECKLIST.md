# Live Validation Run Checklist

This checklist provides exact commands to run the three-paper validation with your own GROQ_API_KEY on Windows PowerShell.

## Prerequisites

1. **Set GROQ_API_KEY environment variable**
   ```powershell
   $env:GROQ_API_KEY = "your_actual_api_key_here"
   ```

2. **Navigate to project directory**
   ```powershell
   cd C:\Users\sneha\Projects\researchintelligence
   ```

3. **Activate virtual environment** (if not already activated)
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. **Verify environment variable is set**
   ```powershell
   echo $env:GROQ_API_KEY
   ```
   Should display your API key (do not share this output).

---

## Step 1: Run Three-Paper Validation

Run the validation harness for all three bundled papers:

```powershell
python validation/step5/run_validation.py --confirm-costs
```

**What this does:**
- Validates all three PDFs in `validation/step5/papers/`:
  - attention_is_all_you_need.pdf
  - bert_pretraining.pdf
  - attention_survey.pdf
- Sends selected excerpts to Groq (openai/gpt-oss-20b model)
- Waits 10 seconds between papers (rate limit protection)
- Creates a new timestamped run directory in `validation/step5/runs/`

**Where the output lands:**
- Run directory: `validation/step5/runs/YYYYMMDDTHHMMSSffffffZ/`
- Inside the run directory:
  - `validation_report.md` - Summary report
  - `evidence_audit.csv` - Citation evidence audit
  - `results/` - Per-paper analysis results
  - `raw_outputs/` - Raw LLM responses

**Expected output:**
- Progress messages for each paper
- Completion summary with token counts and field coverage
- Total runtime (~2-3 minutes for three papers)

---

## Step 2: Identify the Run Directory

After the validation completes, note the run directory name:

```powershell
Get-ChildItem validation/step5/runs/ | Sort-Object LastWriteTime -Descending | Select-Object -First 1 Name
```

This will show the most recent run directory, e.g., `20261110T123456789012Z`.

---

## Step 3: Run Metrics Report

Pass the run directory to the metrics script:

```powershell
python evaluation/report_metrics.py --gold-dir evaluation/gold_set/ --run-dir validation/step5/runs/YOUR_RUN_DIRECTORY
```

Replace `YOUR_RUN_DIRECTORY` with the actual run directory name from Step 2.

**Example:**
```powershell
python evaluation/report_metrics.py --gold-dir evaluation/gold_set/ --run-dir validation/step5/runs/20261110T123456789012Z
```

**What this does:**
- Loads gold annotations from `evaluation/gold_set/`
- Loads validation results from the specified run directory
- Computes metrics (Token-F1, Set-F1, hit-rate@k, MRR, citation traceability)
- Prints n_papers and n_fields
- Shows warning if n_papers < 5
- Labels all metrics as surface-form proxies

**Where the output appears:**
- Printed to console (stdout)
- No file is written by this script

---

## Step 4: Review Validation Report

Open the validation report:

```powershell
notepad validation/step5/runs/YOUR_RUN_DIRECTORY/validation_report.md
```

**What to check:**
- Execution status (succeeded/failed for each paper)
- Schema status (valid/invalid)
- Extraction coverage (how many fields extracted)
- Evidence resolution (how many citations resolved)
- Quality review status (should be "pending")

---

## Step 5: Review Evidence Audit

Open the evidence audit CSV:

```powershell
notepad validation/step5/runs/YOUR_RUN_DIRECTORY/evidence_audit.csv
```

**What to check:**
- Citation entries for each field
- Evidence pages overlap
- Sufficiency labels (sufficient/partial/insufficient)

---

## Cost Considerations

**Estimated cost for three-paper run:**
- Groq model: openai/gpt-oss-20b
- Estimated tokens per paper: ~2,000 prompt + ~1,500 completion = ~3,500 tokens
- Three papers: ~10,500 tokens total
- Cost: Typically < $0.01 (check Groq pricing for current rates)

**The `--confirm-costs` flag:**
- Requires you to confirm before making any API calls
- Prevents accidental charges
- Shows estimated cost if available

---

## Troubleshooting

### Error: "GROQ_API_KEY not set"

**Fix:**
```powershell
$env:GROQ_API_KEY = "your_actual_api_key_here"
```

### Error: "ModuleNotFoundError"

**Fix:**
```powershell
pip install -r requirements.txt
```

### Error: "HTTP 400" or "Failed to generate JSON"

**Possible causes:**
- Output cap too low (should be 2048 tokens)
- Model unavailable
- Rate limit exceeded

**Fix:**
- Check Groq status
- Wait and retry after rate limit period
- Verify MAX_GENERATED_TOKENS = 2048 in analysis/paper_analyzer.py

### Error: "PDF has no extractable text"

**Possible causes:**
- PDF is scanned images (not text-based)
- PDF is corrupted

**Fix:**
- Use text-based PDFs only
- Re-download the PDF if corrupted

---

## After the Run

1. **Human review required**
   - Review extracted values in `results/*.json`
   - Check if citations actually support the claims
   - Do not trust automated metrics alone

2. **Fill gold annotations**
   - Follow docs/ANNOTATION_GUIDE.md
   - Annotate from PDFs BEFORE viewing model output
   - Run metrics script again with filled gold set

3. **Archive the run**
   - Run directories are preserved automatically
   - Do not delete old runs
   - Each run has a unique timestamp

---

## Important Notes

- **Do not share your GROQ_API_KEY** - keep it secret
- **Do not commit API keys** to the repository
- **Human review is required** - metrics are surface-form proxies only
- **Run directories are preserved** - do not delete them
- **Rate limits apply** - the harness waits 10 seconds between papers
