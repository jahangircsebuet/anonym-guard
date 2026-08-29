## Multilingual Translation Pipeline

We generate multilingual prompt translations using the `translate_languages()` function in `src/translate.py`. The function uses an NLLB translation model to translate each source prompt into a target language and then back-translate it into the source language for quality checking.

For each target language, the script adds a `translation` field to every row:

```json
{
  "translation": {
    "Hausa": {
      "prompt_translated_lang": "...",
      "prompt_back_to_original_lang": "...",
      "response_translated_lang": "...",
      "response_back_to_original_lang": "...",
      "model_name": "facebook/nllb-200-3.3B"
    }
  }
}
```

For AEGIS-style input files, the function reads:

```text
prompt
response
```

For other input files, it reads:

```text
agg_prompt_bn
agg_response_bn
```

The translation function supports checkpoint-style resuming. If the output file already exists, the script loads the existing translated rows and skips any language that is already complete. It also saves progress after each completed language when `save_every_language=True`.

Example usage:

```python
from translate import translate_languages

stats = translate_languages(
    input_path=f"{PROJECT_ROOT}/data/raw/aegis_source.json",
    output_path=f"{PROJECT_ROOT}/data/translated/aegis_translated_98_languages.json",
    languages=target_languages,
    lang_name_to_code=lang_name_to_code,
    src_lang_code="eng_Latn",
    model_name="facebook/nllb-200-3.3B",
    gpu_device=0,
    batch_size=64,
    sent_batch_size=32,
    max_new_tokens=256,
    save_every_language=True,
    dataset="Aegis",
    show_progress=True,
)
```

The function returns a summary dictionary:

```text
num_rows
languages_completed
languages_skipped
output_path
total_seconds
```

In GuardBreach, these translated and back-translated prompts are later scored using BERTScore and COMET, assigned to translation-quality buckets, and used to construct the stratified `k=1`, `k=2`, and `k=3` benchmark subsets.