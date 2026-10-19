# Ayaz ASR — training recipe (v0.1)

Training recipe of **Ayaz ASR v0.1**, a streaming, multilingual speech recognition model with improved Turkish, fine-tuned
from [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) on a single
Apple M4 Mac. Model: **https://huggingface.co/VeliAyaz/ayaz-asr-tr-streaming-0.6b**

**Ayaz ASR v0.1**'in eğitim tarifi: Türkçesi geliştirilmiş, akışlı, çok dilli konuşma tanıma modeli; tek bir Apple M4 Mac'te
nvidia/nemotron-3.5-asr-streaming-0.6b'den ince ayarla eğitildi. Model: **https://huggingface.co/VeliAyaz/ayaz-asr-tr-streaming-0.6b**

**Status:** these scripts were ported and cleaned from the private scripts used for v0.1 and have not yet been run end to
end in this form. Expect to fix small issues; please report them.

**Durum:** Bu betikler v0.1'de kullanılan özel betiklerden aktarılıp temizlendi; bu biçimleriyle henüz baştan sona
çalıştırılmadı. Küçük sorunlar çıkabilir; lütfen bildirin.

Code: Apache-2.0 (`LICENSE`). This repository contains **no** model weights, audio, transcripts or manifests (`NOTICE`).
The model weights are published separately under OpenMDW-1.1. Models trained with this recipe derive from the base model
and are subject to OpenMDW-1.1.

Kod: Apache-2.0. Bu depoda model ağırlığı, ses, metin ya da manifest **yoktur**. Model ağırlıkları ayrıca OpenMDW-1.1
lisansıyla yayımlanmıştır. Bu tarifle eğitilen modeller temel modelden türer ve OpenMDW-1.1'e tabidir.

[English](#english) · [Türkçe](#türkçe)

---

## English

### What is in here

| Path | Purpose |
|---|---|
| `data/yodas_index.py` | Index of the Turkish part of YODAS (OWSM v4 cleaned release): byte ranges of every Turkish utterance |
| `data/yodas_download.py` | Downloads manually subtitled Turkish videos (byte ranges only), 16 kHz, cuts ≤ 12 s clips along subtitle lines |
| `data/prepare_eval_sets.py` | FLEURS Turkish test manifest; YODAS development split (30 videos, video level) and training pool |
| `data/build_turkish_manifest.py` | Turkish part of the mixture: ≤ 4 clips per YODAS video (+ optional non-distributed data) |
| `data/build_rehearsal.py` | Rehearsal for 32 other languages (FLEURS train split, labels = the base model's own output) and guard sets (FLEURS dev) |
| `data/filter_rehearsal.py` | Drops rehearsal labels with a foreign script or letter loops |
| `data/build_mixture.py` | Final training mixture (fixed-seed shuffle; optional additional non-distributed manifests) |
| `train/train.py` | Fine-tuning: partial unfreezing, L2-SP anchor, periodic measurement, step selection, resume, verdict |
| `eval/fleurs_multilingual.py` | FLEURS regression check over 34 languages with bootstrap intervals and the multilingual verdict |
| `export/` | sherpa-onnx int8 export (560 ms chunks): exporter, int8 quantisation, metadata and package checks |
| `tools/clean_nemo_config.py` | Removes local manifest paths from a trained `.nemo` before publishing |
| `configs/v0.1.yaml` | Every setting of the published run — documentation only, no script reads it. The CLI defaults are the same values except three: `eval/fleurs_multilingual.py --parallel` defaults to 1 (reproduce.sh passes 3); `train.py --short-test-padding` is a flag; `build_rehearsal.py --max-label-tokens` is computed from `--turkish-manifest` unless given (v0.1: 127) |
| `reproduce.sh` | Steps 1–9 below in order with the v0.1 settings (steps 10–11 are manual) |
| `ayaz_asr/` | Shared helpers (scoring normalisation incl. digits-to-words, metrics, NeMo transcription, FLEURS download) |

### Method in short

* **Base model:** nvidia/nemotron-3.5-asr-streaming-0.6b (NeMo `EncDecRNNTBPEModelWithPrompt`), language prompt `tr-TR`,
  streaming context 560 ms (`[56, 6]`) for every measurement.
* **Trained parameters:** the last 8 encoder layers, the prediction network, the joint network and the language-prompt
  module (prompt kernel) — 230,370,336 parameters according to the v0.1 training log; the rest of the encoder (including
  its input subsampling) is frozen and kept in eval mode.
* **Optimisation:** AdamW, learning rate 3·10⁻⁵, betas (0.9, 0.98), weight decay 10⁻³, fp32, 30 s of audio per batch
  (lhotse, 6 buckets, quadratic duration 15), RNNT loss on the CPU (numba). 9,000 steps, measured every 1,000; the
  published checkpoint is **step 8,000**, chosen by the best balance score because no step met all interim criteria.
* **Against forgetting:** (1) rehearsal — FLEURS train-split clips of 32 languages labelled with the base model's own
  output (self-distillation), German and English twice; (2) **L2-SP anchoring** toward the original weights in discrete
  form: every 10 steps `θ ← θ + k(θ₀ − θ)` with `k = 1 − (1 − 1/τ)^10`, τ = 4,500 steps (λ = 1/(lr·τ) ≈ 7.41); the
  anchor is stored in fp16 on the CPU.
* **Labels:** sentence-final language tags (`… . <tr-TR>`), digits spelled out as Turkish words.
* **Hardware:** one Apple M4 Mac (24 GB), PyTorch MPS backend. Only Apple MPS was used for v0.1. CPU (`--device cpu`)
  and CUDA (`--device cuda`) are untested (train.py only; labelling and evaluation use MPS if available, else CPU). Run time: about 9 h in total (≈ 7.4 h training incl. interim measurements +
  ≈ 1.8 h multilingual verdict). Data preparation is mostly network bound.

### Scoring normalisation

The numbers on the model card were computed with this normalisation (`ayaz_asr/text.py`):

* **Turkish (WER):** strip language tags; Unicode NFC; digits spelled out as Turkish words, with an apostrophe suffix
  attached to the last word (`3'ün` → `üçün`); Turkish casing (`I`→`ı`, `İ`→`i`, then lower case); apostrophes `'` and
  `’` deleted; remaining non-word characters → space. The training labels also have digits spelled out.
* **Other languages:** language tag dropped, NFKC + casefold, punctuation and symbols → space, combining marks kept;
  WER on words, CER on characters ignoring spaces.

### Data

| Source | Use in v0.1 | Licence | In this repo |
|---|---|---|---|
| YODAS Turkish, cleaned OWSM v4 release (`espnet/yodas_owsmv4`, yodas0.10) [6–8]; list of manually subtitled videos from `espnet/yodas2` | 14,387 clips, 24.6 h, 3,848 videos; labels = the videos' manually written subtitles | CC-BY-3.0 | scripts to download and select (selection may differ slightly between runs, see Known gaps) |
| FLEURS (`google/fleurs`) [9, 10] train split, 32 languages | rehearsal: 12,649 clips, 28.0 h; labels = base model output | CC-BY-4.0 | scripts |
| FLEURS dev / test splits | guard sets, Turkish test, multilingual verdict | CC-BY-4.0 | scripts |
| Developer's everyday speech | 215 clips | developer's own voice | **not distributed** |
| Voice messages from volunteers | 881 clips, 0.9 h, 80 messages, 11 speakers | written consent | **not distributed** |
| Developer's short-utterance / command recordings | 331 utterances, 5.7 min | developer's own voice | **not distributed** |
| Empty (silence / noise) examples | 3,285 clips, 1.7 h | — | **not distributed** |

The non-distributed data was labelled with two teacher models — Whisper large-v3-turbo [11, 12] (MIT) and Qwen3-ASR [13]
(Apache-2.0) — keeping only clips on which both agreed; that labelling pipeline is not part of this repository.

**v0.1 cannot be reproduced exactly from this repository.** It also used consented volunteer voice messages and the
developer's own recordings, which are not distributed. With public data alone, or with your own data, results will differ
from the published numbers, especially on the Turkish tests. Even with identical data, runs are not bit-for-bit identical
(MPS, download order). With public data only, the Turkish part of the mixture is YODAS alone.

The scripts accept your own data through optional arguments (`build_turkish_manifest.py --speech/--empty/--noise`,
`build_mixture.py --extra PATH:COPIES`, `train.py --indomain-test/--short-test/--echo-test/...`); the copy counts used
for v0.1 are in `configs/v0.1.yaml`.

Manifests are NeMo JSON lines: `audio_filepath`, `duration`, `text` (training label, with language tags; empty for
empty examples), `target_lang`/`lang` (e.g. `tr-TR`), and for test sets `raw_text` (reference). Test sets for
`train.py` may carry `has_text` (short-utterance set: score only clips with text).

### Acceptance criteria (as on the model card)

1. **Turkish improvement** (in-training verdict on the selected step): ≥ 10 % relative WER improvement on everyday
   speech (in-domain test); at most +0.5 points on the full FLEURS Turkish test; at most 2 empty outputs on the
   short-utterance set; at most +1.5 points on the German and English guard sets. `train.py` additionally runs an
   echo-empty check: the share of empty outputs on a set of clips that must stay empty may drop by at most 5 points.
   The verdict needs `--short-test`, `--echo-test` and `--noise-test` together (plus `--indomain-test`); without all
   three, the short-utterance, echo-empty and in-domain ≥ 10 % interim checks are skipped in step selection and the
   verdict is reported as not passed.
2. **No regression beyond limits on the other FLEURS languages** (step 9: `eval/fleurs_multilingual.py summary --full
   --model … --verdict NAME`): at
   most +1.0 point on German and English; median over the 29 main languages at most +2.0; no language above +5.0
   except Thai (the base model does not work for Thai either, CER ≈ 99 %).

v0.1 passed the multilingual verdict and three of the four Turkish criteria; the short-utterance criterion depends on
the measurement (2 empty outputs unpadded, 4 with silence padding). See the model card for all results.

### Reproduction

Requirements: Python with `requirements.txt` (NeMo 3.0.0), `curl`, an Apple Silicon Mac (MPS), at least 15 GB free
disk for training (resume file ≈ 3.7 GB, model ≈ 2.6 GB) plus room for the downloaded audio. All paths default to a
working directory (`--work-dir`, or `$AYAZ_ASR_WORK`, or `./work`).

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export AYAZ_ASR_WORK=$PWD/work
# base model: put nemotron-3.5-asr-streaming-0.6b.nemo (from the NVIDIA model repo) into $AYAZ_ASR_WORK/models/
bash reproduce.sh            # steps 1–9 below, v0.1 settings
```

Steps 1–9 are run by `reproduce.sh`; steps 10–11 are manual.

1. `python3 train/train.py env` — environment check (MPS allocation, versions).
2. `python3 data/yodas_index.py` then `python3 data/yodas_download.py` — YODAS Turkish index and clips.
3. Download `data/tr_tr/test.tsv` and `data/tr_tr/audio/test.tar.gz` of `google/fleurs` into
   `$AYAZ_ASR_WORK/datasets/fleurs/tr_tr/`, then `python3 data/prepare_eval_sets.py`.
4. `python3 data/build_turkish_manifest.py` (+ optional non-distributed data).
5. `python3 eval/fleurs_multilingual.py download --parallel 3` and `... decode` — base model on 100 test recordings
   per language (the rehearsal label check compares against these values).
6. `python3 data/build_rehearsal.py download`, then
   `... label --turkish-manifest $AYAZ_ASR_WORK/manifests/turkish.jsonl --max-label-tokens 127`, then
   `python3 data/filter_rehearsal.py`. Defaults are the v0.1 settings: 300 recordings per language, 1,300 for German
   and English, 600 for Latvian, Lithuanian, Estonian and Romanian, label-quality margin 0.12, 15–32 h. Replaying the
   script's selection logic on the v0.1 labelling statistics gives the same per-language counts (12,680 rows, 28.08 h;
   12,649 after the filter). A real run can differ if the FLEURS files or the base model's outputs change.
7. `python3 data/build_mixture.py` (+ `--extra PATH:COPIES` for your own data).
8. `python3 train/train.py run` (+ your own test sets) — writes `summary_*.json`, `steps_*.csv` and `model_*.nemo` to
   `$AYAZ_ASR_WORK/train/`.
9. `python3 eval/fleurs_multilingual.py download --full --parallel 3`, then `decode --full --model "NAME|LABEL|PATH|1|VALUE"`
   and `summary --full --model ... --verdict NAME`, where VALUE is the FLEURS-tr subset value of the selected step from
   the training summary (a sanity check that the evaluation decodes like training: ±0.002 on the 150-clip subset).
10. Export: `REFERENCE=<official sherpa-onnx package of the base model> bash export/export_sherpa_onnx.sh model.nemo out_dir`.
11. Before publishing a `.nemo`: `python3 tools/clean_nemo_config.py model.nemo model.clean.nemo --check-restore`.

### Known gaps and limitations

* **v0.1 cannot be reproduced exactly** (see Data). With public data only, the Turkish part is YODAS alone and the
  in-training verdict cannot be computed in full (it needs the in-domain, short-utterance, echo and noise sets).
* With public data only, `reproduce.sh` uses the YODAS dev set as `--indomain-test`, so it drives step selection. Its
  result is then not an unbiased held-out number and is not comparable to the card's YODAS row.
* The torch, lightning, lhotse, numba and Python versions of the v0.1 environment were not recorded (only NeMo 3.0.0
  and, for export, onnx 1.23.0 / onnxruntime 1.28.0 are known).
* The YODAS manifest is written in download-completion order (parallel threads); the seeded per-video sampling depends
  on that order, so the exact selection of YODAS clips can differ between runs.
* FLEURS archives are read in their published order; if the dataset files change upstream, the rehearsal selection
  changes with them.
* A `.nemo` saved by `train.py` contains your local manifest paths in `model_config.yaml` (`train_ds`) — clean it with
  `tools/clean_nemo_config.py` before publishing a model.
* Model limitations (from the model card): short utterances remain weak (about 77–80 % error); narrow speaker
  diversity, part of the Turkish gain comes from a single speaker, the developer; intra-sentence code-switching not measured; some
  Turkish-specific characters leak into other languages' output; the int8 package was tested live only; most YODAS
  content is lectures, so numbers may be written as digits.

### Citation

Please cite the model (BibTeX from the model card):

```bibtex
@misc{ayaz2026ayazasr,
  author       = {Ayaz, Veli},
  title        = {Ayaz ASR: streaming, multilingual speech recognition with improved Turkish, version 0.1},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/VeliAyaz/ayaz-asr-tr-streaming-0.6b}
}
```

This work is not endorsed or supported by NVIDIA.

### References

1. NVIDIA. nvidia/nemotron-3.5-asr-streaming-0.6b — model card and files. Hugging Face, 2026.
2. OpenMDW License Agreement, version 1.1. https://openmdw.ai/license/1-1/
3. McCloskey M, Cohen NJ. Catastrophic interference in connectionist networks: the sequential learning problem. Psychology of Learning and Motivation 24, 1989.
4. Robins A. Catastrophic forgetting, rehearsal and pseudorehearsal. Connection Science 7(2), 1995.
5. Li X, Grandvalet Y, Davoine F. Explicit inductive bias for transfer learning with convolutional networks. ICML 2018.
6. Li X, Takamichi S, Saeki T, Chen W, Shiota S, Watanabe S. YODAS: YouTube-oriented dataset for audio and speech. IEEE ASRU 2023. arXiv:2406.00899.
7. Peng Y, et al. OWSM v4: improving open Whisper-style speech models via data scaling and cleaning. arXiv:2506.00338, 2025.
8. ESPnet. espnet/yodas_owsmv4 — dataset card. https://huggingface.co/datasets/espnet/yodas_owsmv4
9. Conneau A, et al. FLEURS: few-shot learning evaluation of universal representations of speech. IEEE SLT 2022.
10. Google. google/fleurs — dataset card and files. https://huggingface.co/datasets/google/fleurs
11. Radford A, et al. Robust speech recognition via large-scale weak supervision. ICML 2023.
12. OpenAI. openai/whisper-large-v3-turbo — model card. https://huggingface.co/openai/whisper-large-v3-turbo
13. Qwen. Qwen/Qwen3-ASR-1.7B — model card. https://huggingface.co/Qwen/Qwen3-ASR-1.7B
14. NVIDIA-NeMo. NeMo v3.0.0. https://github.com/NVIDIA-NeMo/NeMo/releases/tag/v3.0.0
15. NVIDIA. How to fine-tune Nemotron 3.5 ASR for your language, domain, or accent. Hugging Face Blog, 2026. https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
16. k2-fsa. sherpa-onnx. https://github.com/k2-fsa/sherpa-onnx (`export/export_onnx.py` is derived from its
    `scripts/nemo/nemotron-3.5-asr-streaming-0.6b/export_onnx.py`, Apache-2.0)

---

## Türkçe

### İçerik

| Yol | Ne işe yarar |
|---|---|
| `data/yodas_index.py` | YODAS'ın (OWSM v4 temizlenmiş sürüm) Türkçe dizini: her Türkçe sözcenin bayt aralığı |
| `data/yodas_download.py` | Elle alt yazılı Türkçe videoları (yalnız bayt aralıklarıyla) indirir, 16 kHz'e çevirir, alt yazı satırlarına göre ≤ 12 sn parçalar |
| `data/prepare_eval_sets.py` | FLEURS Türkçe sınav manifesti; YODAS geliştirme kümesi (30 video, video düzeyinde) ve eğitim havuzu |
| `data/build_turkish_manifest.py` | Karışımın Türkçe kısmı: YODAS videosu başına en çok 4 parça (+ isteğe bağlı dağıtılmayan veri) |
| `data/build_rehearsal.py` | 32 öteki dilin provası (FLEURS eğitim bölümü, etiket = temel modelin kendi çıktısı) ve koruma kümeleri (FLEURS geliştirme) |
| `data/filter_rehearsal.py` | Dile yabancı yazı ya da harf döngüsü içeren prova etiketlerini çıkarır |
| `data/build_mixture.py` | Son eğitim karışımı (sabit tohumla karıştırma; isteğe bağlı dağıtılmayan veri manifestleri) |
| `train/train.py` | İnce ayar: kısmi açma, L2-SP çapası, aralıklı ölçüm, adım seçimi, kaldığı yerden sürme, hüküm |
| `eval/fleurs_multilingual.py` | 34 dilde FLEURS gerileme ölçümü (önyükleme aralıklarıyla) ve çok dilli hüküm |
| `export/` | sherpa-onnx int8 dışa aktarma (560 ms parça): dışa aktarıcı, int8 nicemleme, üst veri ve paket denetimi |
| `tools/clean_nemo_config.py` | Yayından önce eğitilmiş `.nemo` içindeki yerel manifest yollarını temizler |
| `configs/v0.1.yaml` | Yayımlanan koşunun bütün ayarları — yalnız belgedir, hiçbir betik okumaz. Komut satırı varsayılanları üç yer dışında aynıdır: `eval/fleurs_multilingual.py --parallel` varsayılanı 1'dir (reproduce.sh 3 verir); `train.py --short-test-padding` bir bayraktır; `build_rehearsal.py --max-label-tokens` verilmezse `--turkish-manifest`'ten hesaplanır (v0.1: 127) |
| `reproduce.sh` | Aşağıdaki 1–9. adımlar sırayla, v0.1 ayarlarıyla (10–11. adımlar elle yapılır) |
| `ayaz_asr/` | Ortak yardımcılar (rakamların sözcüğe çevrilmesi dahil puanlama normalleştirmesi, ölçüler, NeMo çözümleme, FLEURS indirme) |

### Yöntem

* **Temel model:** nvidia/nemotron-3.5-asr-streaming-0.6b (NeMo `EncDecRNNTBPEModelWithPrompt`), dil istemi `tr-TR`,
  her ölçümde 560 ms akış bağlamı (`[56, 6]`).
* **Eğitilen kısım:** kodlayıcının son 8 katmanı, tahmin ağı, ortak katman ve dil istemi modülü (prompt kernel) — v0.1
  eğitim kütüğüne göre 230.370.336 parametre; kodlayıcının kalanı (giriş alt örneklemesi dahil) donuk ve değerlendirme
  kipinde.
* **Eniyileme:** AdamW, öğrenme oranı 3·10⁻⁵, betas (0,9, 0,98), ağırlık azaltma 10⁻³, fp32, parti başına 30 sn ses
  (lhotse, 6 kova, karesel süre 15), RNNT kaybı işlemcide (numba). 9.000 adım, her 1.000 adımda ölçüm; yayımlanan model
  **8.000. adım** (hiçbir adım ara ölçütlerin hepsini karşılamadığı için en iyi denge puanıyla seçildi).
* **Unutmaya karşı:** (1) prova — 32 dilin FLEURS eğitim parçaları, temel modelin kendi çıktısıyla etiketlenmiş (öz
  damıtma), Almanca ve İngilizce iki kez; (2) özgün ağırlıklara **L2-SP çapası**, ayrık biçimde: her 10 adımda
  `θ ← θ + k(θ₀ − θ)`, `k = 1 − (1 − 1/τ)^10`, τ = 4.500 adım (λ = 1/(lr·τ) ≈ 7,41); çapa işlemcide fp16 tutulur.
* **Etiketler:** cümle sonunda dil etiketi (`… . <tr-TR>`), rakamlar Türkçe sözcükle yazılır.
* **Donanım:** tek Apple M4 Mac (24 GB), PyTorch MPS. v0.1'de yalnız Apple MPS kullanıldı. İşlemci (`--device cpu`) ve
  CUDA (`--device cuda`) denenmedi (yalnız train.py için; etiketleme ve değerlendirme MPS varsa onu, yoksa işlemciyi
  kullanır). Süre: toplam yaklaşık 9 sa (≈ 7,4 sa eğitim, ara ölçümler dahil + ≈ 1,8 sa çok dilli
  hüküm). Veri hazırlığı çoğunlukla ağ hızına bağlıdır.

### Puanlama normalleştirmesi

Model kartındaki sayılar bu normalleştirmeyle hesaplandı (`ayaz_asr/text.py`):

* **Türkçe (WER):** dil etiketleri atılır; Unicode NFC; rakamlar Türkçe sözcükle yazılır, kesme işaretli ek son sözcüğe
  yapışır (`3'ün` → `üçün`); Türkçe büyük/küçük harf (`I`→`ı`, `İ`→`i`, sonra küçük harf); `'` ve `’` silinir; kalan
  sözcük dışı karakterler boşluk olur. Eğitim etiketlerinde de rakamlar sözcükle yazılmıştır.
* **Öteki diller:** dil etiketi atılır, NFKC + casefold, noktalama ve simgeler boşluk olur, birleşik işaretler korunur;
  WER sözcüklerle, CER boşluklar sayılmadan karakterlerle.

### Veri

| Kaynak | v0.1'de kullanım | Lisans | Bu depoda |
|---|---|---|---|
| YODAS Türkçe, OWSM v4 temizlenmiş sürümü (`espnet/yodas_owsmv4`, yodas0.10) [6–8]; elle alt yazılı video listesi `espnet/yodas2`'den | 14.387 parça, 24,6 sa, 3.848 video; etiket = videoların elle yazılmış alt yazıları | CC-BY-3.0 | indirme ve seçme betikleri (seçim koşudan koşuya biraz değişebilir, Bilinen eksikler'e bakın) |
| FLEURS (`google/fleurs`) [9, 10] eğitim bölümü, 32 dil | prova: 12.649 parça, 28,0 sa; etiket = temel modelin çıktısı | CC-BY-4.0 | betikler |
| FLEURS geliştirme / sınav bölümleri | koruma kümeleri, Türkçe sınav, çok dilli hüküm | CC-BY-4.0 | betikler |
| Geliştiricinin gündelik konuşması | 215 parça | geliştiricinin kendi sesi | **dağıtılmaz** |
| Gönüllülerin sesli mesajları | 881 parça, 0,9 sa, 80 mesaj, 11 konuşmacı | yazılı rıza | **dağıtılmaz** |
| Geliştiricinin kısa söz / komut kayıtları | 331 söz, 5,7 dk | geliştiricinin kendi sesi | **dağıtılmaz** |
| Boş (sessizlik / gürültü) örnekler | 3.285 parça, 1,7 sa | — | **dağıtılmaz** |

Dağıtılmayan veri iki öğretmen modelle — Whisper large-v3-turbo [11, 12] (MIT) ve Qwen3-ASR [13] (Apache-2.0) —
etiketlendi; yalnız ikisinin anlaştığı parçalar kullanıldı. Bu etiketleme hattı depoda yok.

**v0.1 bu depodan tam olarak yeniden üretilemez.** v0.1 ayrıca gönüllülerin rızayla verdiği sesli mesajları ve
geliştiricinin kendi kayıtlarını kullandı; bunlar dağıtılmaz. Yalnız herkese açık veriyle ya da kendi verinizle elde
edilen sonuçlar, özellikle Türkçe sınavlarda, yayımlanan sayılardan farklı olur. Aynı veriyle bile koşular bit düzeyinde
aynı değildir (MPS, indirme sırası). Yalnız herkese açık veriyle karışımın Türkçe kısmı yalnız YODAS'tan oluşur.

Betikler kendi verinizi isteğe bağlı argümanlarla alır (`build_turkish_manifest.py --speech/--empty/--noise`,
`build_mixture.py --extra YOL:KAT`, `train.py --indomain-test/--short-test/--echo-test/...`); v0.1'de kullanılan kopya
sayıları `configs/v0.1.yaml`'dadır.

Manifestler NeMo JSON satırlarıdır: `audio_filepath`, `duration`, `text` (eğitim etiketi, dil etiketleriyle; boş
örneklerde boş), `target_lang`/`lang` (ör. `tr-TR`) ve sınav kümelerinde `raw_text` (başvuru metni). `train.py`'nin sınav
kümeleri `has_text` taşıyabilir (kısa söz kümesi: yalnız metinli parçalar puanlanır).

### Kabul ölçütleri (model kartındaki gibi)

1. **Türkçe iyileşme** (seçilen adımda eğitim içi hüküm): gündelik konuşmada (alan içi sınav) ≥ %10 göreli WER
   iyileşmesi; FLEURS Türkçe sınavının tamamında en çok +0,5 puan; kısa söz kümesinde en çok 2 boş çıktı; Almanca ve
   İngilizce koruma kümelerinde en çok +1,5 puan. `train.py` ayrıca bir boş-kalma denetimi yapar: boş kalması gereken bir
   kümede boş çıktı payı en çok 5 puan düşebilir. Hüküm için `--short-test`, `--echo-test` ve `--noise-test` birlikte
   (ve `--indomain-test`) gerekir; üçü birden verilmezse kısa söz, boş-kalma ve alan içi ≥ %10 ara
   denetimleri adım seçiminde atlanır ve hüküm "geçmedi" olarak raporlanır.
2. **Öteki FLEURS dillerinde sınırı aşan gerileme yok** (9. adım: `eval/fleurs_multilingual.py summary --full
   --model … --verdict AD`): Almanca ve
   İngilizcede en çok +1,0 puan; 29 ana dilin ortancası en çok +2,0; Tayca dışında hiçbir dil +5,0'ın üstünde değil
   (temel model Taycada da çalışmıyor, CER ≈ %99).

v0.1 çok dilli hükmü ve dört Türkçe ölçütün üçünü geçti; kısa söz ölçütü ölçüme bağlıdır (dolgusuz 2, sessizlik
dolgulu 4 boş çıktı). Bütün sonuçlar için model kartına bakın.

### Yeniden üretim

Gerekenler: `requirements.txt` ile Python ortamı (NeMo 3.0.0), `curl`, Apple Silicon Mac (MPS), eğitim için en az 15 GB
boş disk (sürdürme dosyası ≈ 3,7 GB, model ≈ 2,6 GB) ve indirilen sesler için yer. Bütün yollar bir çalışma dizinine
göredir (`--work-dir`, `$AYAZ_ASR_WORK` ya da `./work`).

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export AYAZ_ASR_WORK=$PWD/work
# temel model: nemotron-3.5-asr-streaming-0.6b.nemo dosyasını (NVIDIA model deposundan) $AYAZ_ASR_WORK/models/ altına koyun
bash reproduce.sh            # aşağıdaki 1–9. adımlar, v0.1 ayarları
```

1–9. adımları `reproduce.sh` koşar; 10–11. adımlar elle yapılır.

1. `python3 train/train.py env` — ortam denetimi (MPS bellek ayırma, sürümler).
2. `python3 data/yodas_index.py`, sonra `python3 data/yodas_download.py` — YODAS Türkçe dizini ve parçaları.
3. `google/fleurs`'un `data/tr_tr/test.tsv` ve `data/tr_tr/audio/test.tar.gz` dosyalarını
   `$AYAZ_ASR_WORK/datasets/fleurs/tr_tr/` altına indirin, sonra `python3 data/prepare_eval_sets.py`.
4. `python3 data/build_turkish_manifest.py` (+ isteğe bağlı dağıtılmayan veri).
5. `python3 eval/fleurs_multilingual.py download --parallel 3` ve `... decode` — temel model, dil başına 100 sınav
   kaydında (prova etiket denetimi bu değerlerle karşılaştırır).
6. `python3 data/build_rehearsal.py download`, sonra
   `... label --turkish-manifest $AYAZ_ASR_WORK/manifests/turkish.jsonl --max-label-tokens 127`, sonra
   `python3 data/filter_rehearsal.py`. Varsayılanlar v0.1 ayarlarıdır: dil başına 300 kayıt; Almanca ve İngilizce 1.300;
   Letonca, Litvanca, Estonca ve Rumence 600; etiket kalite payı 0,12; 15–32 sa. Betiğin seçim mantığı v0.1'in etiketleme
   istatistikleri üzerinde yeniden oynatıldığında dil başına aynı sayılar çıkar (12.680 satır, 28,08 sa; süzgeçten sonra
   12.649). FLEURS dosyaları ya da temel modelin çıktıları değişirse gerçek bir koşu farklı olabilir.
7. `python3 data/build_mixture.py` (+ kendi veriniz için `--extra YOL:KAT`).
8. `python3 train/train.py run` (+ kendi sınav kümeleriniz) — `summary_*.json`, `steps_*.csv` ve `model_*.nemo`
   dosyalarını `$AYAZ_ASR_WORK/train/` altına yazar.
9. `python3 eval/fleurs_multilingual.py download --full --parallel 3`, sonra `decode --full --model "AD|ETİKET|YOL|1|DEĞER"`
   ve `summary --full --model ... --verdict AD`; DEĞER, eğitim özetinde seçilen adımın FLEURS-tr alt küme değeridir
   (değerlendirmenin eğitim gibi çözdüğünün sağlaması: 150 parçalık alt kümede ±0,002).
10. Dışa aktarma: `REFERENCE=<temel modelin resmî sherpa-onnx paketi> bash export/export_sherpa_onnx.sh model.nemo cikti_dizini`.
11. Bir `.nemo` yayımlamadan önce: `python3 tools/clean_nemo_config.py model.nemo model.clean.nemo --check-restore`.

### Bilinen eksikler ve sınırlar

* **v0.1 tam olarak yeniden üretilemez** (bkz. Veri). Yalnız herkese açık veriyle Türkçe kısım yalnız YODAS'tır ve eğitim
  içi hüküm tam hesaplanamaz (alan içi, kısa söz, boş-kalma ve gürültü kümeleri gerekir).
* Yalnız herkese açık veriyle `reproduce.sh`, YODAS geliştirme kümesini `--indomain-test` olarak kullanır; bu küme adım
  seçimini yönlendirir. Bu durumda bu kümenin sonucu yansız bir ölçüm sayılmaz ve kartın YODAS satırıyla
  karşılaştırılamaz.
* v0.1 ortamının torch, lightning, lhotse, numba ve Python sürümleri kaydedilmedi (yalnız NeMo 3.0.0 ve dışa aktarma
  için onnx 1.23.0 / onnxruntime 1.28.0 biliniyor).
* YODAS manifesti paralel indirmenin bitiş sırasıyla yazılır; video başına tohumlu örnekleme bu sıraya bağlıdır, bu
  yüzden seçilen YODAS parçaları koşudan koşuya değişebilir.
* FLEURS arşivleri yayımlandıkları sırayla okunur; veri dosyaları kaynağında değişirse prova seçimi de değişir.
* `train.py`'nin kaydettiği `.nemo`, `model_config.yaml` içinde (`train_ds`) yerel manifest yollarınızı taşır; bir model
  yayımlamadan önce `tools/clean_nemo_config.py` ile temizleyin.
* Model sınırları (model kartından): kısa sözler hâlâ zayıf (yaklaşık %77–80 hata); konuşmacı çeşitliliği dar, Türkçe
  kazancın bir kısmı tek bir konuşmacıya, geliştiriciye aittir; cümle içi dil geçişi ölçülmedi; Türkçeye özgü bazı harfler öteki dillerin
  çıktısına sızıyor; int8 paketi yalnız canlı kullanımda denendi; YODAS içeriğinin çoğu ders anlatımı olduğu için sayılar
  rakamla yazılabilir.

### Atıf

Lütfen modele atıf yapın (BibTeX yukarıda, model kartındakiyle aynı). Bu çalışma NVIDIA tarafından onaylanmamış ya da
desteklenmemiştir.

Kaynak numaraları İngilizce bölümdeki listeye göndermedir.

Yazar: Veli Ayaz
