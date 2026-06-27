# gDTR: Genomic Deep-Thinking Ratio

A training-free residual-stream lens for genomic causal language models. This repository accompanies:

> Cho, Kang, Park, Kim. **"gDTR: Layer-wise Settling Depth Reveals Biological Grammar in Genomic Foundation Models."** Accepted at the ICML 2026 Workshop on Generative and Agentic AI for Biology.

## What is gDTR?

gDTR assigns each nucleotide token a **settling depth** c(t): the first layer at which its representation aligns with the model's output-ready frame. On Evo 2 7B applied to human chr22 and chr17, settling depth gives a chromosome-transferable, ClinVar-consequence-correlated readout of representational dynamics that is not visible from the model's final prediction alone.

## Quick links

- [Paper PDF](paper/gdtr_paper_ICML_3.pdf)
- [Reproduction guide](REPRODUCE.md)
- [Core source (`src/`)](src/): `gdtr.py` (settling depth), `calibration.py` (per-region q70), `ur_gdtr_evo2.py` (cosine lens)
- [Tier-A analysis scripts](scripts/analysis/): chr22 master forward, Option-A merge, rank-biserial, Fig 2 unified baseline
- [Figure regeneration scripts](scripts/figures/)
- [Results tables](results/tier_A/)

## Headline numbers

| Quantity | Value |
|---|---|
| Pooled chr17+chr22 intron baseline | c̄ = 27.72 |
| Splice donor d vs intron (pooled) | -0.354 |
| ENCODE cCRE-ELS d vs intron (pooled) | -0.118 |
| ClinVar 6-way KW | p = 3.0×10⁻¹⁰, H = 53.2, ε² = 0.013 |
| chr22 → chr17 calibration retention | 94.6% |

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{cho2026gdtr,
  title={gDTR: Layer-wise Settling Depth Reveals Biological Grammar in Genomic Foundation Models},
  author={Cho, Yoonjin and Kang, Jiheon and Park, Subin and Kim, Sangwoo},
  booktitle={ICML 2026 Workshop on Generative and Agentic AI for Biology},
  year={2026}
}
```
