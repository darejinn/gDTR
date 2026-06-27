# External data sources

External data is NOT committed to this repo. Download into this directory before running scripts.

## Required

- **GRCh38 reference** chr17.fa + chr22.fa: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/
- **GENCODE v44 GTF**: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz
- **ENCODE SCREEN cCRE registry** (chr22 ELS subset): https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed → filter `chr=='chr22' && $6 ~ /ELS/`
- **GTEx v8 single-tissue cis-eQTL** (Whole Blood): https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar
- **GWAS Catalog associations**: https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations-full.zip
- **ClinVar 2026-04-18 VCF**: ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
- **Evo 2 7B weights**: huggingface-cli download arcinstitute/evo2_7b (auto-fetched at first model load; SHA bda0089f92582d5baabf0f22d9fc85f3588f6b58)
