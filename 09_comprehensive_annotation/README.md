# Step 9 – Comprehensive Annotation and Gene Mapping

In this step, I transitioned from raw statistical features to high-resolution genomic context. By integrating the official Illumina HM450 annotation manifest, I mapped my top 500 age-associated CpGs for both brain and blood to their specific locations in the human genome, identifying the genes and regulatory regions they influence.

---

### Annotation Processing
I processed the raw Illumina HM450 manifest (containing over 486,000 probes) to extract essential genomic features for my top-performing markers. This allows me to see not just if a site ages, but where it is located.

* **Genomic Features Extracted:** Chromosome (CHR), MAPINFO (Position), Gene Feature (e.g., TSS1500, 5'UTR, Body), and CpG Island status.
* **Chromosomal Distribution:** I observed a broad distribution across the genome, with **Chromosome 1** (111 CpGs) and **Chromosome 19** (97 CpGs) showing the highest density of aging markers in my combined top 1000 list.



---

### Gene Mapping for Top 500 CpGs
I mapped each CpG to its official UCSC Gene Symbol. Because many CpGs reside near multiple overlapping genes, my mapping captures the full complexity of the genomic neighborhood.

* **Blood Mapping:** My top 500 CpGs mapped to **523 unique genes** (including *NCOR2*, *ADRA2C*, and *ABO*).
* **Brain Mapping:** My top 500 CpGs mapped to **363 unique genes** (including *FKBP5*, *ELOVL2*, and *GATA3*).
* **Significance:** The identification of well-characterized aging-associated genes such as *ELOVL2* confirms the robustness of the feature selection process, while the discovery of additional, previously unreported gene associations highlights potential avenues for novel biological insights.
---

### Methylation Data Extraction
To prepare for deep-dive biological modeling, I extracted the raw methylation beta-values for only the top 500 CpGs. This "clean" dataset is much more manageable for advanced visualization and regulatory analysis.

| Dataset | Final Shape (Samples x Features) | Key Output File |
| :--- | :--- | :--- |
| **Blood Methylation** | 1322 x 501 | `blood_methylation_top500.csv` |
| **Brain Methylation** | 335 x 501 | `brain_methylation_top500.csv` |

---

### Future Goal: Regulatory Logic Overlap (WIP)
I am currently working on a "Regulatory Logic" analysis to be integrated into this step. By overlapping my CpG coordinates with **ENCODE histone modification data** (e.g., H3K27ac for enhancers and H3K4me3 for active promoters), I aim to prove that my 500 CpGs are located in biologically "active" regions. This will provide evidence that these epigenetic shifts truly influence gene expression and cellular function.



---

### Key Outputs Created
* `HM450_annotation_essential.csv`: A streamlined genomic reference for all 486k probes.
* `top500_annotation.csv`: Detailed genomic context for my most important aging markers.
* `blood_methylation_top500.csv` & `brain_methylation_top500.csv`: Optimized datasets ready for functional analysis.

 Since I have successfully mapped the computational features to their corresponding genomic loci, we can move on to **Step 10: Comparative Analysis of DMRs and Genes**

---
