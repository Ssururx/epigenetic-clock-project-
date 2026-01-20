# Epigenetics Project - Step 9: Comprehensive Annotation and Gene Mapping

"""
Comprehensive script for Illumina 450K annotation processing, CpG-to-gene mapping,
and extraction of methylation data for top CpGs in blood and brain tissues.

The script performs three sequential tasks:
1. Full annotation processing with essential genomic feature extraction
2. Mapping of top 500 CpGs to associated genes
3. Extraction of methylation values and aligned metadata for downstream analysis
"""

# ============================================================================
# Section 1: Imports and Setup
# ============================================================================

import pandas as pd
import os

# ============================================================================
# Section 2: Annotation Processing Functions
# ============================================================================

def setup_annotation_paths():
    """Define file paths for annotation processing."""
    BASE_DIR = "/content/drive/MyDrive/epigenetics_project"

    ANNOTATION_FILE = os.path.join(BASE_DIR, "annotation/HM450_manifest_v1-2.csv")
    BLOOD_TOP_CPGS = os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_blood_cpgs.csv")
    BRAIN_TOP_CPGS = os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_brain_cpgs.csv")

    OUTPUT_DIR = os.path.join(BASE_DIR, "9_gene_mapping/annotation_processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FULL_ANNOTATION = os.path.join(OUTPUT_DIR, "HM450_annotation_full.csv")
    ESSENTIAL_ANNOTATION = os.path.join(OUTPUT_DIR, "HM450_annotation_essential.csv")
    TOP500_ANNOTATION = os.path.join(OUTPUT_DIR, "top500_annotation.csv")

    return (ANNOTATION_FILE, BLOOD_TOP_CPGS, BRAIN_TOP_CPGS,
            OUTPUT_DIR, FULL_ANNOTATION, ESSENTIAL_ANNOTATION, TOP500_ANNOTATION)

def load_full_annotation(annotation_file):
    """
    Load the complete Illumina 450K annotation and extract essential columns.
    """
    print("Loading Illumina HM450 annotation file...")

    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return None, None

    try:
        annot_df = pd.read_csv(annotation_file, skiprows=7, low_memory=False)
        print(f"Raw annotation loaded: {annot_df.shape[0]:,} rows, {annot_df.shape[1]:,} columns")

        print(f"\nAvailable columns ({len(annot_df.columns)} total):")
        for i, col in enumerate(annot_df.columns[:20]):
            print(f"  {i+1:2d}. {col}")
        if len(annot_df.columns) > 20:
            print(f"  ... and {len(annot_df.columns) - 20} additional columns")

        target_columns = {
            'probe_id': ['ilmnid', 'name', 'probe'],
            'chr': ['chr', 'chromosome'],
            'position': ['mapinfo', 'position', 'coord'],
            'gene': ['ucsc_refgene_name', 'refgene', 'gene'],
            'gene_feature': ['ucsc_refgene_group', 'relationship', 'feature'],
            'cpg_island': ['relation_to_cpg_island', 'cpg_island', 'island'],
            'strand': ['strand'],
            'genome_build': ['genome_build', 'build']
        }

        print("\nIdentifying essential columns:")
        essential_cols = []
        column_mapping = {}

        for target_name, keywords in target_columns.items():
            found = False
            for col in annot_df.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    essential_cols.append(col)
                    column_mapping[target_name] = col
                    print(f"  Found {target_name.upper():15} -> {col}")
                    found = True
                    break
            if not found:
                print(f"  Not found {target_name.upper():15}")

        full_annotation = annot_df.copy()

        if essential_cols:
            essential_annotation = annot_df[essential_cols].copy()
            rename_dict = {v: k.upper() for k, v in column_mapping.items()}
            essential_annotation = essential_annotation.rename(columns=rename_dict)

            probe_col = next((c for c in essential_annotation.columns if 'PROBE' in c or 'ILMNID' in c), None)
            if probe_col:
                essential_annotation = essential_annotation.set_index(probe_col)

            print(f"\nEssential annotation prepared: {essential_annotation.shape}")
            print(f"   Columns: {list(essential_annotation.columns)}")
            print(f"\nSample of essential annotation:")
            print(essential_annotation.head(10).to_string())

            return full_annotation, essential_annotation

        return full_annotation, None

    except Exception as e:
        print(f"Error loading annotation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_top500_annotation(essential_annotation, blood_cpg_path, brain_cpg_path):
    """Generate annotation subsets for the top 500 CpGs in each tissue."""
    print("\nCreating annotation for top 500 CpGs")

    blood_cpgs = pd.read_csv(blood_cpg_path)['CpG'].tolist()[:500]
    brain_cpgs = pd.read_csv(brain_cpg_path)['CpG'].tolist()[:500]

    print(f"Blood top CpGs loaded: {len(blood_cpgs)}")
    print(f"Brain top CpGs loaded: {len(brain_cpgs)}")

    blood_annotation = essential_annotation.loc[essential_annotation.index.intersection(blood_cpgs)].copy()
    print(f"\nBlood top 500 annotation: {len(blood_annotation)}/{len(blood_cpgs)} CpGs found")

    brain_annotation = essential_annotation.loc[essential_annotation.index.intersection(brain_cpgs)].copy()
    print(f"Brain top 500 annotation: {len(brain_annotation)}/{len(brain_cpgs)} CpGs found")

    top500_annotation = pd.concat([
        blood_annotation.assign(Tissue='Blood'),
        brain_annotation.assign(Tissue='Brain')
    ])

    print(f"\nCombined top 500 annotation: {len(top500_annotation)} rows")
    print(f"   Blood: {len(blood_annotation)} rows")
    print(f"   Brain: {len(brain_annotation)} rows")

    if 'UCSC_REFGENE_GROUP' in top500_annotation.columns:
        print("\nGene feature distribution (top 10):")
        for feature, count in top500_annotation['UCSC_REFGENE_GROUP'].value_counts().head(10).items():
            print(f"   {feature:20}: {count:4d}")

    if 'RELATION_TO_CPG_ISLAND' in top500_annotation.columns:
        print("\nCpG island context distribution:")
        for context, count in top500_annotation['RELATION_TO_CPG_ISLAND'].value_counts().items():
            print(f"   {context:20}: {count:4d}")

    return top500_annotation, blood_annotation, brain_annotation

def save_annotation_files(full_annotation, essential_annotation, top500_annotation, output_paths):
    """Save processed annotation files to disk."""
    print("\nSaving annotation files...")

    FULL_ANNOTATION, ESSENTIAL_ANNOTATION, TOP500_ANNOTATION = output_paths

    try:
        if full_annotation is not None:
            full_annotation.to_csv(FULL_ANNOTATION, index=False)
            size_mb = os.path.getsize(FULL_ANNOTATION) / (1024 * 1024)
            print(f"Full annotation saved: {FULL_ANNOTATION} ({size_mb:.1f} MB)")

        if essential_annotation is not None:
            essential_annotation.to_csv(ESSENTIAL_ANNOTATION)
            size_kb = os.path.getsize(ESSENTIAL_ANNOTATION) / 1024
            print(f"Essential annotation saved: {ESSENTIAL_ANNOTATION} ({size_kb:.1f} KB)")

        if top500_annotation is not None:
            top500_annotation.to_csv(TOP500_ANNOTATION)
            size_kb = os.path.getsize(TOP500_ANNOTATION) / 1024
            print(f"Top 500 annotation saved: {TOP500_ANNOTATION} ({size_kb:.1f} KB)")

        print("\nAll annotation files saved successfully")

    except Exception as e:
        print(f"Error saving files: {e}")

def perform_annotation_processing():
    """Orchestrate complete annotation processing workflow."""
    print("=" * 80)
    print("STEP 9A: COMPREHENSIVE ANNOTATION PROCESSING")
    print("=" * 80)

    (ANNOTATION_FILE, BLOOD_TOP_CPGS, BRAIN_TOP_CPGS,
     OUTPUT_DIR, FULL_ANNOTATION, ESSENTIAL_ANNOTATION, TOP500_ANNOTATION) = setup_annotation_paths()

    full_annotation, essential_annotation = load_full_annotation(ANNOTATION_FILE)

    if essential_annotation is None:
        print("Essential annotation could not be created")
        return None, None, None

    top500_annotation, blood_annotation, brain_annotation = create_top500_annotation(
        essential_annotation, BLOOD_TOP_CPGS, BRAIN_TOP_CPGS
    )

    save_annotation_files(
        full_annotation, essential_annotation, top500_annotation,
        (FULL_ANNOTATION, ESSENTIAL_ANNOTATION, TOP500_ANNOTATION)
    )

    print("\n" + "=" * 80)
    print("SAMPLE OF TOP 500 ANNOTATION")
    print("=" * 80)

    if top500_annotation is not None:
        print(top500_annotation.head(15).to_string())

        print("\nSummary statistics:")
        print(f"Total CpGs annotated: {len(top500_annotation)}")
        print(f"Blood CpGs: {len(top500_annotation[top500_annotation['Tissue'] == 'Blood'])}")
        print(f"Brain CpGs: {len(top500_annotation[top500_annotation['Tissue'] == 'Brain'])}")

        if 'CHR' in top500_annotation.columns:
            print("\nChromosome distribution:")
            for chr_num, count in top500_annotation['CHR'].value_counts().sort_index().items():
                print(f"  Chr{chr_num}: {count:4d}")

    return essential_annotation, blood_annotation, brain_annotation

# ============================================================================
# Section 3: Gene Mapping Functions
# ============================================================================

def setup_gene_mapping_paths():
    """Define paths for gene mapping outputs."""
    BASE_DIR = "/content/drive/MyDrive/epigenetics_project"
    GENE_OUTPUT_DIR = os.path.join(BASE_DIR, "9_gene_mapping/gene_lists")
    os.makedirs(GENE_OUTPUT_DIR, exist_ok=True)

    BLOOD_CPG_PATH = os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_blood_cpgs.csv")
    BRAIN_CPG_PATH = os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_brain_cpgs.csv")

    return BASE_DIR, GENE_OUTPUT_DIR, BLOOD_CPG_PATH, BRAIN_CPG_PATH

def map_cpgs_to_genes_simple(cpg_list, annotation_df):
    """Map CpG probes to associated gene symbols."""
    mapped_genes = set()
    cpg_gene_rows = []

    gene_col = next((c for c in annotation_df.columns if 'GENE' in c or 'REFGENE' in c), None)

    for cpg in cpg_list:
        if cpg in annotation_df.index and gene_col:
            gene_str = annotation_df.loc[cpg, gene_col]
            if pd.notna(gene_str) and str(gene_str).strip():
                genes = {g.strip() for g in str(gene_str).split(';') if g.strip()}
                if genes:
                    mapped_genes.update(genes)
                    for gene in genes:
                        cpg_gene_rows.append({'CpG_ID': cpg, 'Gene': gene})
                    continue
            cpg_gene_rows.append({'CpG_ID': cpg, 'Gene': 'NO_GENE'})
        else:
            cpg_gene_rows.append({'CpG_ID': cpg, 'Gene': 'NOT_FOUND'})

    return mapped_genes, pd.DataFrame(cpg_gene_rows)

def perform_gene_mapping(essential_annotation=None):
    """Execute gene mapping for top CpGs in both tissues."""
    print("\n" + "=" * 60)
    print("STEP 9B: GENE MAPPING FOR TOP 500 CPGS")
    print("=" * 60)

    BASE_DIR, GENE_OUTPUT_DIR, BLOOD_CPG_PATH, BRAIN_CPG_PATH = setup_gene_mapping_paths()

    if essential_annotation is None:
        annot_path = os.path.join(BASE_DIR, "9_gene_mapping/annotation_processed/HM450_annotation_essential.csv")
        if os.path.exists(annot_path):
            essential_annotation = pd.read_csv(annot_path, index_col=0)
        else:
            print("Essential annotation not available")

    if essential_annotation is None:
        print("Annotation required for gene mapping")
        return None, None, None, None

    print("\nProcessing blood CpGs...")
    blood_cpgs = pd.read_csv(BLOOD_CPG_PATH)['CpG'].tolist()
    blood_genes, blood_mapping = map_cpgs_to_genes_simple(blood_cpgs, essential_annotation)

    blood_mapping.to_csv(os.path.join(GENE_OUTPUT_DIR, "blood_top500_cpg_gene_map.csv"), index=False)
    pd.DataFrame({'Gene': sorted(blood_genes)}).to_csv(os.path.join(GENE_OUTPUT_DIR, "blood_top500_genes.csv"), index=False)

    print(f"Blood: {len(blood_cpgs)} CpGs mapped to {len(blood_genes)} genes")

    print("\nProcessing brain CpGs...")
    brain_cpgs = pd.read_csv(BRAIN_CPG_PATH)['CpG'].tolist()
    brain_genes, brain_mapping = map_cpgs_to_genes_simple(brain_cpgs, essential_annotation)

    brain_mapping.to_csv(os.path.join(GENE_OUTPUT_DIR, "brain_top500_cpg_gene_map.csv"), index=False)
    pd.DataFrame({'Gene': sorted(brain_genes)}).to_csv(os.path.join(GENE_OUTPUT_DIR, "brain_top500_genes.csv"), index=False)

    print(f"Brain: {len(brain_cpgs)} CpGs mapped to {len(brain_genes)} genes")

    print("\nSample outputs:")
    print("\nBlood mapping (first 5 rows):")
    print(blood_mapping.head().to_string())
    print("\nBlood genes (first 10):")
    print(sorted(blood_genes)[:10])

    print("\nBrain mapping (first 5 rows):")
    print(brain_mapping.head().to_string())
    print("\nBrain genes (first 10):")
    print(sorted(brain_genes)[:10])

    print(f"\nGene mapping outputs saved to: {GENE_OUTPUT_DIR}")

    return blood_cpgs, brain_cpgs, blood_genes, brain_genes, GENE_OUTPUT_DIR

# ============================================================================
# Section 4: Methylation Data Extraction Functions
# ============================================================================

def setup_extraction_paths():
    """Define paths for methylation data extraction."""
    BASE_DIR = "/content/drive/MyDrive/epigenetics_project"
    EXTRACTION_OUTPUT_DIR = os.path.join(BASE_DIR, "9_gene_mapping/top500_data")
    os.makedirs(EXTRACTION_OUTPUT_DIR, exist_ok=True)

    BLOOD_METH_PATH = os.path.join(BASE_DIR, "2_data_qc/cleaned_data/cleaned_blood_methylation.csv")
    BRAIN_METH_PATH = os.path.join(BASE_DIR, "2_data_qc/cleaned_data/cleaned_brain_methylation.csv")
    BLOOD_META_PATH = os.path.join(BASE_DIR, "2_data_qc/cleaned_data/cleaned_blood_metadata.csv")
    BRAIN_META_PATH = os.path.join(BASE_DIR, "2_data_qc/cleaned_data/cleaned_brain_metadata.csv")

    return (BASE_DIR, EXTRACTION_OUTPUT_DIR, BLOOD_METH_PATH, BRAIN_METH_PATH,
            BLOOD_META_PATH, BRAIN_META_PATH)

def extract_methylation_data(tissue, cpgs, meth_path, meta_path, output_dir):
    """Extract methylation values and aligned metadata for specified CpGs."""
    print(f"\nProcessing {tissue} methylation data...")

    meth = pd.read_csv(meth_path)
    print(f"   Full dataset shape: {meth.shape}")

    if meth.shape[0] > meth.shape[1]:
        meth = meth.set_index(meth.columns[0])
        common = [c for c in cpgs if c in meth.index]
        meth_subset = meth.loc[common].T
        meth_samples = meth_subset.reset_index(drop=False)
    else:
        common = [c for c in cpgs if c in meth.columns]
        sample_col = meth.columns[0]
        meth_samples = meth[[sample_col] + common]

    print(f"   CpGs found: {len(common)}/{len(cpgs)}")
    print(f"   Final shape: {meth_samples.shape}")

    meth_out = os.path.join(output_dir, f"{tissue}_methylation_top500.csv")
    meth_samples.to_csv(meth_out, index=False)
    print(f"Saved methylation data: {meth_out}")

    meta = pd.read_csv(meta_path)
    if tissue == 'blood' and 'GSM_number' in meta.columns:
        meta = meta.set_index('GSM_number')
    else:
        meta = meta.set_index(meta.columns[0])

    sample_ids = meth_samples.iloc[:, 0] if meth_samples.columns[0] in ['Unnamed: 0', 'index'] else meth_samples.index
    common_samples = [s for s in sample_ids if s in meta.index]
    meta_filtered = meta.loc[common_samples]

    meta_out = os.path.join(output_dir, f"{tissue}_metadata_top500.csv")
    meta_filtered.to_csv(meta_out)
    print(f"Saved metadata: {meta_out}")

    return meth_samples, meta_filtered

def perform_data_extraction(blood_cpgs, brain_cpgs):
    """Execute data extraction for both tissues."""
    print("=" * 60)
    print("STEP 9C: METHYLATION DATA EXTRACTION FOR TOP 500 CPGS")
    print("=" * 60)

    (BASE_DIR, EXTRACTION_OUTPUT_DIR, BLOOD_METH_PATH, BRAIN_METH_PATH,
     BLOOD_META_PATH, BRAIN_META_PATH) = setup_extraction_paths()

    blood_meth, blood_meta = extract_methylation_data(
        'blood', blood_cpgs, BLOOD_METH_PATH, BLOOD_META_PATH, EXTRACTION_OUTPUT_DIR
    )

    print("\n" + "=" * 60)

    brain_meth, brain_meta = extract_methylation_data(
        'brain', brain_cpgs, BRAIN_METH_PATH, BRAIN_META_PATH, EXTRACTION_OUTPUT_DIR
    )

    print(f"\nAll extracted data saved to: {EXTRACTION_OUTPUT_DIR}")
    print("\nFinal dataset sizes:")
    print(f"Blood methylation: {blood_meth.shape}")
    print(f"Blood metadata: {blood_meta.shape}")
    print(f"Brain methylation: {brain_meth.shape}")
    print(f"Brain metadata: {brain_meta.shape}")

    return blood_meth, blood_meta, brain_meth, brain_meta, EXTRACTION_OUTPUT_DIR

# ============================================================================
# Section 5: Main Execution
# ============================================================================

def main():
    """Execute complete Step 9 workflow."""
    print("=" * 80)
    print("EPIGENETICS PROJECT - STEP 9: COMPREHENSIVE ANNOTATION AND GENE MAPPING")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PART 1: ANNOTATION PROCESSING")
    print("=" * 80)

    essential_annotation, blood_annotation, brain_annotation = perform_annotation_processing()

    if essential_annotation is None:
        BASE_DIR = "/content/drive/MyDrive/epigenetics_project"
        blood_cpgs = pd.read_csv(os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_blood_cpgs.csv"))['CpG'].tolist()
        brain_cpgs = pd.read_csv(os.path.join(BASE_DIR, "3_feature_discovery/tables/top_500_brain_cpgs.csv"))['CpG'].tolist()
    else:
        blood_cpgs = blood_annotation.index.tolist() if blood_annotation is not None else []
        brain_cpgs = brain_annotation.index.tolist() if brain_annotation is not None else []

    print("\n" + "=" * 80)
    print("PART 2: GENE MAPPING")
    print("=" * 80)

    mapping_results = perform_gene_mapping(essential_annotation)

    if mapping_results[0] is not None:
        blood_cpgs, brain_cpgs, blood_genes, brain_genes, _ = mapping_results

    print("\n" + "=" * 80)
    print("PART 3: DATA EXTRACTION")
    print("=" * 80)

    extraction_results = perform_data_extraction(blood_cpgs, brain_cpgs)

    print("\n" + "=" * 80)
    print("Step 9 is completed")
    print("=" * 80)

    print("\nTasks completed:")
    print("- Processed complete Illumina 450K annotation")
    print("- Extracted essential genomic features")
    print("- Made annotation for top 500 CpGs")
    print("- Mapped CpGs to gene symbols")
    print("- Extracted methylation data for top CpGs")
    print("- Aligned and saved corresponding metadata")

    print("\nOutput directories:")
    print("Annotation: /content/drive/MyDrive/epigenetics_project/9_gene_mapping/annotation_processed/")
    print("Gene lists: /content/drive/MyDrive/epigenetics_project/9_gene_mapping/gene_lists/")
    print("Extracted data: /content/drive/MyDrive/epigenetics_project/9_gene_mapping/top500_data/")

    if 'blood_genes' in locals() and 'brain_genes' in locals():
        print(f"\nBlood CpGs mapped to {len(blood_genes)} genes")
        print(f"Brain CpGs mapped to {len(brain_genes)} genes")


if __name__ == "__main__":
    main()
