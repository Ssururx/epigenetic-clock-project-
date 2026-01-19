# Epigenetics Project - Step 6: CpG-to-Gene Mapping

# Initial setup and imports
print("Installing packages...")
!pip install pandas numpy matplotlib seaborn requests -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from datetime import datetime
from matplotlib_venn import venn2
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

print("Packages loaded")

# Google Drive setup and project structure
print("Setting up Google Drive project structure...")

drive.mount('/content/drive')
print("Drive mounted")

# Primary project root
PROJECT_ROOT = '/content/drive/MyDrive/epigenetics_project/'

# Step-specific paths
STEP3_CPGS = f'{PROJECT_ROOT}3_feature_discovery/top_cpgs/'
STEP6_ROOT = f'{PROJECT_ROOT}6_mapping/'

STEP6_FIGURES = f'{STEP6_ROOT}figures/'
STEP6_TABLES = f'{STEP6_ROOT}tables/'
STEP6_GENELISTS = f'{STEP6_ROOT}gene_lists/'
STEP6_REPORTS = f'{STEP6_ROOT}reports/'

# Create directories
print("Creating Step 6 structure...")
for folder in [STEP6_ROOT, STEP6_FIGURES, STEP6_TABLES, STEP6_GENELISTS, STEP6_REPORTS]:
    os.makedirs(folder, exist_ok=True)
    print(f"   Created: {folder}")

print("Structure ready")

# Load or download Illumina 450K annotation
print("Loading Illumina 450K annotation...")

annot_url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
local_annot_path = f'{PROJECT_ROOT}annotation/HM450_manifest_v1-2.csv'

os.makedirs(os.path.dirname(local_annot_path), exist_ok=True)

if not os.path.exists(local_annot_path):
    print("   Downloading official Illumina manifest...")
    response = requests.get(annot_url)
    with open(local_annot_path, 'wb') as f:
        f.write(response.content)
    print("   Downloaded successfully")
else:
    print("   Found local annotation")

# Load with correct skiprows
annot_df = pd.read_csv(local_annot_path, skiprows=7, low_memory=False)

print(f"   Raw annotation loaded: {annot_df.shape[0]:,} rows, {annot_df.shape[1]:,} columns")

# Identify correct column names
cols = annot_df.columns.str.lower()

probe_col = [c for c in annot_df.columns if 'ilmnid' in c.lower() or c.lower() == 'name'][0]
gene_col = [c for c in annot_df.columns if 'ucsc_refgene_name' in c.lower() or 'refgene' in c.lower()][0]
chr_col = [c for c in annot_df.columns if 'chr' in c.lower() and 'map' not in c.lower()][0]
pos_col = [c for c in annot_df.columns if 'mapinfo' in c.lower() or 'position' in c.lower()][0]

print(f"   Using columns:")
print(f"     Probe ID: {probe_col}")
print(f"     Gene: {gene_col}")
print(f"     Chromosome: {chr_col}")
print(f"     Position: {pos_col}")

# Select and clean
annot_df = annot_df[[probe_col, gene_col, chr_col, pos_col]].copy()
annot_df.columns = ['ProbeID', 'UCSC_RefGene_Name', 'CHR', 'MAPINFO']
annot_df = annot_df.set_index('ProbeID')

print(f"Final annotation ready: {annot_df.shape[0]:,} probes")

# Utility functions
def print_section(title, char='=', width=80):
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def save_figure(filename, dpi=300):
    path = f'{STEP6_FIGURES}{filename}'
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"   Saved figure to Google Drive: {path}")

def save_table(df, filename, description):
    path = f'{STEP6_TABLES}{filename}'
    df.to_csv(path, index=False)
    print(f"   Saved table to Google Drive: {filename} ({description})")

def save_report(text, filename):
    path = f'{STEP6_REPORTS}{filename}'
    with open(path, 'w') as f:
        f.write(text)
    print(f"   Saved report to Google Drive: {filename}")

def save_gene_list(genes, filename, desc):
    gene_list = sorted(list(genes))
    df = pd.DataFrame({'Gene': gene_list})
    path = f'{STEP6_GENELISTS}{filename}'
    df.to_csv(path, index=False)
    print(f"   Saved gene list to Google Drive: {filename} ({desc})")

    print(f"   First 5 genes in '{desc}':")
    for i, gene in enumerate(gene_list[:5]):
        print(f"     {i+1}. {gene}")
    if len(gene_list) > 5:
        print(f"     ... and {len(gene_list) - 5} more genes")
    print()

    return df, gene_list

# Load top CpGs from Step 3
print_section("Loading Top CpGs from Step 3")

brain_cpgs_path = f'{STEP3_CPGS}top_500_brain_cpgs.csv'
blood_cpgs_path = f'{STEP3_CPGS}top_500_blood_cpgs.csv'

brain_top_cpgs = pd.read_csv(brain_cpgs_path)['CpG'].tolist()
blood_top_cpgs = pd.read_csv(blood_cpgs_path)['CpG'].tolist()

print(f"Brain top CpGs: {len(brain_top_cpgs)}")
print(f"Blood top CpGs: {len(blood_top_cpgs)}")

# Load Horvath and Hannum clocks for uniquety filter
print_section("Loading Horvath and Hannum Clocks")

clock_cpgs = set()

horvath_file = "/content/drive/MyDrive/Hovarth.csv"
if os.path.exists(horvath_file):
    horvath_df = pd.read_csv(horvath_file, header=2)
    horvath_list = horvath_df['CpGmarker'].dropna().tolist()
    if horvath_list and str(horvath_list[0]).lower().startswith('(intercept)'):
        horvath_list = horvath_list[1:]
    clock_cpgs.update(horvath_list)
    print(f"   Horvath: {len(horvath_list)} CpGs")

hannum_file = "/content/drive/MyDrive/Hannum.xlsx"
if os.path.exists(hannum_file):
    hannum_df = pd.read_excel(hannum_file, sheet_name='Model_PrimaryData')
    hannum_list = hannum_df['Marker'].dropna().tolist()
    clock_cpgs.update(hannum_list)
    print(f"   Hannum: {len(hannum_list)} CpGs")

print(f"Total unique clock CpGs: {len(clock_cpgs):,}")

# Map CpGs to genes
print_section("Mapping CpGs to Genes")

def map_cpgs_to_genes(cpg_list, annotation):
    mapped_genes = set()
    cpg_gene_map = {}

    for cpg in cpg_list:
        if cpg in annotation.index:
            gene_str = annotation.loc[cpg, 'UCSC_RefGene_Name']
            if pd.notna(gene_str) and gene_str.strip():
                genes = {g.strip() for g in str(gene_str).split(';') if g.strip()}
                if genes:
                    mapped_genes.update(genes)
                    cpg_gene_map[cpg] = genes

    return mapped_genes, cpg_gene_map

brain_genes_all, brain_cpg_gene = map_cpgs_to_genes(brain_top_cpgs, annot_df)
print(f"Brain: {len(brain_cpg_gene)} CpGs -> {len(brain_genes_all)} unique genes")

blood_genes_all, blood_cpg_gene = map_cpgs_to_genes(blood_top_cpgs, annot_df)
print(f"Blood: {len(blood_cpg_gene)} CpGs -> {len(blood_genes_all)} unique genes")

unique_brain = brain_genes_all - blood_genes_all
unique_blood = blood_genes_all - brain_genes_all
shared_genes = brain_genes_all & blood_genes_all

print(f"\nUnique to Brain: {len(unique_brain)}")
print(f"Unique to Blood: {len(unique_blood)}")
print(f"Shared: {len(shared_genes)}")

# Save gene lists
print("\nSaving gene lists:")
print("=" * 50)
brain_all_df, brain_all_list = save_gene_list(brain_genes_all, 'brain_all_genes.csv', "All Brain-associated genes")
blood_all_df, blood_all_list = save_gene_list(blood_genes_all, 'blood_all_genes.csv', "All Blood-associated genes")
unique_brain_df, unique_brain_list = save_gene_list(unique_brain, 'unique_brain_genes.csv', "Unique to Brain")
unique_blood_df, unique_blood_list = save_gene_list(unique_blood, 'unique_blood_genes.csv', "Unique to Blood")
shared_genes_df, shared_genes_list = save_gene_list(shared_genes, 'shared_genes.csv', "Shared genes")

# Identify unique genes
print_section("Identifying unique Genes")

def get_unique_genes(cpg_gene_map, clock_set):
    unique = set()
    for cpg, genes in cpg_gene_map.items():
        if cpg not in clock_set:
            unique.update(genes)
    return unique

unique_brain = get_unique_genes(brain_cpg_gene, clock_cpgs)
unique_blood = get_unique_genes(blood_cpg_gene, clock_cpgs)

print(f"unique genes in Brain: {len(unique_brain)}")
print(f"unique genes in Blood: {len(unique_blood)}")

print("\nSaving unique gene lists:")
print("=" * 50)
unique_brain_df, unique_brain_list = save_gene_list(unique_brain, 'unique_brain_genes.csv', "unique Brain genes")
unique_blood_df, unique_blood_list = save_gene_list(unique_blood, 'unique_blood_genes.csv', "unique Blood genes")

# Visualizations
print_section("Generating Visualizations")

# Venn diagram
plt.figure(figsize=(10, 8))
venn2(subsets=(len(unique_brain), len(unique_blood), len(shared_genes)),
      set_labels=('Brain Genes', 'Blood Genes'),
      set_colors=('skyblue', 'lightcoral'))
plt.title('Gene Overlap Between Tissue-Specific Clocks')
save_figure('gene_overlap_venn.png')

# unique genes bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(['Brain', 'Blood'], [len(unique_brain), len(unique_blood)], color=['skyblue', 'lightcoral'])
plt.title('unique Age-Associated Genes\n(Not in Horvath or Hannum)')
plt.ylabel('Number of Genes')
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom')
save_figure('unique_genes_bar.png')

# Comprehensive summary figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

ax1 = axes[0, 0]
categories = ['All Brain', 'All Blood', 'Unique Brain', 'Unique Blood', 'unique Brain', 'unique Blood']
counts = [len(brain_genes_all), len(blood_genes_all), len(unique_brain),
          len(unique_blood), len(unique_brain), len(unique_blood)]
colors = ['skyblue', 'lightcoral', 'steelblue', 'indianred', 'royalblue', 'crimson']

bars = ax1.bar(categories, counts, color=colors)
ax1.set_title('Gene Counts by Category')
ax1.set_ylabel('Number of Genes')
ax1.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{count}', ha='center', va='bottom')

ax2 = axes[0, 1]
brain_mapped = len(brain_cpg_gene)
blood_mapped = len(blood_cpg_gene)
brain_efficiency = brain_mapped / len(brain_top_cpgs) * 100
blood_efficiency = blood_mapped / len(blood_top_cpgs) * 100

ax2.bar(['Brain', 'Blood'], [brain_efficiency, blood_efficiency], color=['skyblue', 'lightcoral'])
ax2.set_title('CpG-to-Gene Mapping Efficiency')
ax2.set_ylabel('Percentage of CpGs Mapped to Genes (%)')
ax2.set_ylim([0, 100])
ax2.text(0, brain_efficiency + 2, f'{brain_efficiency:.1f}%', ha='center')
ax2.text(1, blood_efficiency + 2, f'{blood_efficiency:.1f}%', ha='center')

ax3 = axes[1, 0]
uniquety_brain = len(unique_brain) / len(brain_genes_all) * 100 if len(brain_genes_all) > 0 else 0
uniquety_blood = len(unique_blood) / len(blood_genes_all) * 100 if len(blood_genes_all) > 0 else 0

ax3.bar(['Brain', 'Blood'], [uniquety_brain, uniquety_blood], color=['royalblue', 'crimson'])
ax3.set_title('Percentage of unique Genes')
ax3.set_ylabel('unique Genes (%)')
ax3.set_ylim([0, 100])
ax3.text(0, uniquety_brain + 2, f'{uniquety_brain:.1f}%', ha='center')
ax3.text(1, uniquety_blood + 2, f'{uniquety_blood:.1f}%', ha='center')

ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
Comprehensive Summary

CpG Statistics:
- Brain CpGs: {len(brain_top_cpgs)} total, {brain_mapped} mapped ({brain_efficiency:.1f}%)
- Blood CpGs: {len(blood_top_cpgs)} total, {blood_mapped} mapped ({blood_efficiency:.1f}%)

Gene Statistics:
- Brain genes: {len(brain_genes_all)} total
  • {len(unique_brain)} unique to brain
  • {len(unique_brain)} unique ({uniquety_brain:.1f}%)
- Blood genes: {len(blood_genes_all)} total
  • {len(unique_blood)} unique to blood
  • {len(unique_blood)} unique ({uniquety_blood:.1f}%)
- Shared genes: {len(shared_genes)}
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
save_figure('comprehensive_gene_analysis_summary.png')

# Create detailed mapping tables
print_section("Creating Detailed Mapping Tables")

def create_mapping_table(cpg_list, cpg_gene_map, annotation, tissue_name):
    mapping_data = []

    for cpg in cpg_list:
        row = {'CpG': cpg}

        if cpg in annotation.index:
            row['Gene(s)'] = annotation.loc[cpg, 'UCSC_RefGene_Name']
            row['Chromosome'] = annotation.loc[cpg, 'CHR']
            row['Position'] = annotation.loc[cpg, 'MAPINFO']
            row['In_Clock'] = cpg in clock_cpgs
        else:
            row['Gene(s)'] = 'NA'
            row['Chromosome'] = 'NA'
            row['Position'] = 'NA'
            row['In_Clock'] = cpg in clock_cpgs

        row['Mapped_Genes'] = ';'.join(sorted(cpg_gene_map.get(cpg, [])))

        mapping_data.append(row)

    return pd.DataFrame(mapping_data)

brain_mapping_df = create_mapping_table(brain_top_cpgs, brain_cpg_gene, annot_df, 'Brain')
blood_mapping_df = create_mapping_table(blood_top_cpgs, blood_cpg_gene, annot_df, 'Blood')

save_table(brain_mapping_df, 'brain_cpg_gene_mapping.csv', "Complete Brain CpG-to-gene mapping")
save_table(blood_mapping_df, 'blood_cpg_gene_mapping.csv', "Complete Blood CpG-to-gene mapping")

print(f"Brain mapping table: {brain_mapping_df.shape[0]} CpGs x {brain_mapping_df.shape[1]} columns")
print(f"Blood mapping table: {blood_mapping_df.shape[0]} CpGs x {blood_mapping_df.shape[1]} columns")

# Final comprehensive report
print_section("Step 6 Final Report")

report = f"""
Epigenetics Project - Step 6: CpG-to-Gene Mapping
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Analysis Overview:
This step maps tissue-specific age-associated CpGs to genes using Illumina 450K
annotation, identifies unique and unique genes, and prepares gene lists for
pathway enrichment analysis.

Methods:
- Annotation: Illumina HumanMethylation450 manifest (v1-2)
- CpG lists: Top 500 age-associated CpGs from Step 3 for Brain and Blood
- uniquety filter: CpGs not present in Horvath (353 CpGs) or Hannum (71 CpGs) clocks
- Gene mapping: UCSC RefGene annotations from Illumina manifest

Results Summary:
====================================

CpG-to-Gene Mapping Efficiency:
- Brain: {len(brain_top_cpgs)} CpGs -> {len(brain_cpg_gene)} mapped ({brain_efficiency:.1f}%)
- Blood: {len(blood_top_cpgs)} CpGs -> {len(blood_cpg_gene)} mapped ({blood_efficiency:.1f}%)

Gene Discovery Results:
====================================

1. All Associated Genes:
   - Brain: {len(brain_genes_all)} genes
   - Blood: {len(blood_genes_all)} genes

2. Tissue-Specific Genes:
   - Unique to Brain: {len(unique_brain)} genes
   - Unique to Blood: {len(unique_blood)} genes
   - Shared between tissues: {len(shared_genes)} genes

3. unique Genes (not in Horvath/Hannum clocks):
   - unique Brain genes: {len(unique_brain)} ({uniquety_brain:.1f}% of Brain genes)
   - unique Blood genes: {len(unique_blood)} ({uniquety_blood:.1f}% of Blood genes)

First 5 Genes (per category):
====================================

Brain:
  All genes: {brain_all_list[:5]}
  Unique genes: {unique_brain_list[:5]}
  unique genes: {unique_brain_list[:5]}

Blood:
  All genes: {blood_all_list[:5]}
  Unique genes: {unique_blood_list[:5]}
  unique genes: {unique_blood_list[:5]}

Shared between tissues: {shared_genes_list[:5]}

Gene Lists Generated:
====================================

Ready for Pathway Analysis:

1. Brain-Specific Lists:
   - brain_all_genes.csv ({len(brain_all_list)} genes) - All brain age-associated genes
   - unique_brain_genes.csv ({len(unique_brain_list)} genes) - Brain-specific genes
   - unique_brain_genes.csv ({len(unique_brain_list)} genes) - unique brain genes

2. Blood-Specific Lists:
   - blood_all_genes.csv ({len(blood_all_list)} genes) - All blood age-associated genes
   - unique_blood_genes.csv ({len(unique_blood_list)} genes) - Blood-specific genes
   - unique_blood_genes.csv ({len(unique_blood_list)} genes) - unique blood genes

3. Comparative Lists:
   - shared_genes.csv ({len(shared_genes_list)} genes) - Genes shared between tissues

Detailed Mapping Tables:
====================================

- brain_cpg_gene_mapping.csv - Complete Brain CpG-to-gene mapping
- blood_cpg_gene_mapping.csv - Complete Blood CpG-to-gene mapping

Each mapping table includes:
  - CpG identifier
  - Official gene annotation
  - Chromosome and position
  - Whether CpG is in Horvath/Hannum clocks
  - Mapped gene symbols

Visualizations Generated:
====================================

1. gene_overlap_venn.png - Venn diagram of gene overlap between tissues
2. unique_genes_bar.png - Bar plot of unique gene counts
3. comprehensive_gene_analysis_summary.png - Multi-panel summary figure

Biological Interpretation:
====================================

1. Tissue Specificity:
   - {len(unique_brain)} genes unique to brain suggest brain-specific aging mechanisms
   - {len(unique_blood)} genes unique to blood suggest blood-specific aging mechanisms
   - {len(shared_genes)} shared genes may represent conserved aging processes

2. uniquety Significance:
   - unique genes ({uniquety_brain:.1f}% in brain, {uniquety_blood:.1f}% in blood) represent
     previously undiscovered aging associations
   - These unique genes are prime candidates for new therapeutic targets

3. Pathway Analysis Readiness:
   - Gene lists are formatted for immediate upload to pathway enrichment tools
   - Focus on unique and tissue-specific lists for most biologically relevant insights

Technical Notes:
====================================

- Annotation source: Official Illumina HumanMethylation450 manifest
- Gene mapping: UCSC RefGene annotations (most comprehensive)
- uniquety definition: CpGs not in Horvath (2013) or Hannum (2013) clocks
- Multiple genes: CpGs mapping to multiple genes were expanded (; separator)
- Empty annotations: CpGs without gene annotations were excluded from gene lists

Output Location:
====================================

All files saved to: {STEP6_ROOT}
- Figures: {STEP6_FIGURES}
- Tables: {STEP6_TABLES}
- Gene lists: {STEP6_GENELISTS}
- Reports: {STEP6_REPORTS}

Step 6 completed successfully - ready for pathway analysis
"""

print(report)
save_report(report, 'STEP6_FINAL_REPORT.txt')

# Final output summary
print_section("Analysis Complete - Output Summary")

print("Generated Files:")
print("="*60)

print("\nGene Lists (ready for pathway analysis):")
print("-"*40)
print(f"  brain_all_genes.csv ({len(brain_all_list)} genes) - All brain age-associated genes")
print(f"  blood_all_genes.csv ({len(blood_all_list)} genes) - All blood age-associated genes")
print(f"  unique_brain_genes.csv ({len(unique_brain_list)} genes) - Brain-specific genes")
print(f"  unique_blood_genes.csv ({len(unique_blood_list)} genes) - Blood-specific genes")
print(f"  unique_brain_genes.csv ({len(unique_brain_list)} genes) - unique brain genes")
print(f"  unique_blood_genes.csv ({len(unique_blood_list)} genes) - unique blood genes")
print(f"  shared_genes.csv ({len(shared_genes_list)} genes) - Genes shared between tissues")

print("\nVisualizations:")
print("-"*40)
print("  gene_overlap_venn.png - Gene overlap between tissues")
print("  unique_genes_bar.png - unique gene counts")
print("  comprehensive_gene_analysis_summary.png - Multi-panel summary")

print("\nDetailed Mapping Tables:")
print("-"*40)
print(f"  brain_cpg_gene_mapping.csv - Brain CpG-to-gene mapping ({brain_mapping_df.shape[0]} CpGs)")
print(f"  blood_cpg_gene_mapping.csv - Blood CpG-to-gene mapping ({blood_mapping_df.shape[0]} CpGs)")

print("\nReports:")
print("-"*40)
print(f"  STEP6_FINAL_REPORT.txt - Complete analysis report")

print("\n" + "="*60)
print("Step 6 completed successfully")
print("="*60)

