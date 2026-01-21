# Epigenetics Project - Step 10: Comparative Analysis of DMRs and Genes

"""
Enhanced comparative analysis of differentially methylated regions (DMRs) and associated genes
in blood and brain tissues. This script includes DMR clustering, tissue-specific gene
identification, genomic context analysis, functional enrichment, improved visualizations,
and a comprehensive final summary.
"""

# ============================================================================
# Section 1: Setup and Imports
# ============================================================================

import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from gprofiler import GProfiler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# Section 2: Path Configuration
# ============================================================================

base_dir = '/content/drive/MyDrive/epigenetics_project/'
top500_dir = os.path.join(base_dir, '9_gene_mapping/gene_lists')
methyl_dir = os.path.join(base_dir, '9_gene_mapping/top500_data')
annot_path = os.path.join(base_dir, '9_gene_mapping/annotation_processed/HM450_annotation_essential.csv')
results_dir = os.path.join(base_dir, '10_comparative_analysis/results')
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("STEP 10: COMPARATIVE ANALYSIS OF DMRs AND GENES")
print("=" * 80)

# ============================================================================
# Section 3: Load Top 500 Methylation Data
# ============================================================================

print("\nLoading methylation data...")

try:
    blood_meth_path = os.path.join(methyl_dir, 'blood_methylation_top500.csv')
    brain_meth_path = os.path.join(methyl_dir, 'brain_methylation_top500.csv')

    if not os.path.exists(blood_meth_path):
        print(f"Blood methylation file not found: {blood_meth_path}")
        print("   Run Step 9 first")
        raise SystemExit

    blood_meth = pd.read_csv(blood_meth_path)
    brain_meth = pd.read_csv(brain_meth_path)

    if 'Unnamed: 0' in blood_meth.columns:
        blood_meth = blood_meth.set_index('Unnamed: 0')
    if 'Unnamed: 0' in brain_meth.columns:
        brain_meth = brain_meth.set_index('Unnamed: 0')

    blood_meta = pd.read_csv(os.path.join(methyl_dir, 'blood_metadata_top500.csv'), index_col=0)
    brain_meta = pd.read_csv(os.path.join(methyl_dir, 'brain_metadata_top500.csv'), index_col=0)

    print(f"Blood methylation: {blood_meth.shape[0]} samples, {blood_meth.shape[1]} CpGs")
    print(f"Blood metadata: {blood_meta.shape[0]} samples")
    print(f"Brain methylation: {brain_meth.shape[0]} samples, {brain_meth.shape[1]} CpGs")
    print(f"Brain metadata: {brain_meta.shape[0]} samples")

except Exception as e:
    print(f"Error loading methylation data: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit

# ============================================================================
# Section 4: Load CpG-Gene Mappings
# ============================================================================

print("\nLoading CpG-gene mappings...")

try:
    blood_map = pd.read_csv(os.path.join(top500_dir, 'blood_top500_cpg_gene_map.csv'))
    brain_map = pd.read_csv(os.path.join(top500_dir, 'brain_top500_cpg_gene_map.csv'))

    exclude_terms = ['NO_GENE', 'NO_ANNOTATION', 'NOT_FOUND', 'NOT_IN_ANNOTATION']
    blood_map = blood_map[~blood_map['Gene'].isin(exclude_terms)]
    brain_map = brain_map[~brain_map['Gene'].isin(exclude_terms)]

    print(f"Blood mappings: {blood_map.shape[0]} rows")
    print(f"   Unique CpGs: {blood_map['CpG_ID'].nunique()}, Unique genes: {blood_map['Gene'].nunique()}")
    print(f"Brain mappings: {brain_map.shape[0]} rows")
    print(f"   Unique CpGs: {brain_map['CpG_ID'].nunique()}, Unique genes: {brain_map['Gene'].nunique()}")

except Exception as e:
    print(f"Error loading gene mappings: {e}")
    raise SystemExit

# ============================================================================
# Section 5: Load Annotation
# ============================================================================

print("\nLoading annotation...")

try:
    if not os.path.exists(annot_path):
        print(f"Annotation file not found: {annot_path}")
        print("   Run Step 9 first")
        raise SystemExit

    annot_df = pd.read_csv(annot_path, index_col=0)
    print(f"Annotation loaded: {annot_df.shape[0]} probes, {annot_df.shape[1]} columns")

    if 'CHR' in annot_df.columns:
        annot_df['CHR'] = annot_df['CHR'].astype(str).str.replace('chr', '', case=False, regex=False)

    blood_cpgs_in_data = list(set(blood_map['CpG_ID']) & set(blood_meth.columns))
    brain_cpgs_in_data = list(set(brain_map['CpG_ID']) & set(brain_meth.columns))

    blood_annot = annot_df.loc[annot_df.index.intersection(blood_cpgs_in_data)]
    brain_annot = annot_df.loc[annot_df.index.intersection(brain_cpgs_in_data)]

    print(f"Filtered blood annotation: {blood_annot.shape[0]} CpGs")
    print(f"Filtered brain annotation: {brain_annot.shape[0]} CpGs")

except Exception as e:
    print(f"Error loading annotation: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit

# ============================================================================
# Section 6: Prioritize Genes with Multiple CpGs
# ============================================================================

print("\nPrioritizing genes with multiple CpG associations...")

def prioritize_genes(cpg_map, min_cpgs=2):
    counts = cpg_map['Gene'].value_counts()
    print(f"   Genes with 1 CpG: {(counts == 1).sum()}")
    print(f"   Genes with >=2 CpGs: {(counts >= 2).sum()}")
    print(f"   Genes with >=3 CpGs: {(counts >= 3).sum()}")

    top_genes = counts.head(10)
    print(f"\n   Top 10 genes by CpG count:")
    for gene, count in top_genes.items():
        print(f"   {gene:15}: {count} CpGs")

    prioritized = counts[counts >= min_cpgs].index.tolist()
    return prioritized, counts

blood_prioritized_genes, blood_counts = prioritize_genes(blood_map)
brain_prioritized_genes, brain_counts = prioritize_genes(brain_map)

print(f"\nBlood genes with >=2 CpGs: {len(blood_prioritized_genes)}")
print(f"Brain genes with >=2 CpGs: {len(brain_prioritized_genes)}")

pd.DataFrame({'Gene': blood_prioritized_genes}).to_csv(
    os.path.join(results_dir, 'blood_prioritized_genes.csv'), index=False
)
pd.DataFrame({'Gene': brain_prioritized_genes}).to_csv(
    os.path.join(results_dir, 'brain_prioritized_genes.csv'), index=False
)

# ============================================================================
# Section 7: DMR Clustering
# ============================================================================

print("\nClustering CpGs into DMRs...")

def cluster_dmrs(annot_subset, max_gap=1000):
    dmrs = []
    pos_col = next((col for col in ['MAPINFO', 'POSITION'] if col in annot_subset.columns), None)
    if pos_col is None:
        return pd.DataFrame()

    annot_subset = annot_subset.copy()
    annot_subset['CHR'] = annot_subset['CHR'].astype(str)

    def chrom_order(chrom):
        try:
            return int(chrom), chrom
        except ValueError:
            order = {'X': 100, 'Y': 101, 'MT': 102, 'M': 102}
            return order.get(chrom.upper(), 999), chrom

    sorted_chroms = sorted(annot_subset['CHR'].unique(), key=chrom_order)

    for chrom in sorted_chroms:
        chrom_data = annot_subset[annot_subset['CHR'] == chrom].sort_values(pos_col)
        if chrom_data.empty:
            continue

        positions = chrom_data[pos_col].values
        cpg_ids = chrom_data.index.values
        cluster_id = 0
        current_cluster = [cpg_ids[0]]
        current_start = positions[0]

        for i in range(1, len(cpg_ids)):
            if positions[i] - positions[i-1] <= max_gap:
                current_cluster.append(cpg_ids[i])
            else:
                dmrs.append({
                    'CHR': chrom,
                    'DMR_ID': f"chr{chrom}_DMR{cluster_id:03d}",
                    'Start': current_start,
                    'End': positions[i-1],
                    'Length': positions[i-1] - current_start,
                    'n_CpGs': len(current_cluster),
                    'CpG_List': current_cluster
                })
                cluster_id += 1
                current_cluster = [cpg_ids[i]]
                current_start = positions[i]

        if current_cluster:
            dmrs.append({
                'CHR': chrom,
                'DMR_ID': f"chr{chrom}_DMR{cluster_id:03d}",
                'Start': current_start,
                'End': positions[-1],
                'Length': positions[-1] - current_start,
                'n_CpGs': len(current_cluster),
                'CpG_List': current_cluster
            })

    return pd.DataFrame(dmrs)

blood_dmrs = cluster_dmrs(blood_annot)
brain_dmrs = cluster_dmrs(brain_annot)

print(f"\nBlood DMRs: {len(blood_dmrs)} clusters")
if len(blood_dmrs) > 0:
    print(f"   Average CpGs per DMR: {blood_dmrs['n_CpGs'].mean():.1f}")
    print(f"   Maximum CpGs in DMR: {blood_dmrs['n_CpGs'].max()}")
    print(f"   DMR length range: {blood_dmrs['Length'].min():.0f}-{blood_dmrs['Length'].max():.0f} bp")

print(f"\nBrain DMRs: {len(brain_dmrs)} clusters")
if len(brain_dmrs) > 0:
    print(f"   Average CpGs per DMR: {brain_dmrs['n_CpGs'].mean():.1f}")
    print(f"   Maximum CpGs in DMR: {brain_dmrs['n_CpGs'].max()}")
    print(f"   DMR length range: {brain_dmrs['Length'].min():.0f}-{brain_dmrs['Length'].max():.0f} bp")

# ============================================================================
# Section 8: Tissue-Specific Genes
# ============================================================================

print("\nIdentifying tissue-specific genes...")

blood_genes_set = set(blood_map['Gene'].unique())
brain_genes_set = set(brain_map['Gene'].unique())

shared_genes = blood_genes_set.intersection(brain_genes_set)
blood_only = blood_genes_set - shared_genes
brain_only = brain_genes_set - shared_genes

print(f"Total blood genes: {len(blood_genes_set)}")
print(f"Total brain genes: {len(brain_genes_set)}")
print(f"Shared genes: {len(shared_genes)}")
print(f"Blood-specific genes: {len(blood_only)}")
print(f"Brain-specific genes: {len(brain_only)}")

pd.DataFrame({'Gene': list(shared_genes)}).to_csv(os.path.join(results_dir, 'shared_genes.csv'), index=False)
pd.DataFrame({'Gene': list(blood_only)}).to_csv(os.path.join(results_dir, 'blood_specific_genes.csv'), index=False)
pd.DataFrame({'Gene': list(brain_only)}).to_csv(os.path.join(results_dir, 'brain_specific_genes.csv'), index=False)

# ============================================================================
# Section 9: Genomic Context Analysis with Visualization
# ============================================================================

print("\nAnalyzing genomic context...")

def genomic_context_analysis(cpg_map, annot_df, tissue_name):
    cpgs_with_annot = set(cpg_map['CpG_ID']) & set(annot_df.index)
    if not cpgs_with_annot:
        print(f"No CpGs found in annotation for {tissue_name}")
        return None

    cpg_annot = annot_df.loc[list(cpgs_with_annot)]
    feature_col = next((col for col in ['UCSC_REFGENE_GROUP', 'GENE_FEATURE', 'RELATIONSHIP'] if col in cpg_annot.columns), None)
    if feature_col is None:
        print(f"No gene feature column found for {tissue_name}")
        return None

    feature_counts = cpg_annot[feature_col].value_counts()
    total = len(cpg_annot)

    print(f"\n{tissue_name.upper()} genomic context (top 10):")
    for feature, count in feature_counts.head(10).items():
        percentage = (count / total) * 100
        print(f"   {feature:30}: {count:4d} ({percentage:.1f}%)")

    # Feature distribution plot
    plt.figure(figsize=(14, 8))
    top_features = feature_counts.head(15)
    bars = plt.bar(range(len(top_features)), top_features.values, color='steelblue', edgecolor='navy')
    plt.title(f'{tissue_name} - Distribution of CpGs by Genomic Feature')
    plt.xlabel('Genomic Feature')
    plt.ylabel('Number of CpGs')
    plt.xticks(range(len(top_features)), [str(f)[:30] for f in top_features.index], rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\n({height/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # Chromosome distribution plot
    if 'CHR' in cpg_annot.columns:
        chr_counts = cpg_annot['CHR'].astype(str).str.replace('chr', '', case=False).value_counts()

        def chrom_sort_key(chrom):
            try:
                return int(chrom), chrom
            except ValueError:
                order = {'X': 100, 'Y': 101, 'MT': 102, 'M': 102}
                return order.get(chrom.upper(), 999), chrom

        sorted_chroms = sorted(chr_counts.index, key=chrom_sort_key)
        chrom_data = [(f'Chr{c}', chr_counts[c]) for c in sorted_chroms if c in chr_counts]

        if chrom_data:
            chromosomes, counts = zip(*chrom_data)
            plt.figure(figsize=(16, 8))
            bars = plt.bar(range(len(chromosomes)), counts, color=['#3498db' if str(c).isdigit() else '#e74c3c' for c in chromosomes])
            plt.title(f'{tissue_name} - Distribution of CpGs by Chromosome')
            plt.xlabel('Chromosome')
            plt.ylabel('Number of CpGs')
            plt.xticks(range(len(chromosomes)), chromosomes, rotation=45)
            total_cpgs = sum(counts)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                percentage = (height / total_cpgs) * 100
                plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

    return feature_counts

blood_context = genomic_context_analysis(blood_map, annot_df, "Blood")
brain_context = genomic_context_analysis(brain_map, annot_df, "Brain")

# ============================================================================
# Section 10: Functional Enrichment Analysis
# ============================================================================

print("\nPerforming functional enrichment analysis...")

def functional_enrichment(gene_list, tissue_name, background_genes=None, min_genes=10):
    if len(gene_list) < min_genes:
        print(f"Skipping {tissue_name} enrichment: insufficient genes ({len(gene_list)})")
        return None

    print(f"Analyzing {tissue_name} genes ({len(gene_list)} genes)")
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(
        organism='hsapiens',
        query=list(gene_list),
        sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC', 'WP'],
        user_threshold=0.05,
        background=background_genes
    )

    if results.empty:
        print(f"No enrichment results for {tissue_name}")
        return None

    results['p_adj'] = multipletests(results['p_value'], method='fdr_bh')[1]
    significant = results[results['p_adj'] < 0.05].sort_values('p_adj')

    results.to_csv(os.path.join(results_dir, f'{tissue_name}_enrichment_all.csv'), index=False)
    significant.to_csv(os.path.join(results_dir, f'{tissue_name}_enrichment_significant.csv'), index=False)

    print(f"Significant terms: {len(significant)}")

    # Adjusted: For shared genes with only 1 term, print it out instead of visualizing
    if len(significant) == 1:
        print(f"\nTop enriched term for {tissue_name}:")
        row = significant.iloc[0]
        print(f"   Source: {row['source']}")
        print(f"   Term: {row['name']}")
        print(f"   p_adj: {row['p_adj']:.2e}")
        print(f"   Genes: {row.get('intersection', 'N/A')}")
    elif len(significant) > 0:
        top_terms = significant.head(15)
        plt.figure(figsize=(14, 10))
        top_terms['-log10(p_adj)'] = -np.log10(top_terms['p_adj'])
        y_pos = np.arange(len(top_terms))
        source_colors = {'GO:BP': '#3498db', 'GO:MF': '#e74c3c', 'GO:CC': '#f39c12',
                         'KEGG': '#9b59b6', 'REAC': '#1abc9c', 'WP': '#34495e'}
        bars = plt.barh(y_pos, top_terms['-log10(p_adj)'], color=[source_colors.get(s, '#95a5a6') for s in top_terms['source']])
        plt.xlabel('-log10(Adjusted p-value)')
        plt.title(f'{tissue_name} - Top Enriched Terms (FDR < 0.05)')
        plt.yticks(y_pos, [str(term)[:50] + '...' for term in top_terms['name']])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'p={top_terms.iloc[i]["p_adj"]:.1e}', va='center', fontsize=9)
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=source) for source, color in source_colors.items()]
        plt.legend(handles=legend_elements, title='Source', loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

    return significant

blood_enrich = functional_enrichment(list(blood_only), 'blood_specific')
brain_enrich = functional_enrichment(list(brain_only), 'brain_specific')
shared_enrich = functional_enrichment(list(shared_genes), 'shared_genes', min_genes=5)

# ============================================================================
# Section 11: Improved Methylation Heatmaps
# ============================================================================

print("\nGenerating improved methylation heatmaps...")

def plot_top_cpgs_heatmap(meth_df, cpg_list, tissue_name, top_n=25):
    selected_cpgs = cpg_list[:top_n] if len(cpg_list) > top_n else cpg_list
    common_cpgs = [cpg for cpg in selected_cpgs if cpg in meth_df.columns]
    if not common_cpgs:
        print(f"No common CpGs found for {tissue_name} heatmap")
        return

    data = meth_df[common_cpgs]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                       gridspec_kw={'height_ratios': [1, 3, 1]})

    # Box plot of methylation values
    bp = ax1.boxplot(data.values.flatten(), patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('navy')
    mean_val = np.mean(data.values.flatten())
    median_val = np.median(data.values.flatten())
    std_val = np.std(data.values.flatten())
    ax1.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.set_title(f'{tissue_name} - Methylation Value Distribution')
    ax1.set_ylabel('Beta Value')
    ax1.grid(True, alpha=0.3, axis='y')

    # Main heatmap
    im = ax2.imshow(data.T, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_title(f'{tissue_name} - Methylation Pattern (Top {len(common_cpgs)} CpGs)')
    ax2.set_ylabel('CpG Probes')
    ax2.set_xlabel('Samples')
    if len(common_cpgs) > 10:
        yticks = [0, len(common_cpgs)//2, len(common_cpgs)-1]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([common_cpgs[i][:15] for i in yticks])
    plt.colorbar(im, ax=ax2, label='Beta Value')

    # Sample correlation heatmap
    sample_corr = data.T.corr()
    sns.heatmap(sample_corr.iloc[:20, :20], cmap='RdYlBu_r', center=0, ax=ax3,
                cbar_kws={'label': 'Correlation'})
    ax3.set_title(f'{tissue_name} - Sample Correlation (First 20 Samples)')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Samples')

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'{tissue_name}_methylation_heatmap_top{top_n}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_top_cpgs_heatmap(blood_meth, blood_map['CpG_ID'].tolist(), 'Blood')
plot_top_cpgs_heatmap(brain_meth, brain_map['CpG_ID'].tolist(), 'Brain')

# ============================================================================
# Section 12: DMR-Gene Relationship Analysis
# ============================================================================

print("\nAnalyzing DMR-gene relationships...")

def analyze_dmr_gene_relationships(dmr_df, cpg_map, tissue_name):
    multi_cpg_dmrs = dmr_df[dmr_df['n_CpGs'] >= 2]
    if multi_cpg_dmrs.empty:
        print(f"No DMRs with multiple CpGs for {tissue_name}")
        return None

    dmr_genes = []
    for _, row in multi_cpg_dmrs.iterrows():
        cpgs = row['CpG_List']
        genes = set(cpg_map[cpg_map['CpG_ID'].isin(cpgs)]['Gene'])
        if genes:
            dmr_genes.append({
                'DMR_ID': row['DMR_ID'],
                'CHR': row['CHR'],
                'n_CpGs': row['n_CpGs'],
                'Length_bp': row['Length'],
                'Associated_Genes': ','.join(sorted(genes)),
                'n_Genes': len(genes)
            })

    if dmr_genes:
        df = pd.DataFrame(dmr_genes)
        df.to_csv(os.path.join(results_dir, f'{tissue_name}_dmr_gene_relationships.csv'), index=False)

        plt.figure(figsize=(14, 8))
        plt.scatter(df['Length_bp'], df['n_Genes'],
                   s=df['n_CpGs']*20, c=df['n_CpGs'], cmap='viridis', alpha=0.7, edgecolors='black')
        plt.xlabel('DMR Length (bp)')
        plt.ylabel('Number of Associated Genes')
        plt.title(f'{tissue_name} - DMR Length vs Gene Associations')
        plt.colorbar(label='Number of CpGs')
        plt.yticks(np.arange(1, 3.1, 0.5))
        plt.ylim(0.5, max(df['n_Genes']) + 0.5 if not df.empty else 3.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return df
    return None

blood_dmr_genes = analyze_dmr_gene_relationships(blood_dmrs, blood_map, "Blood")
brain_dmr_genes = analyze_dmr_gene_relationships(brain_dmrs, brain_map, "Brain")

# ============================================================================
# Section 13: Comprehensive Tissue Comparison Visualization
# ============================================================================

print("\nCreating comprehensive tissue comparison visualization...")

fig = plt.figure(figsize=(18, 12))

# Panel 1: Gene counts
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(['Blood-Specific', 'Brain-Specific', 'Shared'],
               [len(blood_only), len(brain_only), len(shared_genes)],
               color=['#e74c3c', '#3498db', '#2ecc71'])
ax1.set_title('Gene Counts by Tissue Specificity')
ax1.set_ylabel('Number of Genes')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')

# Panel 2: DMR counts
ax2 = plt.subplot(2, 3, 2)
bars = ax2.bar(['Blood DMRs', 'Brain DMRs'], [len(blood_dmrs), len(brain_dmrs)],
               color=['#e74c3c', '#3498db'])
ax2.set_title('Differentially Methylated Regions')
ax2.set_ylabel('Number of DMRs')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center')

# Panel 3: Multi-CpG genes
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(['Blood (>=2 CpGs)', 'Brain (>=2 CpGs)'],
               [len(blood_prioritized_genes), len(brain_prioritized_genes)],
               color=['#c0392b', '#2980b9'])
ax3.set_title('Genes with Multiple CpG Associations')
ax3.set_ylabel('Number of Genes')

# Panel 4: DMR length distribution
ax4 = plt.subplot(2, 3, 4)
if len(blood_dmrs) > 0 and len(brain_dmrs) > 0:
    bp = ax4.boxplot([blood_dmrs['Length'].values, brain_dmrs['Length'].values],
                     labels=['Blood', 'Brain'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax4.set_title('DMR Length Distribution')
    ax4.set_ylabel('Length (bp)')

# Panel 5: Top genomic features comparison
ax5 = plt.subplot(2, 3, 5)
if blood_context is not None and brain_context is not None:
    top_blood = blood_context.head(5)
    top_brain = brain_context.head(5)
    all_features = set(top_blood.index).union(top_brain.index)
    blood_vals = [top_blood.get(f, 0) for f in all_features]
    brain_vals = [top_brain.get(f, 0) for f in all_features]
    x = np.arange(len(all_features))
    width = 0.35
    ax5.bar(x - width/2, blood_vals, width, label='Blood', color='#e74c3c')
    ax5.bar(x + width/2, brain_vals, width, label='Brain', color='#3498db')
    ax5.set_title('Top Genomic Features')
    ax5.set_ylabel('Number of CpGs')
    ax5.set_xticks(x)
    ax5.set_xticklabels([str(f)[:15] + '...' for f in all_features], rotation=45, ha='right')
    ax5.legend()

# Panel 6: Summary text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
TISSUE COMPARISON SUMMARY

BLOOD
- Total genes: {len(blood_genes_set)}
- Tissue-specific: {len(blood_only)}
- Multi-CpG genes: {len(blood_prioritized_genes)}
- DMRs: {len(blood_dmrs)}

BRAIN
- Total genes: {len(brain_genes_set)}
- Tissue-specific: {len(brain_only)}
- Multi-CpG genes: {len(brain_prioritized_genes)}
- DMRs: {len(brain_dmrs)}

SHARED
- Common genes: {len(shared_genes)}
"""
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.suptitle('Comprehensive Tissue Comparison: Blood vs Brain', fontsize=16)
comp_path = os.path.join(results_dir, 'comprehensive_tissue_comparison.png')
plt.savefig(comp_path, dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Section 14: Save Results
# ============================================================================

print("\nSaving results...")

if len(blood_dmrs) > 0:
    blood_dmrs.to_csv(os.path.join(results_dir, 'blood_dmrs_detailed.csv'), index=False)
if len(brain_dmrs) > 0:
    brain_dmrs.to_csv(os.path.join(results_dir, 'brain_dmrs_detailed.csv'), index=False)

gene_summary = pd.DataFrame({
    'Category': ['Blood_Total', 'Brain_Total', 'Shared', 'Blood_Specific', 'Brain_Specific',
                 'Blood_MultiCpG', 'Brain_MultiCpG'],
    'Count': [len(blood_genes_set), len(brain_genes_set), len(shared_genes),
              len(blood_only), len(brain_only),
              len(blood_prioritized_genes), len(brain_prioritized_genes)]
})
gene_summary.to_csv(os.path.join(results_dir, 'gene_counts_summary.csv'), index=False)

# ============================================================================
# Section 15: Comprehensive Checklist and Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("Step 10 is complete")
print("=" * 80)

print("\nAnalysis objectives completed:")
print("- Loaded and processed top 500 CpG methylation data for both tissues")
print("- Mapped CpGs to genes and identified tissue-specific patterns")
print("- Clustered CpGs into differentially methylated regions (DMRs)")
print("- Prioritized genes with multiple CpG associations")
print("- Analyzed genomic context and feature distribution")
print("- Performed functional enrichment analysis")
print("- Generated comprehensive visualizations including heatmaps and comparisons")

print(f"\nKey findings:")
print(f"   Blood genes: {len(blood_genes_set)} total ({len(blood_only)} tissue-specific)")
print(f"   Brain genes: {len(brain_genes_set)} total ({len(brain_only)} tissue-specific)")
print(f"   Shared genes: {len(shared_genes)}")
print(f"   Blood DMRs: {len(blood_dmrs)}")
print(f"   Brain DMRs: {len(brain_dmrs)}")

print(f"\nAll results saved to: {results_dir}")
print("=" * 80)
