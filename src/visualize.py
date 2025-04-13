#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import tempfile

class ForensicVisualizer:
    """Class for visualizing patterns in potentially manipulated data."""
    
    def __init__(self, output_dir=None):
        """Initialize the visualizer with an optional output directory."""
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_suspicious_vs_normal(self, df, group_col, outcome_vars, suspicious_rows, title=None):
        """
        Create plots comparing suspicious vs normal observations for key outcome variables.
        
        Args:
            df: DataFrame with the data
            group_col: Column name for grouping/condition variable
            outcome_vars: List of outcome variable column names
            suspicious_rows: List of row indices for suspicious observations
            title: Optional title for the plot
            
        Returns:
            Path to saved plot file
        """
        n_vars = len(outcome_vars)
        fig, axes = plt.subplots(n_vars, 2, figsize=(14, 5 * n_vars))
        
        # If only one outcome variable, make axes 2D
        if n_vars == 1:
            axes = axes.reshape(1, -1)
            
        # Create mask for suspicious rows
        suspicious_mask = df.index.isin(suspicious_rows)
        
        # Title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle("Comparison of Suspicious vs. Normal Observations", fontsize=16)
            
        for i, var in enumerate(outcome_vars):
            # Plot suspicious observations
            suspicious_df = df[suspicious_mask].copy()
            normal_df = df[~suspicious_mask].copy()
            
            # Left plot - suspicious observations
            sns.boxplot(x=group_col, y=var, data=suspicious_df, ax=axes[i, 0])
            axes[i, 0].set_title(f"Suspicious Observations (n={len(suspicious_df)})")
            
            # Add swarmplot to see individual points
            sns.swarmplot(x=group_col, y=var, data=suspicious_df, ax=axes[i, 0], color='black', alpha=0.7)
            
            # Calculate and display means
            group_means = suspicious_df.groupby(group_col)[var].mean()
            for j, group in enumerate(suspicious_df[group_col].unique()):
                if group in group_means:
                    axes[i, 0].text(j, group_means[group], f"Mean: {group_means[group]:.2f}", 
                                  ha='center', va='bottom', fontweight='bold')
            
            # Right plot - normal observations
            sns.boxplot(x=group_col, y=var, data=normal_df, ax=axes[i, 1])
            axes[i, 1].set_title(f"Normal Observations (n={len(normal_df)})")
            
            # For larger datasets, use smaller point size or sample
            if len(normal_df) > 100:
                sample_df = normal_df.sample(min(100, len(normal_df)))
                sns.swarmplot(x=group_col, y=var, data=sample_df, ax=axes[i, 1], 
                            color='black', alpha=0.7)
            else:
                sns.swarmplot(x=group_col, y=var, data=normal_df, ax=axes[i, 1], 
                            color='black', alpha=0.7)
            
            # Calculate and display means
            group_means = normal_df.groupby(group_col)[var].mean()
            for j, group in enumerate(normal_df[group_col].unique()):
                if group in group_means:
                    axes[i, 1].text(j, group_means[group], f"Mean: {group_means[group]:.2f}", 
                                  ha='center', va='bottom', fontweight='bold')
            
            # Add variable name as row label
            fig.text(0.01, 0.5 + (0.5 / n_vars) * (n_vars - 1 - 2 * i), var, 
                     va='center', ha='left', fontsize=14, rotation=90)
        
        plt.tight_layout(rect=[0.02, 0, 1, 0.97])
        
        # Save the plot
        if self.output_dir:
            filename = os.path.join(self.output_dir, "suspicious_vs_normal.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            return filename
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                plt.close()
                return tmp.name
    
    def plot_effect_sizes(self, df, group_col, outcome_vars, suspicious_rows):
        """
        Create a plot showing effect sizes for suspicious vs. normal observations.
        
        Args:
            df: DataFrame with the data
            group_col: Column name for grouping/condition variable
            outcome_vars: List of outcome variable column names
            suspicious_rows: List of row indices for suspicious observations
            
        Returns:
            Path to saved plot file
        """
        # Create mask for suspicious rows
        suspicious_mask = df.index.isin(suspicious_rows)
        
        # Calculate effect sizes
        effect_sizes = []
        
        for var in outcome_vars:
            try:
                # Skip if this variable is not numeric
                if not pd.api.types.is_numeric_dtype(df[var]):
                    print(f"Skipping non-numeric variable: {var}")
                    continue
                
                # Suspicious observations
                suspicious_df = df[suspicious_mask]
                group_vals = suspicious_df[group_col].unique()
                
                if len(group_vals) < 2:
                    continue
                    
                # Get the two groups (assuming binary condition)
                group_a = group_vals[0]
                group_b = group_vals[1]
                
                # Check if we have enough data in each group
                a_suspicious = suspicious_df[suspicious_df[group_col] == group_a][var]
                b_suspicious = suspicious_df[suspicious_df[group_col] == group_b][var]
                
                if len(a_suspicious) < 1 or len(b_suspicious) < 1:
                    continue
                
                # Get means for suspicious observations
                mean_a_suspicious = a_suspicious.mean()
                mean_b_suspicious = b_suspicious.mean()
                effect_suspicious = abs(mean_a_suspicious - mean_b_suspicious)
                
                # Get means for normal observations
                normal_df = df[~suspicious_mask]
                a_normal = normal_df[normal_df[group_col] == group_a][var]
                b_normal = normal_df[normal_df[group_col] == group_b][var]
                
                if len(a_normal) < 1 or len(b_normal) < 1:
                    continue
                    
                mean_a_normal = a_normal.mean()
                mean_b_normal = b_normal.mean()
                effect_normal = abs(mean_a_normal - mean_b_normal)
                
                # Calculate Cohen's d for both
                def cohens_d(a, b):
                    if len(a) <= 1 or len(b) <= 1:
                        return 0
                    
                    n1, n2 = len(a), len(b)
                    s1, s2 = a.std(), b.std()
                    
                    # Check for zero variance
                    if s1 == 0 and s2 == 0:
                        return 0
                        
                    try:
                        s_pooled = np.sqrt(((n1-1) * s1**2 + (n2-1) * s2**2) / (n1 + n2 - 2))
                        return (a.mean() - b.mean()) / s_pooled if s_pooled != 0 else 0
                    except (ValueError, ZeroDivisionError):
                        return 0
                
                d_suspicious = cohens_d(a_suspicious, b_suspicious)
                d_normal = cohens_d(a_normal, b_normal)
                
                effect_sizes.append({
                    'Variable': var,
                    'Effect Size Type': 'Raw Difference (Suspicious)',
                    'Effect Size': effect_suspicious
                })
                
                effect_sizes.append({
                    'Variable': var,
                    'Effect Size Type': 'Raw Difference (Normal)',
                    'Effect Size': effect_normal
                })
                
                effect_sizes.append({
                    'Variable': var,
                    'Effect Size Type': "Cohen's d (Suspicious)",
                    'Effect Size': d_suspicious
                })
                
                effect_sizes.append({
                    'Variable': var,
                    'Effect Size Type': "Cohen's d (Normal)",
                    'Effect Size': d_normal
                })
                
            except Exception as e:
                print(f"Error analyzing {var}: {str(e)}")
                continue
        
        if not effect_sizes:
            return None
            
        # Create DataFrame for plotting
        effect_df = pd.DataFrame(effect_sizes)
        
        try:
            # Plot
            plt.figure(figsize=(14, 8))
            
            # Raw differences
            plt.subplot(1, 2, 1)
            raw_df = effect_df[effect_df['Effect Size Type'].str.contains('Raw Difference')]
            sns.barplot(x='Variable', y='Effect Size', hue='Effect Size Type', data=raw_df)
            plt.title("Raw Mean Differences")
            plt.xticks(rotation=45)
            
            # Cohen's d
            plt.subplot(1, 2, 2)
            d_df = effect_df[effect_df['Effect Size Type'].str.contains("Cohen's d")]
            sns.barplot(x='Variable', y='Effect Size', hue='Effect Size Type', data=d_df)
            plt.title("Effect Sizes (Cohen's d)")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            if self.output_dir:
                filename = os.path.join(self.output_dir, "effect_sizes.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                return filename
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                    plt.close()
                    return tmp.name
        except Exception as e:
            print(f"Error creating effect size plot: {str(e)}")
            return None
    
    def plot_id_sequence(self, df, id_col, group_col, suspicious_rows=None):
        """
        Plot the sequence of IDs to visualize sorting anomalies.
        
        Args:
            df: DataFrame with the data
            id_col: Column name for ID variable
            group_col: Column name for grouping/condition variable
            suspicious_rows: Optional list of row indices for suspicious observations
            
        Returns:
            Path to saved plot file
        """
        # Create a copy with row numbers
        plot_df = df.copy()
        plot_df['row_number'] = range(len(plot_df))
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot by group
        groups = plot_df[group_col].unique()
        colors = sns.color_palette("husl", len(groups))
        
        for i, group in enumerate(groups):
            group_df = plot_df[plot_df[group_col] == group]
            plt.scatter(group_df['row_number'], group_df[id_col], 
                       label=f"{group_col}={group}", color=colors[i], alpha=0.7)
            
            # Connect points in this group with lines
            plt.plot(group_df['row_number'], group_df[id_col], 
                    color=colors[i], alpha=0.3)
        
        # Highlight suspicious rows if provided
        if suspicious_rows is not None:
            suspicious_df = plot_df[plot_df.index.isin(suspicious_rows)]
            plt.scatter(suspicious_df['row_number'], suspicious_df[id_col], 
                       color='red', s=100, marker='x', label='Suspicious')
        
        plt.xlabel("Row Number in Dataset")
        plt.ylabel(f"Participant ID ({id_col})")
        plt.title("ID Sequence Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for suspicious points
        if suspicious_rows is not None:
            for idx, row in suspicious_df.iterrows():
                plt.annotate(f"ID {row[id_col]}", 
                           (row['row_number'], row[id_col]),
                           xytext=(5, 5), textcoords='offset points')
        
        # Save the plot
        if self.output_dir:
            filename = os.path.join(self.output_dir, "id_sequence.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            return filename
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                plt.close()
                return tmp.name