import os
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme for a modern/impressive look
plt.style.use('dark_background')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

def generate_impressive_results_chart():
    # Data provided by the user
    models = [
        'Markov\nChain', 
        'TransE /\nRotatE', 
        'GAT /\nR-GCN', 
        'Temporal\nGNN', 
        'LLM-only\n(Base)', 
        'Gemini\n(LLM)'
    ]
    
    hits_at_1 = [1.72, 41.38, 3.85, 3.08, 3.33, 3.335]
    hits_at_3 = [3.45, 87.93, 10.77, 9.23, 3.33, 3.335]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#121212')
    ax.set_facecolor('#1e1e1e')

    # Vibrant colors
    color1 = '#00e5ff' # Cyan for Hits@1
    color2 = '#ff007f' # Neon Pink for Hits@3
    
    # Adding glow effect by plotting multiple times with decreasing alpha and increasing linewidth
    for i in range(1, 4):
        ax.bar(x - width/2, hits_at_1, width, color=color1, alpha=0.1*i, edgecolor='none')
        ax.bar(x + width/2, hits_at_3, width, color=color2, alpha=0.1*i, edgecolor='none')

    rects1 = ax.bar(x - width/2, hits_at_1, width, label='Hits@1 (%)', color=color1, edgecolor='white', linewidth=1)
    rects2 = ax.bar(x + width/2, hits_at_3, width, label='Hits@3 (%)', color=color2, edgecolor='white', linewidth=1)

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='white', labelpad=15)
    ax.set_title('Next-TTP Prediction Performance', fontsize=20, fontweight='heavy', color='white', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold', color='white')
    
    # Customize grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.5)

    legend = ax.legend(fontsize=12, loc='upper left', frameon=True, facecolor='#121212', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='w')

    # Add labels on top of bars
    def autolabel(rects, is_hits3=False):
        for rect in rects:
            height = rect.get_height()
            color = color2 if is_hits3 else color1
            # Add a slight drop shadow text
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(1, 4), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=11, color='black')
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=11, color=color)

    autolabel(rects1, False)
    autolabel(rects2, True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "readme_results_chart.png"), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    plt.close()

if __name__ == "__main__":
    generate_impressive_results_chart()
    print(f"Generated readme_results_chart.png in {fig_dir}")
