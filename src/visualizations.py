import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from collections import defaultdict
import os
import textwrap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

TAG_MAP = {
    0: 'O',
    1: 'B-per', 2: 'I-per',
    3: 'B-gpe', 4: 'I-gpe',
    5: 'B-eve', 6: 'I-eve',
    7: 'B-geo', 8: 'I-geo',
    9: 'B-nat', 10: 'I-nat',
    11: 'B-art', 12: 'I-art',
    13: 'B-tim', 14: 'I-tim',
    15: 'B-org', 16: 'I-org',
}

CATEGORY_COLORS = {
    'per': '#e74c3c',
    'gpe': '#3498db',
    'eve': '#9b59b6',
    'geo': '#2ecc71',
    'nat': '#f39c12',
    'art': '#e91e63',
    'tim': '#00bcd4',
    'org': '#ff9800',
}

CATEGORY_NAMES = {
    'per': 'PERSON',
    'gpe': 'GPE',
    'eve': 'EVENT',
    'geo': 'LOCATION',
    'nat': 'NATURAL',
    'art': 'ART',
    'tim': 'TIME/DATE',
    'org': 'ORGANIZATION',
}


def generate_word_clouds(train_path=None, output_dir=None):
    """Word Cloud для каждой NER-категории из обучающей выборки."""
    train_path = train_path or os.path.join(DATA_DIR, 'train.csv')
    output_dir = output_dir or os.path.join(OUTPUT_DIR, 'wordclouds')
    os.makedirs(output_dir, exist_ok=True)

    print("\n  Генерация Word Clouds...")

    df = pd.read_csv(train_path)
    df['Sentence_id'] = df['Sentence_id'].ffill()
    df['tag_name'] = df['Tag'].map(TAG_MAP)

    category_words = defaultdict(list)
    for _, row in df.iterrows():
        tag = row['tag_name']
        if tag != 'O':
            cat = tag[2:]
            word = str(row['Word'])
            if len(word) > 1:
                category_words[cat].append(word)

    # Общая картинка 2x4
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Word Clouds по NER-категориям', fontsize=20, fontweight='bold', y=1.02)

    cats = sorted(CATEGORY_COLORS.keys())
    for idx, cat in enumerate(cats):
        ax = axes[idx // 4][idx % 4]
        words = category_words.get(cat, [])

        if words:
            wc = WordCloud(width=600, height=400, background_color='white',
                           colormap='viridis', max_words=100, random_state=42
                           ).generate(' '.join(words))
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f"{CATEGORY_NAMES.get(cat, cat)} ({len(words)} слов)",
                         fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(CATEGORY_NAMES.get(cat, cat), fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_wordclouds.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Отдельные файлы
    for cat in cats:
        words = category_words.get(cat, [])
        if words:
            wc = WordCloud(width=800, height=500, background_color='white',
                           colormap='viridis', max_words=150, random_state=42
                           ).generate(' '.join(words))
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title(f"Word Cloud: {CATEGORY_NAMES.get(cat, cat)}",
                          fontsize=16, fontweight='bold')
            fig2.savefig(os.path.join(output_dir, f'wordcloud_{cat}.png'),
                         dpi=150, bbox_inches='tight')
            plt.close(fig2)

    print(f"  Word Clouds сохранены в {output_dir}/")

    print("\n  Статистика по категориям:")
    for cat in cats:
        words = category_words.get(cat, [])
        print(f"    {CATEGORY_NAMES.get(cat, cat):>15}: "
              f"{len(words):>7} слов, {len(set(words)):>5} уникальных")

    return category_words


def generate_ner_highlighting(train_path=None, output_path=None, n_sentences=5):
    """Matplotlib-визуализация с подсветкой NER-тегов цветом."""
    train_path = train_path or os.path.join(DATA_DIR, 'train.csv')
    output_path = output_path or os.path.join(OUTPUT_DIR, 'ner_highlight.png')

    print(f"\n  Генерация подсветки NER ({n_sentences} предложений)...")

    df = pd.read_csv(train_path)
    df['Sentence_id'] = df['Sentence_id'].ffill()
    df['tag_name'] = df['Tag'].map(TAG_MAP)

    # Выбираем предложения с наибольшим числом сущностей
    sent_entity_count = df[df['tag_name'] != 'O'].groupby('Sentence_id').size()
    top_sents = sent_entity_count.nlargest(n_sentences).index.tolist()

    fig, axes = plt.subplots(n_sentences, 1, figsize=(18, 2.5 * n_sentences))
    if n_sentences == 1:
        axes = [axes]

    fig.suptitle('NER Highlighting - примеры из обучающей выборки',
                 fontsize=16, fontweight='bold', y=1.01)

    for ax_idx, sid in enumerate(top_sents):
        ax = axes[ax_idx]
        sent_df = df[df['Sentence_id'] == sid]
        tokens = sent_df['Word'].astype(str).tolist()
        tags = sent_df['tag_name'].tolist()

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Токены с подсветкой
        x, y = 0.01, 0.65
        fontsize = 11
        line_height = 0.35

        for token, tag in zip(tokens, tags):
            if tag != 'O':
                cat = tag[2:]
                color = CATEGORY_COLORS.get(cat, '#cccccc')
                label = CATEGORY_NAMES.get(cat, cat)
                txt = ax.text(x, y, token, fontsize=fontsize, fontweight='bold',
                              va='center', ha='left',
                              bbox=dict(boxstyle='round,pad=0.15',
                                        facecolor=color, alpha=0.25,
                                        edgecolor=color, linewidth=1.5))
            else:
                txt = ax.text(x, y, token, fontsize=fontsize,
                              va='center', ha='left', color='#333333')

            # Вычисляем ширину текста для сдвига
            fig.canvas.draw()
            bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax.transData.inverted())
            x = bbox_data.x1 + 0.005

            if x > 0.95:
                x = 0.01
                y -= line_height

    # Легенда
    legend_patches = [mpatches.Patch(color=CATEGORY_COLORS[cat], alpha=0.4,
                                     label=CATEGORY_NAMES[cat])
                      for cat in sorted(CATEGORY_COLORS.keys())]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=len(legend_patches), fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Подсветка NER: {output_path}")


def plot_tag_distribution(train_path=None, output_path=None):
    """Столбиковая диаграмма распределения NER-тегов (без O)."""
    train_path = train_path or os.path.join(DATA_DIR, 'train.csv')
    output_path = output_path or os.path.join(OUTPUT_DIR, 'tag_distribution.png')

    print("\n  Генерация диаграммы распределения...")

    df = pd.read_csv(train_path)
    df['tag_name'] = df['Tag'].map(TAG_MAP)
    tag_counts = df[df['tag_name'] != 'O']['tag_name'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [CATEGORY_COLORS.get(tag[2:], '#999') for tag in tag_counts.index]

    bars = ax.bar(range(len(tag_counts)), tag_counts.values, color=colors)
    ax.set_xticks(range(len(tag_counts)))
    ax.set_xticklabels(tag_counts.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Количество', fontsize=12)
    ax.set_title('Распределение NER-тегов (без O)', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, tag_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Диаграмма: {output_path}")


def main():
    print("=" * 60)
    print("  NER Визуализации")
    print("=" * 60)

    generate_word_clouds()
    generate_ner_highlighting()
    plot_tag_distribution()

    print("\n  Визуализации готовы.")


if __name__ == '__main__':
    main()
