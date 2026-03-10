import json
import os
import sys
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class EntityManager:
    """Управление NER-сущностями: CRUD, парсинг текстов, поиск связанных текстов."""

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

    def __init__(self, model_path=None, storage_path=None):
        self.model_path = model_path or os.path.join(OUTPUT_DIR, 'crf_model.pkl')
        self.storage_path = storage_path or os.path.join(OUTPUT_DIR, 'entity_store.json')
        self.entities = {}  # {name: {"category": str, "texts": [str]}}

        if os.path.exists(self.storage_path):
            self._load()

        self.crf = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.crf = pickle.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.entities, f, ensure_ascii=False, indent=2)

    def _load(self):
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            self.entities = json.load(f)

    def add_entity(self, name, category):
        """Добавить или обновить сущность."""
        key = name.strip()
        if key not in self.entities:
            self.entities[key] = {"category": category, "texts": []}
            print(f"  Добавлено: '{key}' [{category}]")
        else:
            old_cat = self.entities[key]["category"]
            self.entities[key]["category"] = category
            print(f"  Обновлено: '{key}' [{old_cat}] -> [{category}]")
        self._save()

    def delete_entity(self, name):
        """Удалить сущность."""
        key = name.strip()
        if key in self.entities:
            del self.entities[key]
            self._save()
            print(f"  Удалено: '{key}'")
        else:
            print(f"  Не найдено: '{key}'")

    def _tokenize(self, text):
        import re
        return re.findall(r"\w+|[^\w\s]", text)

    def _pos_tag(self, tokens):
        """Эвристическая POS-разметка."""
        tagged = []
        for tok in tokens:
            if tok[0].isupper() and len(tok) > 1:
                tagged.append((tok, 'NNP'))
            elif tok.isdigit():
                tagged.append((tok, 'CD'))
            elif tok in {'.', ',', '!', '?', ':', ';'}:
                tagged.append((tok, '.'))
            elif tok.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                tagged.append((tok, 'DT'))
            elif tok.lower() in {'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with',
                                 'of', 'about', 'between', 'through', 'during', 'after',
                                 'before', 'under', 'over', 'into'}:
                tagged.append((tok, 'IN'))
            elif tok.lower() in {'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                 'have', 'has', 'had', 'do', 'does', 'did'}:
                tagged.append((tok, 'VBZ'))
            elif tok.lower() in {'and', 'or', 'but', 'nor'}:
                tagged.append((tok, 'CC'))
            elif tok.lower() in {'he', 'she', 'it', 'they', 'we', 'i', 'you'}:
                tagged.append((tok, 'PRP'))
            else:
                tagged.append((tok, 'NN'))
        return tagged

    def _extract_entities(self, tokens, tags):
        """Извлечь сущности из IOB2-разметки."""
        entities = []
        current, current_cat = [], None

        for tok, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if current:
                    entities.append((' '.join(current), current_cat))
                current = [tok]
                current_cat = tag[2:]
            elif tag.startswith('I-') and current_cat == tag[2:]:
                current.append(tok)
            else:
                if current:
                    entities.append((' '.join(current), current_cat))
                    current, current_cat = [], None

        if current:
            entities.append((' '.join(current), current_cat))
        return entities

    def parse_text(self, text):
        """Разобрать текст NER-моделью, добавить найденные сущности в хранилище."""
        if self.crf is None:
            print("  Модель CRF не загружена. Сначала запустите ner_pipeline.py")
            return []

        tokens = self._tokenize(text)
        sent = self._pos_tag(tokens)

        from ner_pipeline import sent_to_features
        features = sent_to_features(sent)
        pred_tags = self.crf.predict([features])[0]

        found = self._extract_entities(tokens, pred_tags)

        for ent_name, ent_cat in found:
            if ent_name not in self.entities:
                self.entities[ent_name] = {"category": ent_cat, "texts": []}
            if text not in self.entities[ent_name]["texts"]:
                self.entities[ent_name]["texts"].append(text)

        self._save()
        return found

    def get_texts(self, entity_name):
        """Получить тексты, содержащие сущность."""
        key = entity_name.strip()
        if key in self.entities:
            return self.entities[key]["texts"]
        print(f"  Не найдено: '{key}'")
        return []

    def get_overview(self):
        """Обзор сущностей по категориям."""
        categories = {}
        for name, info in self.entities.items():
            cat = info["category"]
            categories.setdefault(cat, []).append(name)

        print("\n  Обзор NER-сущностей")
        print("  " + "-" * 40)
        for cat in sorted(categories):
            ents = categories[cat]
            print(f"\n  [{cat.upper()}] - {len(ents)} сущностей:")
            for e in sorted(ents)[:20]:
                n = len(self.entities[e]["texts"])
                suffix = f" ({n} текстов)" if n else ""
                print(f"    - {e}{suffix}")
            if len(ents) > 20:
                print(f"    ... и ещё {len(ents) - 20}")

    def __repr__(self):
        return f"EntityManager(entities={len(self.entities)})"


def demo():
    print("=" * 60)
    print("  Entity Manager - Демонстрация")
    print("=" * 60)

    em = EntityManager()

    print("\n--- Ручное добавление/удаление ---")
    em.add_entity("Moscow", "geo")
    em.add_entity("Vladimir Putin", "per")
    em.add_entity("United Nations", "org")
    em.add_entity("Test Entity", "per")
    print(f"  Сущности: {list(em.entities.keys())}")

    em.delete_entity("Test Entity")
    print(f"  После удаления: {list(em.entities.keys())}")

    print("\n--- Парсинг текста ---")
    texts = [
        "President Barack Obama met with German Chancellor Angela Merkel in Berlin on Monday.",
        "The United Nations Security Council held a meeting in New York on Tuesday.",
        "Apple Inc. announced new products at its headquarters in Cupertino, California.",
    ]
    for text in texts:
        print(f"\n  Текст: \"{text}\"")
        found = em.parse_text(text)
        print(f"  Найдено: {found}" if found else "  Сущности не найдены")

    print("\n--- Поиск текстов ---")
    for ent in list(em.entities.keys())[:3]:
        texts = em.get_texts(ent)
        print(f"\n  '{ent}': {len(texts)} текстов")
        for t in texts[:2]:
            print(f"    > {t[:80]}...")

    em.get_overview()
    print("\n  Готово.")


if __name__ == '__main__':
    demo()
