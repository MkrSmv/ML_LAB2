# Лабораторная работа 2 (ИАД25: Смирнов, Корунов)

Система распознавания именованных сущностей на основе CRF-модели. 

`REPORT.md` - отчет, для **ФОРМАТИРОВАНИЯ** использовался ИИ.

**Macro F1 = 0.9467** на обучающей выборке.


### Быстрый старт

```bash
git clone https://github.com/MkrSmv/ML_LAB2
cd ML_LAB2
python -m venv venv
.\venv\Scripts\Activate.ps1 # source venv/bin/activate
pip install -r requirements.txt
```



### Запуск

#### 1. Обучение NER-модели и генерация submission

```bash
python src/ner_pipeline.py
```

#### 2. Система управления сущностями (демо)

```bash
python src/entity_manager.py
```

#### 3. Визуализации

```bash
python src/visualizations.py
```
