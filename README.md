# Анализ тональности

Этот репозиторий был создан для решения тестового задания по анализу тональности текстов от компании TWIN. Было решено использовать нейронную сеть Bert, предобученную на задаче анализа тональности текста разных языков. В итоге обученная модель показала результат более 0.84 по F1 мере, на выборке из 30к текстов из различных русских наборов данных.

## Зависимости

```bash
pandas
numpy 
argparse
pymorphy2
nltk
transformers
sklearn 
torch 
tqdm 
flask  
requests
```

## Использование
Необходимо поместить веса модели в папку model.\
Скачать параметры модели можно по этой ссылке: 
https://drive.google.com/file/d/18c-M-C1ql8nUv275n-iQkhNHGNWAWQXf/view?usp=sharing

Далее нужно запустить скрипт server.py, если работаете из docker контейнера то пробросьте порт 5000.

```python
python3 server.py 
```
с этого момента web приложению можно отправлять json запросы вида:

```json
{'text': "Текст, тональность которого нужно определить"}
```

ответом сервера будет объект json вида:

```json
{'label': "тональность текста"}
```

## Самостоятельная тренировка модели
### Подготовка датасета
В интернете для задачи анализа тональности текста на русском языке я нашел следующие наборы данных:

- mokoron-sentiment \
http://study.mokoron.com/
- kaggle-sentiment \
https://www.kaggle.com/c/sentiment-analysis-in-russian/data
- medical-sentiment \
https://github.com/blanchefort/datasets/tree/master/medical_comments
- rureviews-sentiment \
https://github.com/sismetanin/rureviews
- rusentiment dataset \
https://drive.google.com/file/d/1fSldAI6zgU6_2K0g7AiQ6UxLwi-3Bq2z/view?usp=sharing

Для того чтобы подготовить итоговый датасет необходимо скачать перечисленные наборы данных, поместить в соответствующие папки и запустить скрипт prepare_dataset.py следующим образом:

```python
python3 prepare_dataset.py --data_directory "путь до папки с наборами данных"\
--val_split 0.1 #размер валидационной выборки 
--mokoron #использование набора данных mokoron
--rureviews #использование набора данных rureviews
--medical #использование набора данных medical
--rusentiment #использование набора данных rusentiment
--kaggle #использование набора данных kaggle
```

После этого в папке с наборами данных появится два jsonl файла (train.jsonl, val.jsonl) со следующей структурой:

```json
[
{"text": "Какой то текст", "label": "тональность этого текста"},
...
]
```

### Тренировка модели

Далее нужно запустить скрипт train_model.py:

```python
python3 train_model.py --data_directory "путь до папки с наборами данных"\
--hug_path "путь расположения модели bert на сервере huggingface"
--batch_size 10 #размер батча
--lr 0.00002 #начальный learning rate
--epochs 20 #число эпох обучения
--warmup 100 #число warmup шагов
--loss_print 200 #число шагов обучения перед выводом loss-а
--val_steps 10000 #число шагов обучения перед валидацией
--to_lower #использовать ли только нижний регистр
--do_filter #производить ли фильтрацию символов по словарю
--remove_rep #удалять ли повторения символов в текстах
--remove_stops #удалять ли стоп-слова из текстов
--remove_punkt #удалять ли символы пунктуации 
--lemmatize #приводить ли слова к начальной форме 
--stemming #использовать ли стемминг
```
после обучения в папке проекта появятся веса модели model.pth

### Проверка точности модели

Для проверки точности модели нужно запустить скрипт test_model.py:

```python
python3 test_model.py --data_directory "путь до папки с наборами данных"\
--hug_path "путь расположения модели bert на сервере huggingface"
--local_path "путь расположения локальной модели bert"
--batch_size 10 #размер батча
--to_lower #использовать ли только нижний регистр
--do_filter #производить ли фильтрацию символов по словарю
--remove_rep #удалять ли повторения символов в текстах
--remove_stops #удалять ли стоп-слова из текстов
--remove_punkt #удалять ли символы пунктуации 
--lemmatize #приводить ли слова к начальной форме 
--stemming #использовать ли стемминг
```

