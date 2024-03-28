# blob-detection

## Введение
В последнее время обработка изображений все чаще использу- ется при решении теоретических и прикладных задач. Резуль- татом обработки изображения является его изменение (напри- мер цветовая коррекция, сглаживание) или получение ценной информации (например распознавание текста, сегментация). В частности, обработка изображений может быть использована и в фотолитографии, например, для анализа SEM-изображе- ний (SEM-scanning electron microscope). В процессе фотолито- графии происходит перенос изображения с фотошаблона пла- стины с ограничением минимально воспроизводимых размеров. Воспроизведение в фоторезистивной маске элементов с крити- ческими размерами, меньшими длины волны экспонирующего излучения, требует применения техник повышения разрешения (RET — resolution enhancement technology). Среди RET наиболее часто используемым является проведение коррекции оптиче- ской близости (OPC — optical proximity correction).

Распознавание контуров структур на основе SEM-сним- ков поможет значительно сэкономить измерительные ресурсы в задаче калибровки модели фоторезиста. В процессе обработки сложных двумерных топологических конфигураций возникает необходимость измерить два и более элемента одной тестовой структуры. Это влечет за собой написание нескольких рецептов измерения, что значительно увеличивает временные затраты. При проведении экстракции контуров необходимости в этом нет, т.к. для всех элементов, попавших в поле зрения, можно экстрагировать контур и использовать его при калибровке модели фоторезиста.

## Необходимость удаления выбросов
На используемых SEM-изображениях в процессе обра- ботки нередко можно обнаружить шум или выбросы, кото- рые могут привести к некорректной экстракции контуров. В компьютерном зрении выбросами принято считать области цифрового изображения, которые отли- чаются по таким параметрам, как цвет или яркость от окружающих областей. При этом в области самого выброса свой- ства изображения можно в некотором приближении считать постоянными. Пример негативного влияния выбросов отражен на рис. 2. Для более наглядной визуализации выбросы были перекра- шены в красный цвет. Из-за присутству- ющих выбросов в контуре появляются дополнительные элементы — небольшие «ответвления» от основной линии, что напрямую влияет на точность измерения CD.


## Алгоритм детектирования выбросов
Предлагаемый алгоритм обработки изображений позволяет болеее эффективно детектировать выбросы, чем существующие решения: BlobDetector и NLOG(Normalized Laplacian of Gaussian). 

В основе алгоритма лежит предположение о том, что т.н. выбросы — это обла- сти белых пикселей в окружении черных. Варьируя как размер самого выброса, так и размер окружения, было найдено оптимальное условие фильтрации пикселей изображения.


# Запуск программы

`usage: kernel_detection.py [-h] [--save_result] [--output OUTPUT] input`

Необходимо определить абсолютные пути до файла входного и выходного(в случае если нужно сохранить результат детекции выбросов) изоюарежний.

**Positional arguments**:

  **input**            Path for input file

**Optional arguments**:

 **-h, --help**       show this help message and exit
  
  **--save_result**    Save output result
  
  **--output** OUTPUT  Path for output file

  Пример 1:
  ```
python kernel_detection ~/input_image.jpg --save_result --output ~/output_image.jpg
```
На вход программы подаётся файл изображения `input_image.jpg`. Сохраняем результат изображение с обнаруженными выбросами в файл `output_image.jpg`.

Пример 2:
  ```
python kernel_detection ~/input_image.jpg 
```
На вход программы подаётся файл изображения `input_image.jpg`. Не сохраняем результата детектирования выбросов.









