from PIL import Image, ImageFilter
import numpy as np


def resize_crop(img, h, w):
    """
    Масштабирование и обрезка изображения img 
    под переданные размеры (h, w)
    """
    img_w, img_h = img.size
    ratio = img_w / img_h
    if ratio > (w / h):
        new_w = int(h * ratio)
        img = img.resize((new_w, h))
        crop_x = round(new_w / 2 - w / 2)
        img = img.crop((crop_x, 0, crop_x + w, h))
    else:
        new_h = int(w / ratio)
        img = img.resize((w, new_h))
        crop_y = round(new_h / 2 - h / 2)
        img = img.crop((0, crop_y, w, crop_y + h))
    return img


def correct_filename(filename, default='png'):
    """
    Исправить название сохраняемого файла filename:
    - если в названии файла нет расширения или оно 
      не поддерживается, добавить расширение default;
    - если в названии файла есть расширение, и оно
      поддерживается, оставить filename как есть
    """
    # Список поддерживаемых расширений
    supported = {ex[1:] for ex, f 
                 in Image.registered_extensions().items() 
                 if f in Image.OPEN}
    # Разделенное по точкам название файла
    sep = filename.split('.')
    if len(sep) == 1 or sep[-1] not in supported:
        # Если у файла нет расширения или оно неправильное
        return f'{filename}.{default}'
    else:
        # Если с названием файла все в порядке
        return filename


def open_img(image_name):
    """
    Открыть изображение image_name, которое может быть
    представлено в виде пути к файлу (str) или готового
    изображения (Image.Image)
    """
    if isinstance(image_name, str):
        return Image.open(f'input/{image_name}')
    elif isinstance(image_name, Image.Image):
        return image_name
    else:
        assert False, 'Image must be defined as str or Image!'
        

def get_vertices(grayscale):
    """
    Найти все точки пересечения каждых двух смежных плоскостей-границ 
    с плоскостью преобразования, заданной через grayscale.
    Всего задано шесть плоскостей-границ, из которых ровно 12 смежных,
    поэтому в результате получается 12 точек. Для каждого возможного
    значения target (от 0 до 255) осуществляется поиск этих 12 точек,
    и результатом является общий массив размером (256, 12, 3)
    """
    vertices = np.zeros((256, 12, 3))
    gs = np.array(grayscale).flatten()
    for t in range(256):
        index = 0
        for i, j in ((0, 1), (1, 2), (0, 2)):
            for i_limit in (0, 1):
                for j_limit in (0, 1):
                    A = np.array([[*gs], 
                                  [*np.eye(3)[i]], 
                                  [*np.eye(3)[j]]])
                    b = np.array([t / 255, i_limit, j_limit])
                    try:
                        vertices[t, index] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        vertices[t, index] = np.array([100] * 3)
                    index += 1
    return vertices


def transform(image_name,
              target=0.2,
              output_name='output',
              grayscale=[0.2126, 0.7152, 0.0722],
              fast_mode=False,
              test_mode=False):
    """
    Основной метод для выравнивания цветов в плоскость grayscale

    PARAMETERS
    ----------
    image_name : str или Image.Image
         исходное изображение:
         - если str, то изображение с этим именем загружается
           из папки input/;
         - если Image.Image, то именно эта картинка берется
           в качестве исходного изображения.
    
    target : float или str или Image.Image
        цель преобразования, то естькакое значение должно 
        получиться в ходе преобразования в оттенки серого:
        - если float (от 0 до 1), то у каждого пикселя исходного 
          изображения будет именно такое значение после
          преобразование в оттенки серого;
        - если str, то по указнному пути загружается картинка,
          которая будет смасштабирована и обрезана по размерам
          исходной картинки, и в таком случае у каждого пикселя
          будет своя собственная цель (взятая с картинки target);
        - если Image.Image, то переданная картинка по аналогии
          с предыдущим пунктом будет смасштабирована и обрезана
          по размерам исходной картинки, чтобы использоваться
          в качестве мульти-цели для преобразования.
    
    output_name : str или None
        имя файла для сохранения. Если не указать расширение,
        по умолчанию будет использовано PNG. Если указать None,
        итоговое изображение не будет сохраняться на диск
    
    grayscale : [float, float, float]
        коэффициенты для преобразования в оттенки серого. 
        По умолчанию установлены наиболее часто используемые
        для этой цели коэффициенты из стандарта ITU-R BT.709

    fast_mode : bool
        переключатель быстрого режима:
        - если False, используется стандартный алгоритм
          преобразования цветов, изменяя их минимальным образом;
        - если True, используется более быстрый алгоритм 
          (примерно в 2 раза быстрее стандартного), но цвета не
          всегда изменяются минимально возможным образом, хоть
          картинка и слабо отличается от стандартного алгоритма.
    
    test_mode : bool или float
        переключатель для тестового режима. В тестовом режиме:
        - во-первых, размер обрабатываемого изображения
          сокращается до небольшого разрешения (если передан
          float, то это значение и берется, если bool - 100);
        - во-вторых, итоговое изображение не будет никуда
          сохраняться.
    """
    # Трансформирование коэффициентов преобразования в вектор-строку
    gs = np.reshape(grayscale, (1, 3))
    gs = gs / gs.sum()
    
    # Загрузка и обработка исходного изображения
    image = open_img(image_name)
    if test_mode: # Уменьшение картинки для тестового режима
        test_mode = 100 if isinstance(test_mode, bool) else test_mode
        new_w = np.sqrt(np.divide(*image.size)) * test_mode
        image.thumbnail((new_w, -1))
        output_name = None
    image_np = np.array(image)[..., :3] / 255
    h, w, _ = image_np.shape
    
    # Формулирование цели для алгоритма
    if isinstance(target, float):
        t = target
        t_index = np.full((h, w), round(t * 255))
    else:
        target = open_img(target)
        target = resize_crop(target, h, w)
        target_np = np.array(target)[..., :3] / 255
        t = target_np.reshape((h, w, 1, 3)) @ gs.T
        t_index = np.round(t.reshape(h, w) * 255).astype(int)
        
    # Массив из цветов исходного изображения
    colors = image_np.reshape((h, w, 1, 3))
    
    # Идеальные цвета для цели (пересечение Плоскости и нормали к ней)
    ci = gs * (t - colors @ gs.T) / (gs @ gs.T) + colors

    # Индексы внутри RGB для рассчетов
    ks = (np.eye(6, k=-1) + np.eye(6))[..., ::2]
    # Границы (правая часть в уравнениях их плоскостей)
    alphas = np.reshape([0, 1] * 3, (6, 1))
    
    if fast_mode:
        # Точки пересечения вектора ((ci - t), ci) с границами
        _cis = ci.repeat(6, axis=2)
        _kci = (ks.reshape(6, 1, 3) @ _cis[..., None]).reshape(h, w, 6, 1)
        xs = (_cis - t) * (alphas - t) / (_kci - t) + t
        
        # Все варианты расположения наилучшей точки
        options = np.concatenate([ci, xs], axis=-2)
    else:
        # Точки, полученные пересечением всех пар границ с Плоскостью
        vertices = get_vertices(gs)[t_index]
        
        # Направляющие вектора от ci к пересечениям Плоскости и границ для RGB
        ms = np.cross(gs.flatten(), np.cross(gs.flatten(), ks))
        # Точки, лежащие на пересечениях Плоскости и границ для RGB
        _ksms = (ks[..., None, :] @ ms[..., None]).reshape(6, 1)
        modas = ms * (alphas - ks @ ci.reshape((h, w, 3, 1))) / _ksms + ci
        
        # Все варианты расположения наилучшей точки
        options = np.concatenate([ci, vertices, modas], axis=-2)
    
    # Отсеивание точек за пределами 0 и 255
    _casted = np.round(options * 255)
    mask = np.any(_casted < 0, axis=-1) | np.any(_casted > 255, axis=-1)
    options[mask] = np.array([100] * 3)
    # Выбор лучшего варианта
    _ds = np.sum((options - ci) ** 2, axis=-1)
    best_indices = (np.arange(h)[:, None, None], 
                    np.arange(w)[None, :, None], 
                    np.argmin(_ds, axis=-1)[..., None], 
                    np.arange(3))
    result = options[best_indices]

    # Итоговый результат
    result = Image.fromarray(np.round(result * 255).astype('uint8'))

    # Сохранение изображения при необходимости
    if isinstance(output_name, str):
        filename = correct_filename(output_name)
        result.save(f'output/{filename}')
        print(f'Изображение сохранено в output/{filename}')
        
    return result


def color_blurring(image_name, 
                   blur_factor=0.1,
                   output_name='output', 
                   grayscale=[0.2126, 0.7152, 0.0722],
                   fast_mode=False, 
                   test_mode=False):
    """
    Эффект размытия цветов

    PARAMETERS
    ----------
    blur_factor : float
        коэффициент размытия от 0 до 1 (но можно и больше)

    image_name, output_name, grayscale, fast_mode, test_mode:
        параметры функции transform()
    """
    image = open_img(image_name)
    resolution = np.sqrt(np.prod(image.size))
    blur_radius = blur_factor * resolution / 2
    blured = image.filter(ImageFilter.GaussianBlur(blur_radius))
    return transform(image_name=blured, 
                     target=image, 
                     output_name=output_name, 
                     grayscale=grayscale, 
                     fast_mode=fast_mode, 
                     test_mode=test_mode)


def illumination(image_name,
                 color=[0, 0, 255],
                 intensity=0.1,
                 output_name='output',
                 fast_mode=False, 
                 test_mode=False):
    """
    Эффект цветного освещения

    PARAMETERS
    ----------
    color : [int, int, int]
        цвет освещения в формате RGB от 0 до 255
        
    intensity : float
        интенсивность остальных цветов от 0 до 1

    image_name, output_name, fast_mode, test_mode:
        параметры функции transform()
    """
    image = open_img(image_name)
    inverted_color = 255 - np.array(color)
    color = list(np.clip(inverted_color, 1e-10, 255))
    return transform(image_name=image, 
                     target=intensity, 
                     output_name=output_name, 
                     grayscale=color, 
                     fast_mode=fast_mode, 
                     test_mode=test_mode)
