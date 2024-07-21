from PIL import Image, ImageFilter
import numpy as np


def resize_crop(img, h, w):
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
    if isinstance(image_name, str):
        return Image.open(f'input/{image_name}')
    elif isinstance(image_name, Image.Image):
        return image_name
    else:
        assert False, 'Image must be defined as int, str or Image!'
        

def get_vertices(gs):
    vertices = np.zeros((256, 12, 3))
    gs = np.array(gs).flatten()
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
                        vertices[t, index] = np.array([2, 2, 2])
                    index += 1
    return vertices


def color_blur(image_name, 
               blur_factor=0.1,
               output_name='output', 
               grayscale=[0.2126, 0.7152, 0.0722], 
               test_mode=False):
    image = open_img(image_name)
    resolution = np.sqrt(np.prod(image.size))
    blur_radius = blur_factor * resolution / 2
    blured = image.filter(ImageFilter.GaussianBlur(blur_radius))
    return transform(image_name=blured, 
                     target=image, 
                     output_name=output_name, 
                     grayscale=grayscale, 
                     test_mode=test_mode)


def illumination(image_name,
                 intensity=0.9,
                 color=[0, 0, 255],
                 output_name='output',
                 test_mode=False):
    image = open_img(image_name)
    color = list(256 - np.array(color))
    return transform(image_name=image, 
                     target=1 - intensity, 
                     output_name=output_name, 
                     grayscale=color, 
                     test_mode=test_mode)


def transform(image_name,
              target=0.2,
              output_name='output',
              grayscale=[0.2126, 0.7152, 0.0722],
              test_mode=False):
    # Преобразование коэффициентов учета RGB в вектор-строку
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
    
    # Точки, полученные пересечением всех пар границ с Плоскостью
    vertices = get_vertices(gs)[t_index]
    
    # Индексы внутри RGB для рассчетов
    ks = (np.eye(6, k=-1) + np.eye(6))[..., ::2]
    # Границы (правая часть в уравнениях их плоскостей)
    alphas = np.reshape([0, 1] * 3, (6, 1))
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
    