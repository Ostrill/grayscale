from PIL import Image
import numpy as np


def prepare_input(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    img1_ratio = img1.size[0] / img1.size[1]
    img2_ratio = img2.size[0] / img2.size[1]
    
    if img2_ratio > img1_ratio:
        img2 = img2.resize((int(img1.size[1] * img2_ratio),
                            img1.size[1]))
        crop_x = round(img2.size[0] / 2 - img1.size[0] / 2)
        img2 = img2.crop((crop_x, 
                          0, 
                          crop_x + img1.size[0], 
                          img2.size[1]))
    else:
        img2 = img2.resize((img1.size[0], 
                            int(img1.size[0] / img2_ratio)))
        crop_y = round(img2.size[1] / 2 - img1.size[1] / 2)
        img2 = img2.crop((0,
                          crop_y, 
                          img2.size[0], 
                          crop_y + img1.size[1]))
        
    return img1, img2


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


def transform(image, 
              target=0.5,
              grayscale=[0.2126, 0.7152, 0.0722]):
    # Масштабирование цветов в диапазон [0, 1]
    image_np = np.array(image)[..., :3] / 255
    # Размеры исходного изображения
    h, w, _ = image_np.shape
    # Преобразование в вектор-строку
    gs = np.reshape(grayscale, (1, 3))
    gs = gs / gs.sum()
    # Формулирование цели для алгоритма
    if isinstance(target, float):
        t = target
        t_index = np.full((h, w), round(t * 255))
    else:
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
    nc = options[best_indices]
    # Итоговый результат
    return Image.fromarray(np.round(nc * 255).astype('uint8'))