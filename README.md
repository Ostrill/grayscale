# Преобразование цвета в один тон

Здесь представлен алгоритм, который позволяет преобразовать цвета изображения таким образом, чтобы при конвертации в режим оттенков серого картинка исчезала.

<blockquote>

Например, если конвертировать обычное изображение в режим оттенков серого, получится примерно следующее:

<div style="display: flex; flex-direction: row; flex-wrap: nowrap; flex-flow: row nowrap; justify-content: flex-start; align-items: center; margin: 0 0 1rem 0">
    <img src="assets/original.png" style="height: 8rem;">
    <div style="font-size: 1.5rem; margin: 0 0.5rem;">→</div>
    <img src="assets/grays.png" style="height: 8rem;">
</div>

Но если перед конвертацией в режим оттенков серого применить представленный алгоритм, то получится следующий результат:

<div style="display: flex; flex-direction: row; flex-wrap: nowrap; flex-flow: row nowrap; justify-content: flex-start; align-items: center; margin: 0 0 1rem 0">
    <img src="assets/original.png" style="height: 8rem;">
    <div style="font-size: 1.5rem; margin: 0 0.5rem;">→</div>
    <img src="assets/transform.png" style="height: 8rem;">
    <div style="font-size: 1.5rem; margin: 0 0.5rem;">→</div>
    <img src="assets/empty.png" style="height: 8rem;">
</div>

То есть картинка посередине не может быть конвертирована в оттенки серого, вместо ожидаемого результата получается однотонное пустое изображение.

</blockquote>

Для конвертации цветного изображения с каналами ($R$, $G$, $B$) в режим оттенков серого с одним каналом ($L$), как правило, используют взвешенное среднее:

$L=0.2126R + 0.7152G + 0.0722B$

Данные коэффициенты, взятые из стандарта ITU-R BT.709, учитывают разную чувствительность человеческого глаза к тем или иным цветам и являются наиболее распространенными для преобразования изображений в оттенки серого.

Таким образом, изображение становится в некотором смысле однотонным, поскольку при конвертации в режим оттенков серого каждый его пиксель преобразуется в одно и то же число.