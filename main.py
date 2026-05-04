"""
Имитационная модель производственного участка
Вариант 10, ДР: 19.03
Симуляция выполнена с помощью библиотеки SimPy (процессно-ориентированный подход).

Ресурсы:
    - 2 обрабатывающих центра (simpy.Resource, capacity=2)
    - 1 робот-манипулятор   (simpy.Resource, capacity=1)

Потоки заявок:
    - Генерация заготовок: random.triangular(100.0, 122.0, 100.0)  [левотреугольное]
    - Перемещение роботом:  random.uniform(10.0, 13.0)
    - Обработка на станке:  random.uniform(178.0, 200.0)
"""

import simpy
import random

# ══════════════════════════════════════════════════════════
#  Параметры модели
# ══════════════════════════════════════════════════════════
RANDOM_SEED  = 7          # зерно для воспроизводимости
SIM_TIME     = 28_800     # горизонт моделирования, с (8 ч)
NUM_MACHINES = 2          # обрабатывающих центров
NUM_ROBOTS   = 1          # роботов-манипуляторов

# Треугольное распределение (левотреугольное: мода = левый конец)
ARRIVAL_LOW  = 100.0
ARRIVAL_HIGH = 122.0
ARRIVAL_MODE = 100.0

# Равномерное распределение времени перемещения роботом
ROBOT_MIN  = 10.0
ROBOT_MAX  = 13.0

# Равномерное распределение времени обработки на станке
MACHINE_MIN = 178.0
MACHINE_MAX = 200.0


# ══════════════════════════════════════════════════════════
#  Ручной сбор статистики
# ══════════════════════════════════════════════════════════
class Statistics:
    """
    Ручное накопление четырёх метрик:
        1. Коэффициент загрузки робота
        2. Коэффициент загрузки станков
        3. Средняя длина очереди в бункере
        4. Среднее время ожидания детали в очереди
    """

    def __init__(self, env: simpy.Environment):
        self.env = env

        # ── загрузка ресурсов ──────────────────────────────────────────
        self.robot_busy_time   = 0.0   # сумма времён перемещений роботом
        self.machine_busy_time = 0.0   # сумма времён обработки на станках

        # ── очередь бункера (интеграл длины очереди по времени) ────────
        self.queue_length    = 0       # текущее число деталей в бункере
        self.queue_area      = 0.0     # накопленная площадь: Σ L(t)·Δt
        self.last_event_time = 0.0     # момент последнего изменения очереди

        # ── время ожидания деталей в бункере ──────────────────────────
        self.total_wait_time = 0.0     # суммарное время ожидания
        self.parts_entered   = 0       # деталей, вошедших в бункер

        # ── общие счётчики ─────────────────────────────────────────────
        self.parts_generated = 0
        self.parts_completed = 0

    # ── фиксирует площадь под кривой до текущего момента ──────────────
    def _flush_area(self):
        now = self.env.now
        self.queue_area      += self.queue_length * (now - self.last_event_time)
        self.last_event_time  = now

    # ── деталь входит в бункер (начало ожидания) ──────────────────────
    def hopper_enter(self):
        self._flush_area()
        self.queue_length += 1
        self.parts_entered += 1

    # ── деталь покидает бункер (робот захватил для загрузки) ──────────
    def hopper_leave(self, wait_seconds: float):
        self._flush_area()
        self.queue_length    -= 1
        self.total_wait_time += wait_seconds

    # ── итоговые метрики ───────────────────────────────────────────────
    def robot_utilization(self) -> float:
        """Доля времени, когда робот занят перемещениями."""
        return self.robot_busy_time / SIM_TIME

    def machine_utilization(self) -> float:
        """Средняя доля занятости каждого станка."""
        return self.machine_busy_time / (SIM_TIME * NUM_MACHINES)

    def avg_queue_length(self) -> float:
        """Средняя длина очереди в бункере (L_q = интеграл / T)."""
        # дополняем площадь «хвостом»: детали, ещё ожидающие в момент T
        tail_area  = self.queue_length * (SIM_TIME - self.last_event_time)
        return (self.queue_area + tail_area) / SIM_TIME

    def avg_wait_time(self) -> float:
        """Среднее время ожидания одной детали в бункере (W_q)."""
        if self.parts_entered == 0:
            return 0.0
        return self.total_wait_time / self.parts_entered


# ══════════════════════════════════════════════════════════
#  Процесс одной детали
# ══════════════════════════════════════════════════════════
TRACE_LIMIT = 20  # трассировка первых N деталей

def trace(env, part_id, event):
    if part_id <= TRACE_LIMIT:
        print(f"  {env.now:10.2f}  | Деталь {part_id:3d} | {event}")


def part_process(
        env:      simpy.Environment,
        part_id:  int,
        robot:    simpy.Resource,
        machines: simpy.Resource,
        stats:    Statistics,
):
    """
    Жизненный цикл заготовки. Строго соответствует 10-шаговой логике:
        1  → бункер (начало ожидания)
        2  → захват робота
        3  → перемещение роботом к станку
        4  → освобождение робота
        5  → захват станка
        6  → обработка
        7  → освобождение станка
        8  → захват робота для выгрузки
        9  → перемещение роботом на конвейер
        10 → освобождение робота, выход из системы
    """

    # ── Шаг 1. Деталь попадает в бункер ──────────────────────────────
    stats.hopper_enter()
    entered_hopper_at = env.now
    trace(env, part_id, "бункер (начало ожидания)")

    # ── Шаг 2. Захват робота (ожидание, если занят) ───────────────────
    with robot.request() as rob_req:
        yield rob_req
        trace(env, part_id, "захват робота")

        # ── Шаг 3–4. Перемещение к станку + автоосвобождение робота ──
        stats.hopper_leave(env.now - entered_hopper_at)

        travel_time = random.uniform(ROBOT_MIN, ROBOT_MAX)
        trace(env, part_id, f"перемещение к станку ({travel_time:.2f} с)")
        yield env.timeout(travel_time)
        stats.robot_busy_time += travel_time
        trace(env, part_id, "освобождение робота")
    # робот свободен

    # ── Шаг 5. Захват обрабатывающего центра ─────────────────────────
    with machines.request() as mc_req:
        yield mc_req
        trace(env, part_id, "захват станка")

        # ── Шаг 6–7. Обработка + автоосвобождение станка ─────────────
        process_time = random.uniform(MACHINE_MIN, MACHINE_MAX)
        trace(env, part_id, f"обработка ({process_time:.2f} с)")
        yield env.timeout(process_time)
        stats.machine_busy_time += process_time
        trace(env, part_id, "освобождение станка")
    # станок свободен

    # ── Шаг 8. Захват робота для выгрузки ────────────────────────────
    with robot.request() as rob_req:
        yield rob_req
        trace(env, part_id, "захват робота (выгрузка)")

        # ── Шаг 9–10. Перемещение на конвейер + освобождение ─────────
        travel_time = random.uniform(ROBOT_MIN, ROBOT_MAX)
        trace(env, part_id, f"перемещение на конвейер ({travel_time:.2f} с)")
        yield env.timeout(travel_time)
        stats.robot_busy_time += travel_time
        trace(env, part_id, "освобождение робота, выход из системы")
    # робот свободен, деталь покидает систему

    stats.parts_completed += 1


# ══════════════════════════════════════════════════════════
#  Генератор потока заготовок
# ══════════════════════════════════════════════════════════
def part_generator(
        env:      simpy.Environment,
        robot:    simpy.Resource,
        machines: simpy.Resource,
        stats:    Statistics,
):
    """
    Бесконечно генерирует заготовки с интервалом, подчинённым
    левотреугольному распределению triangular(100, 122, 100).
    Каждая заготовка запускается как отдельный SimPy-процесс.
    """
    part_id = 0
    while True:
        # интервал между появлениями заготовок
        inter_arrival = random.triangular(ARRIVAL_LOW, ARRIVAL_HIGH, ARRIVAL_MODE)
        yield env.timeout(inter_arrival)

        stats.parts_generated += 1
        part_id += 1
        env.process(
            part_process(env, part_id, robot, machines, stats)
        )


# ══════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════
def main():
    random.seed(RANDOM_SEED)

    env      = simpy.Environment()
    robot    = simpy.Resource(env, capacity=NUM_ROBOTS)
    machines = simpy.Resource(env, capacity=NUM_MACHINES)
    stats    = Statistics(env)

    sep = "=" * 46
    print(sep)
    print(f"{'ТРАССИРОВКА (первые 20 деталей)':^46}")
    print(sep)
    print(f"  {'Время':>10}  | {'Деталь':^10} | Событие")
    print(sep)

    env.process(part_generator(env, robot, machines, stats))
    env.run(until=SIM_TIME)

    # ── Вывод четырёх ключевых метрик ─────────────────────────────────
    sep = "=" * 46
    print(sep)
    print(f"{'РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ':^46}")
    print(sep)
    print(f"  Горизонт моделирования      : {SIM_TIME} с")
    print(f"  Деталей сгенерировано        : {stats.parts_generated}")
    print(f"  Деталей полностью обработано : {stats.parts_completed}")
    print(sep)
    print(f"  1. Коэф. загрузки робота     : {stats.robot_utilization():.4f}")
    print(f"  2. Коэф. загрузки станков    : {stats.machine_utilization():.4f}")
    print(f"  3. Сред. длина оч. бункера   : {stats.avg_queue_length():.4f} дет.")
    print(f"  4. Сред. время ожидания      : {stats.avg_wait_time():.2f} с")
    print(sep)


if __name__ == "__main__":
    main()
