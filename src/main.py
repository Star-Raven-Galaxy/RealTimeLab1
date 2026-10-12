"""
Практическая работа № 1 (СРВ, Python).
Классификация систем реального времени и расчёт требуемого времени реакции.

Читает набор задач из JSON/CSV, классифицирует их по жёсткости и типу дедлайна,
вычисляет запас, коэффициент загрузки, критическую задачу, требуемое время
реакции и архитектурный класс системы.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------

Hardness = Literal["hard", "firm", "soft"]

HARDNESS_RU: dict[Hardness, str] = {
    "hard": "жёсткое",
    "firm": "твёрдое",
    "soft": "мягкое",
}


@dataclass(frozen=True)
class Task:
    """Описание одной задачи реального времени (все времена — в миллисекундах)."""

    name: str
    hardness: Hardness
    C: float  # время выполнения
    D: float  # относительный дедлайн
    T: float  # период


@dataclass(frozen=True)
class SystemConfig:
    """Конфигурация вычислительной системы."""

    n_cpu: int
    has_network: bool


# ---------------------------------------------------------------------------
# Функции расчёта (по одной на каждый пункт задания)
# ---------------------------------------------------------------------------


def slack(task: Task) -> float:
    """Запас задачи: L = D − C."""
    return task.D - task.C


def is_feasible(task: Task) -> bool:
    """Выполнимость дедлайна задачи: L ≥ 0."""
    return slack(task) >= 0


def deadline_type(task: Task) -> str:
    """Тип дедлайна по соотношению D и T."""
    if task.D == task.T:
        return "неявный"
    if task.D < task.T:
        return "ограниченный"
    return "произвольный"


def classify_system(tasks: Iterable[Task]) -> str:
    """Класс системы в целом: жёсткого РВ, если есть хотя бы одна жёсткая задача."""
    tasks = list(tasks)
    if any(t.hardness == "hard" for t in tasks):
        return "жёсткого реального времени"
    return "мягкого реального времени"


def critical_task(tasks: Iterable[Task]) -> Task | None:
    """Критическая задача — с наименьшим запасом среди жёстких.

    Если жёстких задач нет — возвращается None.
    """
    hard = [t for t in tasks if t.hardness == "hard"]
    if not hard:
        return None
    return min(hard, key=slack)


def required_reaction_time(tasks: Iterable[Task]) -> float:
    """R_треб = min(D) по жёстким задачам; при их отсутствии — по всем."""
    tasks = list(tasks)
    hard = [t for t in tasks if t.hardness == "hard"]
    pool = hard if hard else tasks
    if not pool:
        raise ValueError("Список задач пуст — невозможно вычислить R_треб")
    return min(t.D for t in pool)


def utilization(tasks: Iterable[Task]) -> float:
    """Коэффициент загрузки U = Σ (C_i / T_i)."""
    tasks = list(tasks)
    if not tasks:
        return 0.0
    return sum(t.C / t.T for t in tasks)


def classify_architecture(n_cpu: int, has_network: bool) -> str:
    """Архитектурный класс системы."""
    if has_network:
        return "распределённая"
    if n_cpu <= 1:
        return "однопроцессорная"
    return "многопроцессорная"


# ---------------------------------------------------------------------------
# Ввод данных
# ---------------------------------------------------------------------------


def load_from_json(path: Path) -> tuple[SystemConfig, list[Task]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg_raw = raw["system"]
    cfg = SystemConfig(n_cpu=int(cfg_raw["n_cpu"]),
                       has_network=bool(cfg_raw["has_network"]))

    tasks = [
        Task(
            name=item["name"],
            hardness=item["class"],
            C=float(item["C"]),
            D=float(item["D"]),
            T=float(item["T"]),
        )
        for item in raw["tasks"]
    ]
    return cfg, tasks


def load_from_csv(path: Path) -> tuple[SystemConfig, list[Task]]:
    """CSV с шапкой: name,class,C,D,T и отдельная строка-комментарий с системой.

    Ожидаемый формат:
        # system: n_cpu=1; has_network=false
        name,class,C,D,T
        ...
    """
    n_cpu = 1
    has_network = False
    tasks: list[Task] = []

    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    data_lines: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("#"):
            body = ln.lstrip("#").strip()
            if body.lower().startswith("system:"):
                kv = body.split(":", 1)[1]
                for part in kv.split(";"):
                    k, _, v = part.partition("=")
                    k, v = k.strip().lower(), v.strip().lower()
                    if k == "n_cpu":
                        n_cpu = int(v)
                    elif k == "has_network":
                        has_network = v in ("true", "1", "yes")
        else:
            data_lines.append(ln)

    reader = csv.DictReader(data_lines)
    for row in reader:
        tasks.append(
            Task(
                name=row["name"],
                hardness=row["class"].strip(),
                C=float(row["C"]),
                D=float(row["D"]),
                T=float(row["T"]),
            )
        )

    return SystemConfig(n_cpu=n_cpu, has_network=has_network), tasks


def load_data(path: Path) -> tuple[SystemConfig, list[Task]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_from_json(path)
    if suffix == ".csv":
        return load_from_csv(path)
    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------


def fmt(value: float) -> str:
    """Формат числа: целое без дробной части, иначе — 3 знака."""
    return f"{value:g}"


def print_table(tasks: list[Task]) -> None:
    headers = ["Задача", "Класс", "C", "D", "T", "L", "Выполнимо", "Тип дедлайна"]
    rows: list[list[str]] = []
    for t in tasks:
        rows.append([
            t.name,
            HARDNESS_RU[t.hardness],
            fmt(t.C),
            fmt(t.D),
            fmt(t.T),
            fmt(slack(t)),
            "да" if is_feasible(t) else "нет",
            deadline_type(t),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(render_row(headers))
    print(sep)
    for row in rows:
        print(render_row(row))
    print(sep)


def print_conclusion(cfg: SystemConfig, tasks: list[Task]) -> None:
    sys_class = classify_system(tasks)
    crit = critical_task(tasks)
    r_req = required_reaction_time(tasks)
    u = utilization(tasks)
    arch = classify_architecture(cfg.n_cpu, cfg.has_network)

    all_feasible = all(is_feasible(t) for t in tasks)

    print()
    print("Заключение")
    print("----------")
    print(f"Класс системы:                {sys_class}")
    if crit is not None:
        print(f"Критическая задача:           {crit.name} "
              f"(L = {fmt(slack(crit))} мс)")
    else:
        print("Критическая задача:           отсутствует (нет жёстких задач)")
    print(f"Требуемое время реакции R_треб: {fmt(r_req)} мс")
    print(f"Коэффициент загрузки U:       {u:.3f}")
    print(f"Необходимое условие U ≤ 1:    {'выполнено' if u <= 1 else 'НАРУШЕНО'}")
    print(f"Все дедлайны выполнимы:       {'да' if all_feasible else 'нет'}")
    print(f"Архитектурный класс:          {arch}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР №1: классификация СРВ и расчёт времени реакции")
    parser.add_argument("path", type=Path, help="Путь к файлу данных (.json или .csv)")
    args = parser.parse_args()

    cfg, tasks = load_data(args.path)
    print_table(tasks)
    print_conclusion(cfg, tasks)


if __name__ == "__main__":
    main()