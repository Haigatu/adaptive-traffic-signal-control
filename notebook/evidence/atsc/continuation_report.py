"""Build the continuation report strictly from completed, saved runs."""

import base64
import io
import json
import platform
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .continuation import OUT, PROTOCOL, digest, save_json
from .data import ROOT
from .replay import TEMPLATE

LABELS = {
    "webster": "Webster",
    "max_pressure": "Max Pressure",
    "d3qn_original": "D3QN: исходный",
    "d3qn_guard_only": "D3QN: только ограничение",
    "d3qn_guard_trained": "D3QN: ограничение + обучение",
}


def build():
    rows = json.loads((OUT / "test_results.json").read_text())
    df = pd.DataFrame(rows)
    expected = {
        (name, seed, sat)
        for name in PROTOCOL["controllers"]
        for seed in PROTOCOL["test_seeds"]
        for sat in PROTOCOL["test_saturations"]
    }
    assert len(df) == len(expected) == 60
    assert set(zip(df.controller, df.seed, df.saturation)) == expected
    assert (df.elapsed_s == 3600).all()
    assert (df.teleports == 0).all() and (df.collision_vehicle_events == 0).all()
    guarded = df[df.controller.isin(["d3qn_guard_only", "d3qn_guard_trained"])]
    assert (guarded.max_without_green_s <= 120).all()
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["checkpoint_sha256"] == digest(OUT / "selected.pt")
    assert (
        df[df.controller == "d3qn_guard_trained"].controller_sha256
        == digest(OUT / "selected.pt")
    ).all()
    history = pd.DataFrame(json.loads((OUT / "training_history.json").read_text()))
    assert len(history) == 30
    validation = [
        json.loads(p.read_text()) for p in sorted(OUT.glob("validation_*.json"))
    ]
    assert len(validation) == 4
    df.to_csv(OUT / "raw_results.csv", index=False)
    numeric = [
        "delay_per_requested_vehicle_s",
        "throughput",
        "conflicts_per_1000_arrivals",
        "max_without_green_s",
        "mean_queue_vehicles",
        "pending_insertion",
        "guard_overrides",
    ]
    means = df.groupby(["controller", "saturation"])[numeric].mean()
    means.to_csv(OUT / "summary.csv")
    paired = []
    for sat in PROTOCOL["test_saturations"]:
        sub = df[df.saturation == sat].pivot(
            index="seed", columns="controller", values=numeric[0]
        )
        for base in ["d3qn_original", "d3qn_guard_only", "webster", "max_pressure"]:
            delta = sub.d3qn_guard_trained - sub[base]
            paired.append(
                {
                    "saturation": sat,
                    "comparison": base,
                    "paired_delay_differences_s": delta.tolist(),
                    "mean_difference_s": float(delta.mean()),
                    "relative_mean_change_pct": float(
                        100 * (sub.d3qn_guard_trained.mean() / sub[base].mean() - 1)
                    ),
                    "note": "Descriptive only: four demand seeds, one trained policy.",
                }
            )
    save_json(OUT / "paired_differences.json", paired)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = ["#95a5a6", "#386cb0", "#da6d57", "#b493d0", "#00876c"]
    x = np.arange(3)
    for i, name in enumerate(PROTOCOL["controllers"]):
        vals = [
            means.loc[(name, sat), numeric[0]] for sat in PROTOCOL["test_saturations"]
        ]
        axes[0].plot(x, vals, "o-", color=colors[i], label=LABELS[name])
    axes[0].set_xticks(x, ["0.40", "0.70", "0.95"])
    axes[0].set_xlabel("Номинальная нагрузка")
    axes[0].set_ylabel("Прокси задержки, с / запрошенную машину")
    axes[0].set_title("Новые сценарии: среднее по 4 seeds")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].plot(
        history.episode,
        history.delay_per_requested_vehicle_s,
        color="#00876c",
        label="Train, с exploration",
    )
    axes[1].plot(
        [v["episode"] for v in validation],
        [v["mean_delay"] for v in validation],
        "o-",
        color="#386cb0",
        label="Validation, greedy",
    )
    axes[1].set_title("30 дополнительных эпизодов")
    axes[1].set_xlabel("Дополнительный эпизод")
    axes[1].set_ylabel("Прокси задержки, с")
    axes[1].legend(fontsize=8)
    red = (
        df.groupby("controller")
        .max_without_green_s.max()
        .reindex(PROTOCOL["controllers"])
    )
    axes[2].barh(np.arange(5), red, color=colors)
    axes[2].set_yticks(
        np.arange(5),
        [
            "Webster",
            "Max Pressure",
            "Исходный D3QN",
            "Только ограничение",
            "С дообучением",
        ],
        fontsize=8,
    )
    axes[2].axvline(120, color="#333", linestyle="--", linewidth=1)
    axes[2].set_title("Самый долгий интервал без зелёного")
    axes[2].set_xlabel("Секунды; максимум по 12 сценариям")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    (OUT / "comparison.png").write_bytes(buf.getvalue())
    plot = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    table = ""
    for sat in PROTOCOL["test_saturations"]:
        for name in PROTOCOL["controllers"]:
            sub = df[(df.saturation == sat) & (df.controller == name)]
            delay = sub[numeric[0]]
            table += f"<tr><td>{sat:.2f}</td><td>{LABELS[name]}</td><td>{delay.mean():.1f} ± {delay.std(ddof=1):.1f}</td><td>{sub.throughput.mean():.0f}</td><td>{sub.max_without_green_s.max():.0f}</td><td>{sub.conflicts_per_1000_arrivals.mean():.1f}</td><td>{sub.pending_insertion.mean():.0f}</td></tr>"
    valtable = "".join(
        f"<tr><td>{v['episode']}</td><td>{v['mean_delay']:.2f}</td></tr>"
        for v in validation
    )
    summary = {
        "completed_additional_episodes": len(history),
        "completed_validation_runs": 12,
        "completed_new_test_runs": len(df),
        "selected_additional_episode": selection["episode"],
        "maximum_guarded_red_s": float(guarded.max_without_green_s.max()),
        "zero_teleports_and_collisions": True,
        "paired_comparisons": paired,
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ["torch", "numpy", "libsumo", "eclipse-sumo"]
        },
    }
    save_json(OUT / "validation.json", summary)
    findings = ""
    for sat in PROTOCOL["test_saturations"]:
        learned = means.loc[("d3qn_guard_trained", sat), numeric[0]]
        original = means.loc[("d3qn_original", sat), numeric[0]]
        guard = means.loc[("d3qn_guard_only", sat), numeric[0]]
        base = min(
            ["webster", "max_pressure"],
            key=lambda name: means.loc[(name, sat), numeric[0]],
        )
        base_delay = means.loc[(base, sat), numeric[0]]
        findings += f"<p><strong>Нагрузка {sat:.2f}:</strong> исходный D3QN {original:.1f} с → дообученный {learned:.1f} с ({100 * (learned / original - 1):+.1f}%). Только ограничение: {guard:.1f} с. Лучший из двух базовых методов — {LABELS[base]}: {base_delay:.1f} с.</p>"
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATSC — дообучение и ограничение ожидания</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f6f7;color:#183039}}main{{max-width:1200px;margin:auto;padding:38px 24px}}h1{{font-size:36px;line-height:1.2;max-width:900px}}h2{{font-size:23px;margin-top:32px}}.tag{{color:#00876c;font-size:13px;letter-spacing:.1em}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.card{{padding:18px 24px;background:white;border:1px solid #d9e4e5;border-radius:12px;flex:1;min-width:180px}}.card b{{display:block;font-size:32px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dde4e6}}th{{background:#e8f0f1}}.scroll{{overflow:auto}}img{{width:100%;background:white;border-radius:12px}}p{{max-width:1000px}}a{{color:#007d68}}code{{background:#e5eeee;padding:2px 5px}}.note{{border-left:4px solid #dd9d37;padding:12px 18px;background:#fff6e6}}</style>
<main><span class="tag">SUMO · ПРОДОЛЖЕНИЕ ЭКСПЕРИМЕНТА</span>
<h1>Контроль ожидания и ещё 30 эпизодов D3QN</h1>
<p>Добавлены четыре таймера времени без зелёного. Недопустимые действия исключены и при выборе фазы, и при вычислении Double-DQN target. Ограничение 120 секунд учитывает жёлтые интервалы и резерв времени для остальных подходов.</p>
<div class="cards"><div class="card">Реально выполнено<b>30</b>дополнительных эпизодов обучения</div><div class="card">Новая оценка<b>60</b>парных прогонов: 5 методов × 4 seeds × 3 нагрузки</div><div class="card">С ограничением<b>{guarded.max_without_green_s.max():.0f} с</b>максимум времени без зелёного на тестах</div></div>
<h2>Результаты</h2><p>Меньшая задержка лучше. Дообучение не даёт универсального выигрыша: при малой нагрузке результат ухудшается; лучшая базовая политика остаётся быстрее на каждом из трёх уровней нагрузки в этом тесте.</p>{findings}<img src="data:image/png;base64,{plot}" alt="Сравнение контроллеров, обучение и максимальное ожидание зелёного">
<div class="scroll"><table><tr><th>Нагрузка</th><th>Контроллер</th><th>Задержка, с: среднее ± SD</th><th>Выехало, среднее</th><th>Без зелёного, max с</th><th>TTC / 1000 выездов</th><th>Ожидает въезда</th></tr>{table}</table></div>
<p>SD — стандартное отклонение по четырём сценариям. Каждая строка сравнивается на тех же seeds. «Только ограничение» использует исходные веса без обучения; этот контроль отделяет эффект ограничения от эффекта дообучения.</p>
<h2>Выбор модели без настройки по тесту</h2>
<p>Выбран checkpoint после <strong>{selection["episode"]}</strong> дополнительных эпизодов: средняя validation-задержка {selection["validation_mean_delay"]:.2f} с. Проверены только заранее заданные checkpoints 0/10/20/30 на трёх сценариях seed 500–502 при нагрузке 0.7. Финальные seeds 1000–1003 не использованы для выбора весов. Это новые реализации дорожного спроса в прежнем временном test-сегменте PeMS, а не новый независимый реальный датасет.</p>
<table style="max-width:480px"><tr><th>Дополнительный эпизод</th><th>Validation-задержка, с</th></tr>{valtable}</table>
<h2>Что изменилось в обучении</h2><p>Исходный efficiency-only D3QN с прогнозом перенесён с 52 на 56 входов. Новые столбцы весов инициализированы нулями; остальные веса сохранены. Оптимизатор и replay buffer для нового этапа созданы заново. Затем выполнены 30 новых эпизодов при нагрузке 0.7, seed обучения 0, дорожные seeds 10000–10029. Epsilon снижен с 0.30 до 0.05. Остались Adam 1e−4, batch 32, target sync 500 и discount с учётом длительности действия. Ограничение не заменяет модель: среди допустимых действий фазу выбирает D3QN.</p>
<p>Сохранены финальные обученные веса, выбранный validation-checkpoint, четыре промежуточных кандидата и <code>resume.pt</code>: optimizer, target network, replay buffer и RNG. Возобновление возможно между завершёнными эпизодами. Во время незавершённого эпизода процесс начнёт этот эпизод заново.</p>
<h2>Границы вывода</h2><p class="note">Это пилот на одном синтетическом перекрёстке и одном seed обучения. Четыре seeds тестового спроса недостаточны для вывода об общей статистической устойчивости. Ограничение максимального ожидания не является доказательством безопасности движения. Новое обучение оптимизирует задержку (w=1); TTC только измеряется. Нулевые столкновения в выполненных прогонах не доказывают безопасность.</p>
<p>Задержка остаётся интегральным прокси потери скорости и ожидания вставки в сеть, делённым на запрошенный спрос. Это не полное время поездки минус free-flow для завершившейся когорты. При нагрузке 0.95 возможна перегрузка; короткое ожидание зелёного само по себе не устраняет транспортный затор. Все ограничения геометрии, датчиков и переноса данных из первоначального отчёта сохраняются.</p>
<h2>Файлы и воспроизведение</h2><p><a href="REPLAY_GUARDED.html">Открыть запись нового агента</a> · <a href="RESULTS.html">Первоначальный пилот</a> · <a href="README_RU.md">Инструкции</a></p>
<p>Исходный код: <code>atsc/guarded.py</code>, <code>atsc/continuation.py</code>. Результаты: <code>artifacts/continuation/</code>. Таблицы: <code>raw_results.csv</code>, <code>summary.csv</code>, <code>paired_differences.json</code>. Восстановить отчёт: <code>python -m atsc.continuation_report</code>.</p></main></html>"""
    (ROOT / "CONTINUATION.html").write_text(html, encoding="utf-8")
    replay = json.loads((OUT / "replay.json").read_text())
    original_row = df[
        (df.controller == "d3qn_guard_trained")
        & (df.seed == replay["seed"])
        & (df.saturation == replay["saturation"])
    ].iloc[0]
    for key in [
        "integrated_delay_vehicle_s",
        "throughput",
        "conflicts",
        "max_without_green_s",
    ]:
        assert np.isclose(original_row[key], replay["metrics"][key]), key
    guarded_template = (
        TEMPLATE.replace(
            "<th>Прогноз*</th>", "<th>Прогноз*</th><th>Без зелёного, с</th>"
        )
        .replace(
            "<td>${a.forecast[i].toFixed(1)}</td>",
            "<td>${a.forecast[i].toFixed(1)}</td><td>${a.red_age[i].toFixed(0)}</td>",
        )
        .replace(
            "Ограничения: минимум 10 с зелёного, 3 с жёлтого.",
            "Ограничения: до 120 с без зелёного; минимум 10 с зелёного, 3 с жёлтого.",
        )
    )
    (ROOT / "REPLAY_GUARDED.html").write_text(
        guarded_template.replace(
            "__DATA__", json.dumps(replay, separators=(",", ":"))
        ).replace("RESULTS.html и README_RU.md", "CONTINUATION.html и README_RU.md"),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "paired_comparisons"}, indent=2
        )
    )


if __name__ == "__main__":
    build()
