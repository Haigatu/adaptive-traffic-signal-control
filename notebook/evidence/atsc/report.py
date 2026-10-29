"""Generate an evidence-based report from completed JSON artifacts only."""

import base64, html, json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ROOT
from .experiment import summarize

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.titlesize": 12,
    }
)
NAMES = {
    "webster": "Webster",
    "max_pressure": "Max Pressure",
    "forecast_rule": "Forecast rule",
    "d3qn_no_forecast": "D3QN без прогноза",
    "d3qn_forecast": "D3QN + прогноз",
    "d3qn_safety": "D3QN + прогноз + safety",
}


def embedded(fig, name):
    dest = ROOT / "artifacts" / name
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (
        '<img alt="'
        + name
        + '" src="data:image/png;base64,'
        + base64.b64encode(dest.read_bytes()).decode()
        + '">'
    )


def build():
    df = summarize()
    forecast = []
    for ds in ["PEMS04", "PEMS08"]:
        for p in (ROOT / "artifacts" / "forecast" / ds).glob("*.json"):
            row = json.loads(p.read_text())
            forecast.append(
                {
                    "Датасет": ds,
                    "Модель": row["model"],
                    "MAE": row["mae"],
                    "RMSE": row["rmse"],
                    "MAPE* %": row["mape_nonzero_percent"],
                }
            )
    ft = pd.DataFrame(forecast).sort_values(["Датасет", "MAE"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axes, ["PEMS04", "PEMS08"]):
        sub = ft[ft["Датасет"] == ds]
        colors = ["#16a085" if x == "tgcn" else "#58779c" for x in sub["Модель"]]
        ax.bar(sub["Модель"], sub.MAE, color=colors)
        ax.set(title=f"{ds}: ошибка прогноза", ylabel="MAE, машин / 5 минут")
        ax.tick_params(axis="x", rotation=30)
    forecast_chart = embedded(fig, "forecast_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    completed = {}
    for name in ["d3qn_no_forecast", "d3qn_forecast", "d3qn_safety"]:
        p = ROOT / "artifacts" / "control" / f"{name}_seed0.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        completed[name] = len(rows)
        x = [r["episode"] for r in rows]
        y = [r["delay_per_requested_vehicle_s"] for r in rows]
        axes[0].plot(x, y, label=NAMES[name], alpha=0.8)
        axes[1].plot(
            x,
            [r["conflicts_per_1000_arrivals"] for r in rows],
            label=NAMES[name],
            alpha=0.8,
        )
    axes[0].set(
        title="Обучение: разные сценарии",
        xlabel="Эпизод",
        ylabel="Интегральная задержка / спрос, с",
    )
    axes[1].set(
        title="TTC-события при обучении",
        xlabel="Эпизод",
        ylabel="На 1 000 выехавших машин",
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    training_chart = embedded(fig, "training_curves.png")
    evaluation = ""
    eval_chart = ""
    verdict = []
    for ds in ["PEMS04", "PEMS08"]:
        d = ft[ft["Датасет"] == ds].set_index("Модель")
        passes = d.loc["tgcn", "MAE"] < min(d.loc["gru", "MAE"], d.loc["lstm", "MAE"])
        verdict.append(
            f"{ds}: преимущество T-GCN над обеими рекуррентными baseline "
            + ("получено в этом запуске." if passes else "не получено в этом запуске.")
        )
    if df is not None:
        rows = []
        for (name, load), g in df.groupby(["controller", "saturation"]):

            def ms(col):
                return f"{g[col].mean():.1f} ± {g[col].std():.1f}"

            rows.append(
                {
                    "Контроллер": NAMES.get(name, name),
                    "Нагрузка": load,
                    "N": len(g),
                    "Задержка / спрос, с": ms("delay_per_requested_vehicle_s"),
                    "Средняя очередь": ms("mean_queue_vehicles"),
                    "Выехало": ms("throughput"),
                    "TTC / 1000": ms("conflicts_per_1000_arrivals"),
                    "Ждут въезда": ms("pending_insertion"),
                }
            )
        evaluation = pd.DataFrame(rows).to_html(index=False, escape=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        names = list(NAMES)
        for i, name in enumerate(names):
            sub = df[df.controller == name]
            g = sub.groupby("saturation")
            means = g["delay_per_requested_vehicle_s"].mean()
            std = g["delay_per_requested_vehicle_s"].std()
            if len(means):
                axes[0].errorbar(
                    means.index,
                    means.values,
                    yerr=std.values,
                    label=NAMES[name],
                    marker="o",
                    capsize=3,
                )
            means = g["conflicts_per_1000_arrivals"].mean()
            std = g["conflicts_per_1000_arrivals"].std()
            if len(means):
                axes[1].errorbar(
                    means.index,
                    means.values,
                    yerr=std.values,
                    label=NAMES[name],
                    marker="o",
                    capsize=3,
                )
        axes[0].set(
            title="Отложенные сценарии: эффективность",
            xlabel="Номинальная нагрузка",
            ylabel="Интегральная задержка / спрос, с",
        )
        axes[1].set(
            title="Отложенные сценарии: TTC",
            xlabel="Номинальная нагрузка",
            ylabel="Событий / 1 000 выехавших",
        )
        axes[1].legend(fontsize=7)
        eval_chart = embedded(fig, "control_comparison.png")
        for load in [0.7, 0.95]:
            sub = (
                df[df.saturation == load]
                .groupby("controller")["delay_per_requested_vehicle_s"]
                .mean()
            )
            if {"d3qn_forecast", "max_pressure", "d3qn_no_forecast"} <= set(sub.index):
                verdict.append(
                    f"Нагрузка {load}: D3QN + прогноз {sub['d3qn_forecast']:.1f} с; без прогноза {sub['d3qn_no_forecast']:.1f} с; Max Pressure {sub['max_pressure']:.1f} с. Это метрика пилота, а не завершённая проверка формулы задержки из постера."
                )
    sfile = ROOT / "artifacts" / "evaluation" / "safety_selection.json"
    safety = json.loads(sfile.read_text()) if sfile.exists() else None
    safety_table = ""
    if safety:
        safety_table = (
            pd.DataFrame(
                [
                    {"Вес эффективности": 1.0, **safety["reference"]},
                    {"Вес эффективности": 0.5, **safety["safety_candidate"]},
                ]
            )
            .rename(
                columns={
                    "delay_per_requested_vehicle_s": "Задержка / спрос, с",
                    "conflicts_per_1000_arrivals": "TTC-событий / 1000",
                }
            )
            .to_html(index=False, float_format=lambda x: f"{x:.2f}")
        )
        verdict.append(
            "Safety-кандидат w=0.5 "
            + ("прошёл" if safety["passes_delay_and_conflict_gate"] else "не прошёл")
            + " tuning-критерий: меньше TTC при росте задержки максимум на 5%. Выбран w="
            + str(safety["selected_weight"])
            + "."
        )
    testsfile = ROOT / "artifacts" / "evaluation" / "paired_tests.json"
    tests = (
        pd.DataFrame(json.loads(testsfile.read_text()))
        if testsfile.exists()
        else pd.DataFrame()
    )
    counts = len(df) if df is not None else 0
    sim = json.loads(
        (ROOT / "artifacts" / "forecast" / "simulation_tgcn.json").read_text()
    )
    body = f"""<span class="tag">ВЫПОЛНЕННЫЙ ЭКСПЕРИМЕНТ · CPU · ПИЛОТ</span><h1>Adaptive Traffic Signal Control</h1>
<p class="lead">Рабочая симуляция SUMO, обученные прогнозаторы и агенты. Все числа получены из сохранённых запусков.</p>
<div class="cards"><div><b>2</b>реальных PeMS датасета</div><div><b>8</b>нейросетевых прогнозаторов</div><div><b>{sum(completed.values())}</b>эпизодов RL суммарно</div><div><b>{counts}</b>отложенных прогонов управления</div></div>
<h2>Что получилось</h2><ul>{"".join("<li>" + html.escape(v) + "</li>" for v in verdict)}</ul>
<p>Обучение каждого агента: {html.escape(str(completed))}. Один training seed на модель. Seeds проверки меняют спрос, но не обученную политику.</p>
<h2>Прогнозирование</h2><p>Вход — час истории; выход — следующие 15 минут. Все датчики, хронологическое 60/20/20, шесть эпох, train/validation stride 3, test stride 1. Checkpoint выбирается по validation, тест выполняется после выбора.</p>
{forecast_chart}{ft.to_html(index=False, float_format=lambda x: f"{x:.3f}")}
<p class="note">* MAPE вычислен только там, где фактический поток больше нуля. Значения MAE/RMSE включают нулевой поток. Источник: открытый benchmark ASTGNN; происхождение и SHA-256 файлов сохранены в data/manifest.json.</p>
<p>После переноса на четыре входящих подхода и дообучения по реальным показаниям виртуальных датчиков SUMO: MAE T-GCN {sim["tgcn"]["mae"]:.2f}; persistence {sim["persistence"]["mae"]:.2f}. Train/validation/test: {sim["training_windows"]}/{sim["validation_windows"]}/{sim["test_windows"]} окон из отдельных сценариев.</p>
<h2>Обучение управления</h2>{training_chart}<p>Графики отражают разные сценарии и уменьшающуюся exploration. Они не доказывают сходимость. D3QN: Double Q-learning, dueling architecture, prioritized replay, 256–256 hidden, gamma 0.99, batch 32.</p>
<h2>Отложенная оценка</h2>{eval_chart}<div class="scroll">{evaluation}</div>
<p>Таблица: среднее ± стандартное отклонение по фактически выполненным seeds. По каждому контроллеру и нагрузке показано N. Полный журнал — artifacts/evaluation/raw_results.csv.</p>
<h2>Как определены задержка и безопасность</h2><p><b>Задержка / спрос</b> — накопленная потеря времени из-за снижения скорости плюс ожидание снаружи сети за час, делённые на число запланированных машин за этот час. Это конечное окно с незавершёнными поездками; метрика отличается от точной средней полной задержки travel time − free-flow time в постере. Результаты не следует выдавать за полностью выполненный RQ2.</p>
<p><b>TTC-событие</b> — вход пары машин на одной полосе в диапазон 0 &lt; TTC &lt; 1.5 с. Повторный вход считается новым эпизодом. Нормирование на 1000 покинувших сеть машин. Это surrogate задних конфликтов; не реальные аварии. Пешеходы и боковые конфликты отсутствуют.</p>
<h2>Выбор safety-веса на tuning-сценариях</h2>{safety_table}<p>Десять парных сценариев со средней нагрузкой, seeds 100–109. Порог допуска: меньше TTC-событий при росте задержки максимум на 5%. Отклонённый кандидат сохраняется для анализа; финальные тесты проводятся с допущенным весом. Полного перебора остальных весов нет.</p>
<h2>Парные сравнения</h2><div class="scroll">{tests.to_html(index=False, float_format=lambda x: f"{x:.5f}") if len(tests) else "Недостаточно завершённых парных прогонов."}</div><p class="note">Wilcoxon two-sided, exploratory p-values, без коррекции множественных сравнений. Сценарные seeds для одной обученной политики не заменяют независимые training seeds.</p>
<h2>Чего этот результат не доказывает</h2><ul><li>Работу на реальном перекрёстке Астаны или на сети 3×3: они не реализованы в этом пилоте.</li><li>Устойчивость по десяти независимым обучениям: выполнен один training seed.</li><li>Оптимальность после 200 эпизодов: выполнен сокращённый запуск.</li><li>Преимущество всех предложенных компонентов: вывод делается только по измеренным результатам.</li><li>Снижение реальных ДТП или готовность к работе на дороге.</li></ul>
<p>Сценарии построены на калифорнийских профилях. Нагрузки 0.4/0.7/0.95 относительны к принятой пропускной способности; фактическая мощность может отличаться. При перегрузке растут незавершённые поездки и ожидание въезда. Полный safety sweep не выполнялся.</p>
<h2>Запуск и проверка</h2><p>Откройте <a href="REPLAY.html">REPLAY.html</a> для записи движения или README_RU.md для команд воспроизведения. Проверки: хронологические окна без утечки, перестановочная эквивариантность T-GCN, размеры прогнозов, prioritized replay и реальные временные ограничения SUMO.</p>"""
    doc = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATSC — результаты обучения</title><style>body{background:#f3f6f9;color:#172839;font:16px/1.65 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:40px 24px}h1{font-size:38px;line-height:1.2}h2{margin-top:40px}.lead{font-size:19px;color:#526477}.tag{font-size:12px;letter-spacing:.08em;color:#087e70}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.cards div{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 6px #162a4510}.cards b{display:block;font-size:34px;color:#087e70}img{max-width:100%;border-radius:10px;background:white}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:10px;border-bottom:1px solid #e4e9ef;text-align:left}th{background:#e5edf4}.scroll{overflow:auto}.note{font-size:13px;color:#627486}a{color:#087e70}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}h1{font-size:29px}}</style><main>"""
        + body
        + "</main></html>"
    )
    (ROOT / "RESULTS.html").write_text(doc, encoding="utf-8")
    return {
        "forecast": ft.to_dict("records"),
        "episodes": completed,
        "evaluation_runs": counts,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
